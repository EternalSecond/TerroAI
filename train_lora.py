"""Train a new adapter or continue from the currently published LoRA weights."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

import config as cfg

# isort: split

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from dataset_manager import read_records
from local_model import (
    OOM_MESSAGE,
    active_adapter,
    attach_adapter,
    attention_kernels,
    chat_tokens,
    claim_gpu,
    cuda_budget,
    load_base,
    load_tokenizer,
    vram_report,
)


def training_dataset(tokenizer, seq_length: int) -> Dataset:
    seen = set()
    examples = []
    skipped = 0
    longest = 0
    for path in sorted(cfg.DATA_DIR.glob("*.jsonl")):
        rows = read_records(path)
        print(f"{path.name}: {len(rows)} прикладів")
        for row in rows:
            pair = row["prompt"], row["response"]
            if pair in seen:
                continue
            seen.add(pair)
            prefix = chat_tokens(tokenizer, [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ])
            # Tokenize the answer separately so that the training prefix is
            # exactly the same token sequence the inference engine receives.
            answer = tokenizer.encode(row["response"], add_special_tokens=False)
            answer.append(tokenizer.eos_token_id)
            length = len(prefix) + len(answer)
            longest = max(longest, length)
            if length > seq_length:
                skipped += 1
                continue
            examples.append({
                "input_ids": prefix + answer,
                "completion_mask": [0] * len(prefix) + [1] * len(answer),
            })
    print(f"Відібрано {len(examples)}, завеликих {skipped}, найдовший запис {longest} токенів.")
    if not examples:
        raise ValueError(
            "У поточний ліміт не вмістився жоден приклад або датасет порожній. "
            "Запустіть dataset_manager.py seed, додайте короткі пари або підніміть --seq-length."
        )
    return Dataset.from_list(examples)


class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        vram_report(f"Крок {state.global_step}")


def chunked_mlp(forward, hidden_states):
    size = cfg.TRAIN_MLP_CHUNK_TOKENS
    if hidden_states.shape[-2] <= size or not torch.is_grad_enabled():
        return forward(hidden_states)
    # A Qwen MLP acts independently on each token. Recompute the small chunks
    # independently during backward so the whole FFN activation is never retained.
    chunks = hidden_states.split(size, dim=-2)
    return torch.cat([checkpoint(forward, chunk, use_reentrant=False) for chunk in chunks], dim=-2)


class QwenSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # A large Qwen vocabulary makes standard [batch, context, vocabulary]
        # logits an avoidable VRAM peak. The single-GPU decoder already contains
        # the PEFT layers. Project only supervised positions in small chunks.
        base = model.get_base_model()
        labels = inputs["labels"][:, 1:]
        ids = inputs["input_ids"]
        offload = ids.is_cuda and ids.shape[-1] >= cfg.TRAIN_OFFLOAD_MIN_TOKENS
        activations = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
        embedding = base.get_input_embeddings()
        if embedding.weight.device == ids.device:
            decoder_input = {"input_ids": ids}
        else:
            # Only transfer the token IDs and the selected vectors. The full
            # embedding matrix never enters the decoder's VRAM working set.
            vectors = embedding(ids.to(embedding.weight.device))
            decoder_input = {"inputs_embeds": vectors.to(ids.device)}
        # Calling the decoder directly bypasses Accelerate's autocast on the
        # outer model forward. Restore it here for the LoRA matmuls.
        with activations, torch.autocast("cuda", dtype=base.lm_head.weight.dtype, enabled=ids.is_cuda):
            hidden = base.model(
                **decoder_input, attention_mask=inputs["attention_mask"],
                use_cache=False, return_dict=True,
            ).last_hidden_state
        valid = labels.ne(-100)
        answer_hidden = hidden[:, :-1, :][valid]
        targets = labels[valid]

        def chunk_loss(states, token_ids):
            return F.cross_entropy(base.lm_head(states).float(), token_ids, reduction="sum")

        # Recompute each small projection during backward. Checkpointing the
        # decoder layers alone would still retain loss activations for all tokens.
        size = cfg.LOSS_CHUNK_TOKENS
        loss = sum(
            checkpoint(chunk_loss, states, token_ids, use_reentrant=False, preserve_rng_state=False)
            for states, token_ids in zip(answer_hidden.split(size), targets.split(size))
        )
        count = targets.numel() if num_items_in_batch is None else num_items_in_batch
        loss = loss / count
        return (loss, {"loss": loss}) if return_outputs else loss


def train(args) -> Path:
    with_lock = claim_gpu()
    try:
        dtype = cuda_budget()
        torch.cuda.reset_peak_memory_stats(cfg.CUDA_DEVICE)
        set_seed(42)
        adapter = Path(args.from_adapter).resolve() if args.from_adapter else None
        if args.resume:
            adapter = active_adapter()
            if adapter is None:
                raise ValueError("Для --resume потрібен уже опублікований адаптер.")

        tokenizer = load_tokenizer()
        dataset = training_dataset(tokenizer, args.seq_length)
        base = load_base(dtype)
        # Freeze the quantized backbone. Non-reentrant checkpointing can compute
        # LoRA gradients with frozen embeddings, which stay in the loading dtype.
        base.requires_grad_(False)
        base.config.use_cache = False
        if args.seq_length > cfg.TRAIN_MLP_CHUNK_TOKENS:
            for layer in base.model.layers:
                layer.mlp.forward = partial(chunked_mlp, layer.mlp.forward)
        if adapter is not None:
            print(f"Продовжую з вагів {adapter}")
            model = attach_adapter(base, adapter, trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(
                r=cfg.LORA_R,
                lora_alpha=cfg.LORA_ALPHA,
                target_modules=cfg.LORA_TARGET_MODULES,
                lora_dropout=cfg.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
                revision=cfg.MODEL_REVISION,
            ))
        embedding = base.get_input_embeddings()
        if (
            cfg.TRAIN_OFFLOAD_EMBEDDINGS
            and not base.config.tie_word_embeddings
            and not any(parameter.requires_grad for parameter in embedding.parameters())
        ):
            embedding.to("cpu")
        model.print_trainable_parameters()
        torch.cuda.empty_cache()
        vram_report("Модель і LoRA завантажено")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version = f"{stamp}-{uuid4().hex[:8]}"
        trainer = QwenSFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            callbacks=[MemoryCallback()],
            args=SFTConfig(
                output_dir=str(cfg.TRAIN_RUNS / version),
                max_length=args.seq_length,
                packing=False,
                completion_only_loss=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                learning_rate=args.learning_rate,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps,
                optim=args.optimizer,
                bf16=dtype == torch.bfloat16,
                fp16=dtype == torch.float16,
                average_tokens_across_devices=False,
                warmup_ratio=0.0 if 0 < args.max_steps < 10 else cfg.WARMUP_RATIO,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                seed=42,
            ),
        )
        trainer.model_accepts_loss_kwargs = True
        with attention_kernels():
            trainer.train()
        vram_report("Після навчання")
        destination = cfg.ADAPTER_ROOT / "versions" / version
        # Publish only after PEFT has written the complete version directory.
        # The previous adapter remains available if training is interrupted.
        model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
        trainer.save_state()
        pending = cfg.ADAPTER_ROOT / f".current-{version}.tmp"
        pending.write_text(version + "\n", encoding="utf-8")
        pending.replace(cfg.ADAPTER_ROOT / "current.txt")
        return destination
    finally:
        with_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", action="store_true", help="Continue from the active adapter")
    source.add_argument("--from-adapter", type=Path, help="Continue from a specific PEFT adapter")
    parser.add_argument("--seq-length", type=int, default=cfg.TRAIN_SEQ_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--epochs", type=float, default=cfg.TRAIN_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--optimizer", choices=["paged_adamw_8bit", "adamw_bnb_8bit"], default=cfg.OPTIMIZER)
    args = parser.parse_args()
    if not 2 <= args.seq_length <= cfg.MAX_SEQ_LENGTH:
        parser.error(f"--seq-length має бути в діапазоні 2..{cfg.MAX_SEQ_LENGTH}.")
    if not args.learning_rate > 0 or not args.epochs > 0 or (args.max_steps != -1 and args.max_steps < 1):
        parser.error("Вкажіть додатні learning-rate, epochs і max-steps або max-steps=-1.")
    try:
        print(f"Опубліковано адаптер: {train(args)}")
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            vram_report("Після OOM")
        parser.exit(1, OOM_MESSAGE + "\n")
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
