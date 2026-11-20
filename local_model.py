"""Shared Qwen loading, VRAM accounting, adapter discovery and text generation."""

import logging
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread

# Initialize Hugging Face settings before importing its stack.
import config as cfg

# isort: split

import torch
from filelock import FileLock, Timeout
from peft import PeftConfig, PeftModel
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

LOG = logging.getLogger(__name__)
MIB = 1024**2
OOM_MESSAGE = (
    "Досягнуто доступного ліміту VRAM. Закрийте інші GPU-процеси, скоротіть контекст або "
    "перейдіть на Qwen/Qwen3-4B. Для навчання зменште --seq-length."
)


def attention_kernels():
    # PyTorch on Windows can otherwise select the math kernel for Qwen's
    # grouped-query attention. At 4096 tokens that allocates a 2 GiB score matrix.
    return sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
         SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        set_priority=True,
    )


def claim_gpu() -> FileLock:
    cfg.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cfg.RUNTIME_DIR / "gpu.lock"), thread_local=False)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(
            "GPU вже зайнята іншим процесом проєкту. Закрийте тренування або "
            "сервер. Для одночасного доступу до API та TerroAI запустіть api_server.py --with-ui."
        ) from None
    return lock


def cuda_budget() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Перевірте драйвер NVIDIA та колесо PyTorch з CUDA.")
    torch.cuda.set_device(cfg.CUDA_DEVICE)
    free, total = torch.cuda.mem_get_info(cfg.CUDA_DEVICE)
    budget = min(free - cfg.VRAM_HEADROOM_MIB * MIB, int(total * cfg.VRAM_ALLOCATOR_FRACTION))
    if budget <= 0:
        raise RuntimeError(OOM_MESSAGE)
    torch.cuda.set_per_process_memory_fraction(budget / total, cfg.CUDA_DEVICE)
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if cfg.PRECISION not in {"auto", "bf16", "fp16"}:
        raise ValueError("PRECISION має бути 'auto', 'bf16' або 'fp16'.")
    if cfg.PRECISION == "bf16" and not native_bf16:
        raise RuntimeError("GPU не підтримує bf16. Встановіть PRECISION = 'auto' або 'fp16'.")
    use_bf16 = cfg.PRECISION == "bf16" or (cfg.PRECISION == "auto" and native_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"CUDA: {free / MIB:.0f}/{total / MIB:.0f} MiB free, "
          f"allocator limit {budget / MIB:.0f} MiB, {dtype}", flush=True)
    return dtype


def vram_report(prefix: str) -> None:
    device = cfg.CUDA_DEVICE
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"{prefix} | CUDA зайнято на GPU {(total - free) / MIB:.0f}/{total / MIB:.0f} MiB, "
        f"PyTorch allocated {torch.cuda.memory_allocated(device) / MIB:.0f} MiB, "
        f"reserved {torch.cuda.memory_reserved(device) / MIB:.0f} MiB, "
        f"peak allocated {torch.cuda.max_memory_allocated(device) / MIB:.0f} MiB, "
        f"peak reserved {torch.cuda.max_memory_reserved(device) / MIB:.0f} MiB",
        flush=True,
    )


def load_base(dtype: torch.dtype):
    # NF4 stores linear layers in 4 bits. Double quantization compresses the
    # scales as well. Matmuls and unquantized layers use bf16/fp16.
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    return AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID,
        **cfg.model_files(),
        use_safetensors=True,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": cfg.CUDA_DEVICE},
        attn_implementation="sdpa",
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, **cfg.model_files())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def supports_thinking(tokenizer) -> bool:
    return "enable_thinking" in (tokenizer.chat_template or "")


def chat_tokens(
    tokenizer, messages: list[dict], *, continue_answer: bool = False, thinking: bool = cfg.ENABLE_THINKING,
) -> list[int]:
    # Use the model's own ChatML template, including its assistant header and,
    # for Qwen3 with thinking disabled, the prefilled empty reasoning block.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=not continue_answer,
        continue_final_message=continue_answer,
        enable_thinking=thinking,
    )


def active_adapter():
    pointer = cfg.ADAPTER_ROOT / "current.txt"
    if pointer.is_file():
        version = pointer.read_text(encoding="utf-8").strip()
        if not version or not version.isascii() or not version.replace("-", "").isalnum():
            raise ValueError(f"Некоректна версія адаптера у {pointer}.")
        found = cfg.ADAPTER_ROOT / "versions" / version
        if not (found / "adapter_config.json").is_file():
            raise ValueError(f"Версію адаптера {version} не знайдено.")
        return found
    if (cfg.ADAPTER_ROOT / "adapter_config.json").is_file():
        return cfg.ADAPTER_ROOT
    return None


def attach_adapter(base, path, *, trainable: bool):
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"У {path} немає конфігурації PEFT-адаптера.")
    settings = PeftConfig.from_pretrained(path)
    if settings.peft_type != "LORA" or settings.task_type != "CAUSAL_LM":
        raise ValueError(f"{path} має містити LoRA для CAUSAL_LM.")
    if settings.base_model_name_or_path != cfg.MODEL_ID:
        raise ValueError(f"{path} створено для {settings.base_model_name_or_path}, а не {cfg.MODEL_ID}.")
    if settings.revision not in (None, cfg.MODEL_REVISION):
        raise ValueError(f"{path} створено для ревізії {settings.revision}.")
    return PeftModel.from_pretrained(base, path, is_trainable=trainable)


@dataclass(frozen=True)
class Generation:
    input_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    adapter: bool
    stop: tuple[str, ...]
    thinking: bool = False


def visible_fragments(stream, thinking: bool):
    """Suppress Qwen's optional leading reasoning block, including split tags."""
    stream = iter(stream)
    if not thinking:
        yield from stream
        return
    opening, closing = "<think>", "</think>"
    pending = ""
    inside = False
    for fragment in stream:
        if fragment is None:
            yield None
            continue
        if not fragment:
            continue
        pending += fragment
        if inside:
            end = pending.find(closing)
            if end >= 0:
                yield pending[end + len(closing):].lstrip("\n")
                break
            pending = pending[-(len(closing) - 1):]
        else:
            text = pending.lstrip()
            if text.startswith(opening):
                inside = True
                pending = text[len(opening):]
                if closing in pending:
                    yield pending.split(closing, 1)[1].lstrip("\n")
                    break
            elif not opening.startswith(text):
                yield pending
                break
    else:
        return
    yield from stream


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class StopOnCancel(StoppingCriteria):
    def __init__(self, event: Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class LocalQwen:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.gpu_lock = None
        self.tokenizer_lock = Lock()
        self.model_lock = Lock()

    def prepare(
        self,
        messages: list[dict],
        *,
        adapter: bool = False,
        max_tokens: int | None = None,
        temperature: float = cfg.TEMPERATURE,
        top_p: float = cfg.TOP_P,
        stop: tuple[str, ...] = (),
        trim_history: bool = False,
        context_limit: int = cfg.MAX_SEQ_LENGTH,
        continue_answer: bool = False,
        thinking: bool = cfg.ENABLE_THINKING,
    ) -> Generation:
        context_limit = min(context_limit, cfg.MAX_SEQ_LENGTH)
        if adapter and active_adapter() is None:
            raise ValueError("Адаптер ще не створено. Запустіть train_lora.py або виберіть базову модель.")
        with self.tokenizer_lock:
            if self.tokenizer is None:
                self.tokenizer = load_tokenizer()
            thinking = thinking and not continue_answer and supports_thinking(self.tokenizer)
            ids = chat_tokens(self.tokenizer, messages, continue_answer=continue_answer, thinking=thinking)
            if trim_history:
                budget = context_limit - (max_tokens or cfg.MAX_NEW_TOKENS)
                # The chat supplies [system, (user, assistant)*, current user].
                # Drop whole turns so an answer never loses its question.
                removable = (len(messages) - 2 - int(continue_answer)) // 2
                if len(ids) > budget and removable > 0:
                    first, last = 1, removable
                    while first < last:
                        dropped = (first + last) // 2
                        suffix = chat_tokens(
                            self.tokenizer, messages[:1] + messages[1 + 2 * dropped:],
                            continue_answer=continue_answer, thinking=thinking,
                        )
                        if len(suffix) > budget:
                            first = dropped + 1
                        else:
                            last = dropped
                    ids = chat_tokens(
                        self.tokenizer, messages[:1] + messages[1 + 2 * first:],
                        continue_answer=continue_answer, thinking=thinking,
                    )
        remaining = context_limit - len(ids)
        if remaining < 1 or (max_tokens is not None and remaining < max_tokens):
            if trim_history:
                raise ValueError("Повідомлення завелике для контексту чату. Скоротіть його або надішліть частинами.")
            raise ValueError(
                f"Запит має {len(ids)} токенів, загальний ліміт {context_limit}. "
                f"Доступно для відповіді: {max(0, remaining)}. Скоротіть код, історію або max_tokens."
            )
        tokens = min(cfg.MAX_NEW_TOKENS, remaining) if max_tokens is None else max_tokens
        return Generation(ids, tokens, temperature, top_p, adapter, stop, thinking)

    def _get_model(self, use_adapter: bool):
        if self.model is None:
            self.gpu_lock = claim_gpu()
            try:
                self.model = load_base(cuda_budget()).eval()
            except BaseException:
                self.gpu_lock.release()
                raise
        if use_adapter and not isinstance(self.model, PeftModel):
            path = active_adapter()
            if path is None:
                raise ValueError("Активний адаптер не знайдено.")
            self.model = attach_adapter(self.model, path, trainable=False).eval()
        return self.model

    def stream(self, request: Generation, *, heartbeat: bool = False) -> Iterator[str | GenerationStats | None]:
        # One shared quantized checkpoint. PEFT's adapter switches mutate model
        # state, so a single worker may perform generation at any one time.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.25,
            clean_up_tokenization_spaces=False,
        )
        cancel = Event()
        outcome = Queue(maxsize=1)

        def run():
            try:
                with self.model_lock, torch.inference_mode():
                    if cancel.is_set():
                        return
                    model = self._get_model(request.adapter)
                    if cancel.is_set():
                        return
                    scope = (
                        model.disable_adapter() if isinstance(model, PeftModel) and not request.adapter
                        else nullcontext()
                    )
                    device = torch.device("cuda", cfg.CUDA_DEVICE)
                    ids = torch.tensor([request.input_ids], device=device)
                    # Zero temperature explicitly enables greedy decoding. Sampling
                    # knobs are passed only for positive temperatures.
                    sample = request.temperature > 0
                    options = (
                        {"temperature": request.temperature, "top_p": request.top_p, "top_k": cfg.TOP_K}
                        if sample else {"temperature": None, "top_p": None, "top_k": None}
                    )
                    long_context = len(request.input_ids) + request.max_tokens >= cfg.INFERENCE_OFFLOAD_MIN_TOKENS
                    cache_options = (
                        {"cache_implementation": "offloaded", "prefill_chunk_size": cfg.PREFILL_CHUNK_TOKENS}
                        if long_context else {}
                    )
                    with scope, attention_kernels():
                        result = model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            max_new_tokens=request.max_tokens,
                            do_sample=sample,
                            repetition_penalty=cfg.REPETITION_PENALTY,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=True,
                            logits_to_keep=1,
                            streamer=streamer,
                            stopping_criteria=StoppingCriteriaList([StopOnCancel(cancel)]),
                            return_dict_in_generate=False,
                            **cache_options,
                            **options,
                        )
                    count = result.shape[-1] - ids.shape[-1]
                    eos = model.generation_config.eos_token_id
                    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
                    reason = "stop" if result[0, -1].item() in eos_ids else "length"
                    outcome.put(GenerationStats(ids.shape[-1], count, reason))
            except torch.cuda.OutOfMemoryError:
                outcome.put(RuntimeError(OOM_MESSAGE))
            except Exception as exc:
                LOG.exception("Qwen generation failed")
                outcome.put(RuntimeError(str(exc)))
            finally:
                # generate() ends the stream on success. Also wake the reader on
                # errors and on cancellation before any tokens were generated.
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=run, name="qwen-generate", daemon=True)
        thread.start()
        pending = ""
        stopped = False
        keep = max((len(item) - 1 for item in request.stop), default=0)
        def fragments():
            while True:
                try:
                    yield next(streamer)
                except Empty:
                    if thread.is_alive():
                        if heartbeat:
                            yield None
                        continue
                    return
                except StopIteration:
                    return

        try:
            for fragment in visible_fragments(fragments(), request.thinking):
                if fragment is None:
                    yield None
                    continue
                pending += fragment
                positions = [i for stop in request.stop if (i := pending.find(stop)) >= 0]
                if positions:
                    cancel.set()
                    stopped = True
                    if text := pending[:min(positions)]:
                        yield text
                    pending = ""
                    break
                ready = max(0, len(pending) - keep)
                if ready:
                    yield pending[:ready]
                    pending = pending[ready:]
            if pending:
                yield pending
            thread.join()
            result = outcome.get_nowait()
            if isinstance(result, Exception):
                raise result
            yield GenerationStats(
                result.prompt_tokens, result.completion_tokens, "stop" if stopped else result.finish_reason,
            )
        finally:
            cancel.set()
