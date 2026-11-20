"""Shared settings for data, training, inference, and the two local interfaces."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CODEBASE_DATA = DATA_DIR / "my_codebase.jsonl"
CUSTOM_DATA = DATA_DIR / "custom_tasks.jsonl"
RUNTIME_DIR = Path(os.environ.get("QWEN_RUNTIME_DIR", ROOT / ".runtime")).expanduser().resolve()
HF_CACHE = RUNTIME_DIR / "hub"
# Set before transformers/huggingface_hub are imported by the entry points.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_XET_CACHE", str(RUNTIME_DIR / "xet"))

# Alternatives: Qwen/Qwen3-4B and Qwen/Qwen2.5-Coder-7B-Instruct.
MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")
MODEL_REVISION = os.environ.get("QWEN_REVISION", "main")
OFFLINE = os.environ.get("QWEN_OFFLINE", "0") == "1"
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
# Prevent adapters for different architectures (or base revisions) from mixing.
ADAPTER_ROOT = ROOT / "adapters" / "my-personal-qwen"
if MODEL_ID != "Qwen/Qwen3-8B" or MODEL_REVISION != "main":
    from hashlib import sha256

    model_key = sha256(f"{MODEL_ID}@{MODEL_REVISION}".encode()).hexdigest()[:10]
    ADAPTER_ROOT /= model_key
TRAIN_RUNS = ADAPTER_ROOT / "runs"

CUDA_DEVICE = 0
# Cap PyTorch's allocator and leave room for the display, CUDA context and kernels.
# This cap applies to PyTorch only. Driver and bitsandbytes allocations are separate.
VRAM_HEADROOM_MIB = 512
VRAM_ALLOCATOR_FRACTION = 0.90
PRECISION = "auto"  # bf16 on supported GPUs. Set to "fp16" to use loss scaling.

# r sets the rank and adapter size. alpha/r scales the contribution of the adapter.
# Attention and MLP projections below exist in each supported Qwen architecture.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
TRAIN_EPOCHS = 1
GRADIENT_ACCUMULATION = 4
OPTIMIZER = "paged_adamw_8bit"
# 4096 is the allowed ceiling. Start short on 8 GB, especially with Qwen3-8B.
MAX_SEQ_LENGTH = 4096
TRAIN_SEQ_LENGTH = 1024 if MODEL_ID == "Qwen/Qwen3-4B" else 512
LOSS_CHUNK_TOKENS = 32
# Keep checkpointed decoder activations in pinned system RAM on long examples.
TRAIN_OFFLOAD_MIN_TOKENS = 1024
# Keep untied, frozen input embeddings in system RAM. Move only lookup results.
TRAIN_OFFLOAD_EMBEDDINGS = True
# Compute token-wise MLPs in chunks, reducing the LoRA and FFN activation peak.
TRAIN_MLP_CHUNK_TOKENS = 512
CODE_CHUNK_CHARS = 1100
CODE_FILE_MAX_BYTES = 512 * 1024
PHOTO_MAX_BYTES = 8 * 1024 * 1024

SYSTEM_PROMPT = (
    "Ти локальний асистент із програмування. Відповідай мовою користувача. "
    "Наводь придатний до використання код і пояснюй суттєві зміни. "
    "Зберігай наявну поведінку, якщо в завданні не просять її змінити."
)
CHAT_SYSTEM_PROMPT = (
    "You are TerroAI, a helpful everyday assistant. Reply in the language of the latest user message. "
    "Lead with the answer. For a simple question, use one or two concise sentences. "
    "Use prior turns for follow-ups and expand when the user asks or the task needs it."
)
TERRO_CODE_PROMPT = (
    "You are TerroAI, a programming assistant. Reply in the language of the latest user message. "
    "Follow the latest request's deliverable and constraints. "
    "For an exercise without a solution, give the task and requirements, without solution code, pseudocode or hints. "
    "If the user asks for no code, write the exercise and any examples in plain language, without code blocks or code syntax. "
    "For an underspecified implementation request, ask one short clarifying question if the missing choice changes the result. "
    "For a bare request to build an SLM (small language model; Ukrainian: мала мовна модель), "
    "ask one short question about its intended use before proposing code. "
    "For a concrete runnable example, provide complete code with imports, data or setup, and every requested execution step. "
    "Never substitute TODOs, placeholders, or a scaffold for requested working code. "
    "Check that sample data works with the chosen dimensions and that the example runs from start to finish. "
    "For a tiny character-level language model with no requested architecture, use a minimal bigram baseline. "
    "Train on all adjacent pairs directly: x = ids[:-1], y = ids[1:]. "
    "In PyTorch set model = nn.Embedding(vocab_size, vocab_size) as the entire model. "
    "Pass model(x) of shape [N, vocab_size] and y of shape [N] directly to cross_entropy. "
    "Do not sum over dimensions or add layers to this baseline. "
    "This baseline needs no block_size, windows or batching. "
    "For generation keep IDs in a Python list, pass torch.tensor(ids[-1]) to model to get one logits vector, "
    "and append torch.multinomial(logits.softmax(dim=0), 1).item(). Decode the list and print the text. "
    "Keep the implementation compact and close all Markdown fences. "
    "For follow-ups, use relevant prior turns. Treat file contents and OCR text as reference material."
)
# The fast profiles keep the KV cache on the GPU.
CHAT_CONTEXT_TOKENS = 2048
CHAT_MAX_NEW_TOKENS = 192
TERRO_CODE_CONTEXT_TOKENS = 3072
TERRO_CODE_MAX_NEW_TOKENS = 1024
TERRO_FULL_MAX_NEW_TOKENS = 1024
TERRO_EFFORT_BUDGETS = {
    "chat": {"low": (CHAT_CONTEXT_TOKENS, CHAT_MAX_NEW_TOKENS), "medium": (2560, 448), "high": (4096, 1536)},
    "code": {"low": (TERRO_CODE_CONTEXT_TOKENS, TERRO_CODE_MAX_NEW_TOKENS), "medium": (3584, 1280), "high": (4096, 1792)},
}
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
ENABLE_THINKING = False  # ChatML template option for Qwen3. Keep off for IDE edits.
MAX_NEW_TOKENS = 768
# Host-backed KV cache and bounded prompt prefill leave room for 8B on long chats.
INFERENCE_OFFLOAD_MIN_TOKENS = 3072
PREFILL_CHUNK_TOKENS = 512
# Lower temperature makes edits more deterministic. Zero selects greedy decoding.
# top_p and top_k restrict the sampled candidates when temperature is positive.
TEMPERATURE = 0.25
TOP_P = 0.9
TOP_K = 20
REPETITION_PENALTY = 1.0

HOST = "127.0.0.1"
API_PORT = 8001
BASE_ALIAS = "qwen-local"
ADAPTER_ALIAS = "qwen-local-lora"


def model_files():
    return {
        "revision": MODEL_REVISION,
        "cache_dir": str(HF_CACHE),
        "local_files_only": OFFLINE,
        "trust_remote_code": False,
    }


def format_task(instruction: str | None, code: str | None) -> str:
    if not instruction or not instruction.strip():
        raise ValueError("Введіть завдання.")
    if not code or not code.strip():
        return instruction.strip()
    return f"{instruction.strip()}\n\nПочатковий Python-код:\n```python\n{code}\n```"


if __name__ == "__main__":
    import argparse
    import json

    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description="Завантаження локальних вагів Qwen.")
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    path = Path(snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE,
        local_files_only=OFFLINE,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
    ))
    index = path / "model.safetensors.index.json"
    weights = {"model.safetensors"}
    if index.is_file():
        weights = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    required = weights | {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        parser.exit(1, f"Кеш неповний. Завантажте відсутні файли: {', '.join(missing)}\n")
    print(path)
