"""Local OpenAI-compatible text chat and the optional offline TerroAI workspace."""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, time
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import config as cfg
from dataset_manager import append_tasks, task_record
from local_model import Generation, GenerationStats, LocalQwen, active_adapter

app = FastAPI(title="TerroAI")
engine = LocalQwen()


@app.exception_handler(HTTPException)
async def chat_error(request: Request, exc: HTTPException):
    kind = "server_error" if exc.status_code >= 500 else "invalid_request_error"
    return JSONResponse(
        {"error": {"message": str(exc.detail), "type": kind}},
        status_code=exc.status_code,
        headers=exc.headers,
    )


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]

    def for_template(self) -> dict:
        return {
            "role": "system" if self.role == "developer" else self.role,
            "content": self.content if isinstance(self.content, str) else "".join(p.text for p in self.content),
        }


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    n: Literal[1] = 1
    tools: list[dict] | None = None


@app.get("/v1/models")
def list_models():
    names = [cfg.BASE_ALIAS]
    if active_adapter() is not None:
        names.append(cfg.ADAPTER_ALIAS)
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "created": 0, "owned_by": "local"} for name in names],
    }


def sse(data) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
def completions(body: ChatRequest, request: Request):
    if body.model not in {cfg.BASE_ALIAS, cfg.ADAPTER_ALIAS}:
        raise HTTPException(404, f"Моделі {body.model} немає. Перевірте /v1/models.")
    if body.tools:
        raise HTTPException(400, "Сервер підтримує текстовий чат. Виклики tools не підтримуються.")
    if body.max_completion_tokens is not None and body.max_tokens is not None:
        raise HTTPException(400, "Вкажіть лише max_tokens або max_completion_tokens.")
    stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
    if len(stop) > 4 or any(not item or len(item) > 512 for item in stop):
        raise HTTPException(400, "Вкажіть до чотирьох непорожніх stop-рядків по 512 символів.")
    messages = [message.for_template() for message in body.messages]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": cfg.SYSTEM_PROMPT})
    try:
        generation = engine.prepare(
            messages,
            adapter=body.model == cfg.ADAPTER_ALIAS,
            max_tokens=body.max_completion_tokens or body.max_tokens,
            temperature=cfg.TEMPERATURE if body.temperature is None else body.temperature,
            top_p=cfg.TOP_P if body.top_p is None else body.top_p,
            stop=stop,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc

    metadata = {"id": f"chatcmpl-{uuid4().hex}", "created": int(time()), "model": body.model}
    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return stream_completion(request, generation, metadata, include_usage)

    parts = []
    stats = None
    try:
        for item in engine.stream(generation):
            if isinstance(item, GenerationStats):
                stats = item
            else:
                parts.append(item)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if stats is None:
        raise HTTPException(503, "Генерація завершилася без результату.")
    return {
        **metadata, "object": "chat.completion",
        "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "".join(parts)},
            "finish_reason": stats.finish_reason,
        }],
        "usage": stats.usage(),
    }


def stream_completion(request: Request, generation: Generation, metadata: dict, include_usage: bool):
    def chunk(delta, reason=None):
        value = {
            **metadata, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }
        if include_usage:
            value["usage"] = None
        return sse(value)

    async def events():
        stream = engine.stream(generation, heartbeat=True)
        sentinel = object()
        last_send = monotonic()
        pending = None
        try:
            yield chunk({"role": "assistant", "content": ""})
            while not await request.is_disconnected():
                pending = asyncio.create_task(run_in_threadpool(next, stream, sentinel))
                item = await asyncio.shield(pending)
                if item is sentinel:
                    break
                if item is None:
                    if monotonic() - last_send > 15:
                        yield ": waiting\n\n"
                        last_send = monotonic()
                elif isinstance(item, GenerationStats):
                    yield chunk({}, item.finish_reason)
                    if include_usage:
                        yield sse({
                            **metadata, "object": "chat.completion.chunk", "choices": [], "usage": item.usage(),
                        })
                    break
                else:
                    yield chunk({"content": item})
                    last_send = monotonic()
            yield "data: [DONE]\n\n"
        except RuntimeError as exc:
            yield sse({"error": {"message": str(exc), "type": "server_error"}})
            yield "data: [DONE]\n\n"
        finally:
            if pending is not None and not pending.done():
                # Disconnect can cancel the await before the worker leaves next().
                def close_stream(task):
                    if not task.cancelled():
                        task.exception()
                    stream.close()

                pending.add_done_callback(close_stream)
            else:
                stream.close()

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TerroAIMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=cfg.CODE_FILE_MAX_BYTES * 2)


class TerroAIRequest(BaseModel):
    mode: Literal["chat", "code"]
    messages: list[TerroAIMessage] = Field(min_length=1, max_length=81)
    adapter: bool = False
    effort: Literal["low", "medium", "high"] = "low"
    continuation: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)


class DatasetRequest(BaseModel):
    task: str = Field(min_length=1, max_length=cfg.CODE_FILE_MAX_BYTES)
    source: str = Field(default="", max_length=cfg.CODE_FILE_MAX_BYTES)
    solution: str = Field(min_length=1, max_length=cfg.CODE_FILE_MAX_BYTES)


def terroai_generate(body: TerroAIRequest, request: Request):
    roles = [message.role for message in body.messages]
    if any(role != ("assistant" if i % 2 else "user") for i, role in enumerate(roles)):
        raise HTTPException(400, "Історія має чергувати повідомлення користувача й асистента.")
    if (roles[-1] == "assistant") != body.continuation:
        raise HTTPException(400, "Немає повідомлення для цього режиму генерації.")
    code = body.mode == "code"
    context, limit = cfg.TERRO_EFFORT_BUDGETS[body.mode][body.effort]
    if body.continuation:
        context, limit = cfg.MAX_SEQ_LENGTH, cfg.TERRO_FULL_MAX_NEW_TOKENS
    thinking = body.effort == "high" and not body.continuation
    system = cfg.TERRO_CODE_PROMPT if code else cfg.CHAT_SYSTEM_PROMPT
    if body.effort != "low":
        system += " Cover relevant details and edge cases according to the task's complexity."
    adapter = code and body.adapter
    try:
        generation = engine.prepare(
            [{"role": "system", "content": system}, *(message.model_dump() for message in body.messages)],
            adapter=adapter,
            max_tokens=limit,
            trim_history=True,
            context_limit=context,
            continue_answer=body.continuation,
            thinking=thinking,
            temperature=(cfg.THINKING_TEMPERATURE if thinking else cfg.TEMPERATURE)
            if body.temperature is None else body.temperature,
            top_p=cfg.THINKING_TOP_P if thinking else cfg.TOP_P,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    metadata = {
        "id": f"chatcmpl-{uuid4().hex}", "created": int(time()),
        "model": cfg.ADAPTER_ALIAS if adapter else cfg.BASE_ALIAS,
    }
    return stream_completion(request, generation, metadata, include_usage=True)


def terroai_settings():
    return {
        "model": cfg.MODEL_ID,
        "adapter": active_adapter() is not None,
        "file_limit": cfg.CODE_FILE_MAX_BYTES,
        "photo_limit": cfg.PHOTO_MAX_BYTES,
    }


OCR_FORMATS = {"image/png": ".png", "image/jpeg": ".jpg", "image/bmp": ".bmp"}


def recognize_photo(data: bytes, suffix: str):
    if os.name != "nt":
        raise HTTPException(503, "Локальне розпізнавання фото потребує Windows OCR.")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        with TemporaryDirectory(prefix="terroai-ocr-") as directory:
            photo = Path(directory) / ("photo" + suffix)
            photo.write_bytes(data)
            process = subprocess.run(
                [
                    str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned",
                    "-File", str(cfg.ROOT / "local_ocr.ps1"), str(photo),
                ],
                capture_output=True, check=True, timeout=20, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            result = json.loads(process.stdout)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(422, "Розпізнавання тривало надто довго. Спробуйте менший фрагмент.") from exc
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise HTTPException(503, "Windows OCR недоступний. Перевірте мовні компоненти та налаштування PowerShell.") from exc
    if result["status"] == "unavailable":
        raise HTTPException(503, "Для Windows OCR потрібен встановлений мовний пакет розпізнавання тексту.")
    if result["status"] == "dimensions":
        raise HTTPException(422, "Фото має завеликі розміри. Обріжте зображення до потрібної області.")
    if result["status"] != "ok" or not (recognized := result["text"].strip()):
        raise HTTPException(422, "На фото не вдалося розпізнати текст. Спробуйте чіткіший знімок або вставте текст вручну.")
    return {"text": recognized, "language": result["language"]}


async def terroai_ocr(request: Request):
    suffix = OCR_FORMATS.get(request.headers.get("content-type", "").lower())
    if not suffix:
        raise HTTPException(415, "Підтримуються фото у форматах PNG, JPEG і BMP.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > cfg.PHOTO_MAX_BYTES:
            raise HTTPException(413, "Фото завелике. Максимум 8 МіБ.")
        data.extend(chunk)
    return await run_in_threadpool(recognize_photo, bytes(data), suffix)


def terroai_save(body: DatasetRequest):
    try:
        added = append_tasks([task_record(body.task, body.source, body.solution)])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"added": added}


def mount_terroai(application: FastAPI):
    application.add_api_route("/ui/api/generate", terroai_generate, methods=["POST"])
    application.add_api_route("/ui/api/settings", terroai_settings, methods=["GET"])
    application.add_api_route("/ui/api/ocr", terroai_ocr, methods=["POST"])
    application.add_api_route("/ui/api/dataset", terroai_save, methods=["POST"])
    application.mount("/ui", StaticFiles(directory=cfg.ROOT / "terroai", html=True), name="terroai")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-ui", action="store_true", help="Serve TerroAI at /ui")
    parser.add_argument("--port", type=int, default=cfg.API_PORT, help="Local HTTP port (default: %(default)s)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.with_ui:
        mount_terroai(app)
    uvicorn.run(app, host=cfg.HOST, port=args.port)


if __name__ == "__main__":
    main()
