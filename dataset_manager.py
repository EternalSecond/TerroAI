"""Local JSONL collection. Python source is read as text and never imported."""

import argparse
import ast
import json
import os
import tempfile
import tokenize
from pathlib import Path

from filelock import FileLock

import config as cfg

STARTER_DATA = cfg.DATA_DIR / "examples" / "starter_tasks.jsonl"
IGNORED_DIRS = {
    "venv", "env", "__pycache__", "site-packages", "node_modules", "build", "dist",
}


def task_record(instruction: str, code: str, solution: str) -> dict:
    if not solution.strip():
        raise ValueError("Введіть повне рішення.")
    return {"prompt": cfg.format_task(instruction, code), "response": solution}


def parse_records(text: str, source: Path) -> list[dict]:
    records = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("prompt", "response")
            ):
                raise ValueError("потрібні непорожні рядки prompt і response")
        except ValueError as exc:
            raise ValueError(f"{source}:{number}: {exc}") from exc
        records.append(row)
    return records


def read_records(path: Path) -> list[dict]:
    # Writers publish a complete file with os.replace. An open reader may otherwise
    # prevent that replacement on Windows.
    with FileLock(f"{path}.lock", timeout=15):
        return parse_records(path.read_text(encoding="utf-8-sig"), path)


def write_records(path: Path, records: list[dict]) -> None:
    """Publish a new JSONL file while the caller holds its lock."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_tasks(records: list[dict]) -> int:
    path = cfg.CUSTOM_DATA
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{path}.lock", timeout=15):
        existing = parse_records(path.read_text(encoding="utf-8-sig"), path) if path.exists() else []
        seen = {(row["prompt"], row["response"]) for row in existing}
        added = 0
        for record in records:
            key = record["prompt"], record["response"]
            if key not in seen:
                existing.append(record)
                seen.add(key)
                added += 1
        if added:
            write_records(path, existing)
    return added


def code_chunks(source: str):
    """Prefer module-level boundaries, then split large blocks by lines."""
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
        boundaries = {
            min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))]) - 1
            for node in tree.body
        }
    except SyntaxError:
        boundaries = set()
    limit = cfg.CODE_CHUNK_CHARS
    chunk = ""
    for number, line in enumerate(lines):
        if (number in boundaries and len(chunk) >= limit // 2) or len(chunk) + len(line) > limit:
            if chunk.strip():
                yield chunk
            chunk = ""
        while len(line) > limit:
            part, line = line[:limit], line[limit:]
            if part.strip():
                yield part
        chunk += line
    if chunk.strip():
        yield chunk


def continuation(name: str, chunk: str) -> dict | None:
    middle = len(chunk) // 2
    breaks = [index + 1 for index, character in enumerate(chunk) if character == "\n"]
    splits = [point for point in breaks if len(chunk) // 4 <= point <= 3 * len(chunk) // 4]
    split = min(splits, key=lambda point: abs(point - middle), default=middle)
    prefix, suffix = chunk[:split], chunk[split:]
    if not prefix.strip() or not suffix.strip():
        return None
    prompt = cfg.format_task(
        f"Продовж цей фрагмент з {name}. Виведи тільки продовження Python-коду.",
        prefix,
    )
    return {"prompt": prompt, "response": suffix}


def import_codebase(root: Path) -> tuple[int, int]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Вкажіть папку: {root}")
    records = []
    skipped = 0
    is_junction = getattr(os.path, "isjunction", lambda path: False)
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if not name.startswith(".") and name.lower() not in IGNORED_DIRS
            and not (Path(directory) / name).is_symlink()
            and not is_junction(Path(directory) / name)
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() != ".py" or name.startswith(".") or path.is_symlink():
                continue
            try:
                if path.stat().st_size > cfg.CODE_FILE_MAX_BYTES:
                    skipped += 1
                    continue
                with tokenize.open(path) as handle:
                    source = handle.read()
            except (OSError, UnicodeError, SyntaxError, LookupError) as exc:
                print(f"Пропущено {path}: {exc}")
                skipped += 1
                continue
            relative = path.relative_to(root).as_posix()
            for chunk in code_chunks(source):
                if record := continuation(relative, chunk):
                    records.append(record)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{cfg.CODEBASE_DATA}.lock", timeout=15):
        write_records(cfg.CODEBASE_DATA, records)
    return len(records), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Create my_codebase.jsonl from a source folder")
    scan.add_argument("folder", type=Path)
    add = commands.add_parser("add", help="Add an instruction, source file and reviewed solution")
    add.add_argument("--task", required=True)
    add.add_argument("--code", type=Path, required=True)
    add.add_argument("--solution", type=Path, required=True)
    commands.add_parser("seed", help="Add five starter refactoring examples")
    args = parser.parse_args()
    try:
        if args.command == "scan":
            count, skipped = import_codebase(args.folder)
            print(f"{cfg.CODEBASE_DATA}: {count} прикладів, {skipped} пропущених файлів.")
        elif args.command == "seed":
            records = parse_records(STARTER_DATA.read_text(encoding="utf-8"), STARTER_DATA)
            print(f"{cfg.CUSTOM_DATA}: додано {append_tasks(records)} прикладів.")
        else:
            record = task_record(
                args.task,
                args.code.read_text(encoding="utf-8-sig"),
                args.solution.read_text(encoding="utf-8-sig"),
            )
            print(f"{cfg.CUSTOM_DATA}: додано {append_tasks([record])} прикладів.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
