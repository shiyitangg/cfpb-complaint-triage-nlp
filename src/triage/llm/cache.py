"""Disk cache for API responses: one JSON file per key."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_key(model: str, prompt_version: str, prompt_sha: str, run: int, text: str) -> str:
    return f"{model}/{prompt_version}-{prompt_sha[:12]}/run{run}/{text_hash(text)}"


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, record: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        tmp.replace(p)
