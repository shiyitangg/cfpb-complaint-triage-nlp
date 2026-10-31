"""Versioned prompt files. A file has a '## system' section and a '## user' section; the user
section must contain {narrative} exactly once."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Prompt:
    version: str
    sha256: str
    system: str
    user_template: str

    def render(self, narrative: str) -> str:
        return self.user_template.replace("{narrative}", narrative)


def parse_prompt(raw: str, version: str) -> Prompt:
    parts = _SECTION.split(raw)
    sections = dict(zip(parts[1::2], (p.strip() for p in parts[2::2])))
    if set(sections) != {"system", "user"}:
        raise ValueError(f"prompt {version}: need exactly one '## system' and one '## user' section")
    if sections["user"].count("{narrative}") != 1:
        raise ValueError(f"prompt {version}: user section must contain {{narrative}} exactly once")
    return Prompt(version, hashlib.sha256(raw.encode()).hexdigest(), sections["system"], sections["user"])


def load_prompt(path: str | Path) -> Prompt:
    p = Path(path)
    return parse_prompt(p.read_text(encoding="utf-8"), p.stem)
