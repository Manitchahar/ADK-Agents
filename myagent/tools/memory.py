# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Claw-style markdown memory tools for Rocky."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.adk.tools import FunctionTool

_DEFAULT_MEMORY_DIR = "~/.rocky/wiki"
_MEMORY_DIR_ENV = "ROCKY_MEMORY_DIR"
_MAX_FACT_CHARS = 2000
_TOPIC_PATTERN = re.compile(r"[^a-z0-9]+")
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization:",
    "bearer ",
    "client_secret",
    "password",
    "private key",
    "secret",
    "sk-",
    "token",
    "-----begin",
)


def _memory_root() -> Path:
    return Path(os.environ.get(_MEMORY_DIR_ENV, _DEFAULT_MEMORY_DIR)).expanduser()


def _slugify(topic: str) -> str:
    slug = _TOPIC_PATTERN.sub("-", topic.lower()).strip("-")
    return (slug or "general")[:80]


def _topic_title(topic: str) -> str:
    title = " ".join(topic.strip().split())
    return title or "general"


def _topic_path(topic: str) -> Path:
    return _memory_root() / f"{_slugify(topic)}.md"


def _looks_sensitive(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_topic_file(topic: str) -> Path:
    root = _memory_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _topic_path(topic)
    if not path.exists():
        title = _topic_title(topic)
        path.write_text(
            f"# {title}\n\n"
            "Rocky memory page. Store only durable, non-secret facts here.\n\n",
            encoding="utf-8",
        )
    return path


def _iter_memory_entries() -> list[dict[str, Any]]:
    root = _memory_root()
    if not root.exists():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.md")):
        topic = path.stem
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.startswith("- "):
                continue
            text = line[2:].strip()
            entries.append(
                {
                    "topic": topic,
                    "line_number": line_number,
                    "text": text,
                    "path": str(path),
                }
            )
    return entries


async def remember_memory(topic: str, fact: str) -> dict[str, Any]:
    """Persist a durable, non-secret memory fact under a wiki topic."""
    topic = _topic_title(topic)
    fact = " ".join(fact.strip().split())
    if not fact:
        return {"status": "error", "error": "fact must not be empty"}
    if len(fact) > _MAX_FACT_CHARS:
        return {
            "status": "error",
            "error": f"fact must be {_MAX_FACT_CHARS} characters or fewer",
        }
    if _looks_sensitive(fact):
        return {
            "status": "error",
            "error": "Refusing to store secret-looking memory.",
        }

    path = _ensure_topic_file(topic)
    entry = f"- {_timestamp()} - {fact}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return {
        "status": "ok",
        "topic": _slugify(topic),
        "memory": fact,
        "path": str(path),
    }


async def recall_memory(query: str = "", limit: int = 8) -> dict[str, Any]:
    """Search Rocky's markdown memory pages."""
    limit = max(1, min(int(limit), 25))
    query = " ".join(query.strip().lower().split())
    entries = _iter_memory_entries()

    if query:
        terms = query.split()

        def score(entry: dict[str, Any]) -> int:
            haystack = f"{entry['topic']} {entry['text']}".lower()
            return sum(term in haystack for term in terms)

        entries = [
            {**entry, "score": score(entry)} for entry in entries if score(entry) > 0
        ]
        entries.sort(key=lambda entry: (entry["score"], entry["text"]), reverse=True)
    else:
        entries.sort(key=lambda entry: entry["text"], reverse=True)

    return {
        "status": "ok",
        "query": query,
        "count": min(len(entries), limit),
        "memories": entries[:limit],
    }


async def list_memory_topics() -> dict[str, Any]:
    """List available Rocky memory wiki topics."""
    root = _memory_root()
    if not root.exists():
        return {"status": "ok", "count": 0, "topics": []}

    topics = []
    for path in sorted(root.glob("*.md")):
        entry_count = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        )
        topics.append(
            {
                "topic": path.stem,
                "entries": entry_count,
                "path": str(path),
            }
        )
    return {"status": "ok", "count": len(topics), "topics": topics}


async def forget_memory(topic: str, phrase: str) -> dict[str, Any]:
    """Remove memory entries in a topic that contain a phrase."""
    phrase = phrase.strip().lower()
    if not phrase:
        return {"status": "error", "error": "phrase must not be empty"}

    path = _topic_path(topic)
    if not path.exists():
        return {"status": "error", "error": f"unknown memory topic: {_slugify(topic)}"}

    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if line.startswith("- ") and phrase in line.lower():
            removed += 1
            continue
        kept.append(line)

    path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "topic": path.stem,
        "removed": removed,
        "path": str(path),
    }


remember_memory_tool = FunctionTool(func=remember_memory)
recall_memory_tool = FunctionTool(func=recall_memory)
list_memory_topics_tool = FunctionTool(func=list_memory_topics)
forget_memory_tool = FunctionTool(func=forget_memory)
