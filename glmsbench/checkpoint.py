from __future__ import annotations
import json
import os
from dataclasses import asdict
from typing import Set
from .models import RequestRecord


class Checkpoint:
    """Append-only JSONL of completed RequestRecords; supports resume.

    Each line: a serialized RequestRecord. The (provider, suite, item_id,
    pass_type, pass_index) tuple is the unique key for resume decisions.
    """

    def __init__(self, path: str):
        self.path = path
        self._done: Set[tuple] = set()
        if os.path.exists(path):
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (d["provider"], d["suite"], d["item_id"], d["pass_type"], d["pass_index"])
                self._done.add(key)

    def _key(self, provider, suite, item_id, pass_type, pass_index) -> tuple:
        return (provider, suite, item_id, pass_type, pass_index)

    def is_done(self, provider, suite, item_id, pass_type, pass_index) -> bool:
        return self._key(provider, suite, item_id, pass_type, pass_index) in self._done

    def completed_keys(self) -> set:
        return set(self._done)

    def append(self, record: RequestRecord):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")
        self._done.add(self._key(record.provider, record.suite, record.item_id,
                                 record.pass_type, record.pass_index))
