from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Union


@dataclass
class DatasetItem:
    id: str
    prompt: str
    gold: Union[str, int, list]
    meta: dict


class Loader(Protocol):
    name: str

    def load(self, n: int) -> list[DatasetItem]:
        ...
