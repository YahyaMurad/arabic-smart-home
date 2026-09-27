from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Action:
    device: str
    action: str
    value: int | None = None
    steps: int | None = None


class Matcher(Protocol):
    def match(self, transcript: str) -> Action | None: ...
