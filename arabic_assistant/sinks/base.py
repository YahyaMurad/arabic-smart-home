from typing import Protocol

from arabic_assistant.nlu.base import Command


class Sink(Protocol):
    def send(self, action: Command) -> None: ...
