from typing import Protocol

from arabic_assistant.nlu.base import Action


class Sink(Protocol):
    def send(self, action: Action) -> None: ...
