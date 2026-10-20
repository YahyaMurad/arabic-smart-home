import json
from dataclasses import asdict
from enum import Enum

from arabic_assistant.nlu.base import Command


def _default(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class StdoutSink:
    def send(self, action: Command) -> None:
        print(json.dumps(asdict(action), default=_default))
