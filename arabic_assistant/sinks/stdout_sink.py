import json
from dataclasses import asdict

from arabic_assistant.nlu.base import Action


class StdoutSink:
    def send(self, action: Action) -> None:
        print(json.dumps(asdict(action)))
