import re
from collections import defaultdict

import yaml

from src.config import PathConfig
from src.nlu.base import Action
from src.transformation.normalization import normalize_arabic


class KeywordMatcher:
    def __init__(self, paths: PathConfig | None = None):
        paths = paths or PathConfig()
        cfg = yaml.safe_load(paths.prompts_config.read_text(encoding="utf-8"))
        prompts = cfg["prompts"]

        device_words: dict[str, set[str]] = defaultdict(set)
        action_words: dict[tuple[str, str], set[str]] = defaultdict(set)
        extra_key: dict[tuple[str, str], str] = {}

        for p in prompts:
            act = p["action"]
            device, action = act["device"], act["action"]
            words = set(normalize_arabic(p["text"]).split())

            device_words[device].update(words)
            action_words[(device, action)].update(words)

            for key in ("value", "steps"):
                if key in act:
                    extra_key[(device, action)] = key

        # Words unique to one device (e.g. "الضو") reliably identify it;
        # shared verbs (e.g. "افتح") do not, since they mean different
        # things per device.
        self.device_unique_words = {
            device: words
            - set().union(*(w for d, w in device_words.items() if d != device))
            for device, words in device_words.items()
        }

        # Strip each device's own naming words out of its action vocab so
        # that only the action-distinguishing verbs remain (otherwise every
        # action for a device would "match" on the device noun alone).
        self.action_words = {
            (device, action): words - self.device_unique_words[device]
            for (device, action), words in action_words.items()
        }
        self.extra_key = extra_key
        self.devices = list(device_words)

    def match(self, transcript: str) -> Action | None:
        words = set(normalize_arabic(transcript).split())

        device = next(
            (d for d in self.devices if self.device_unique_words[d] & words), None
        )
        if device is None:
            return None

        candidates = [
            (action, len(action_word_set & words))
            for (d, action), action_word_set in self.action_words.items()
            if d == device
        ]
        action, best_overlap = max(candidates, key=lambda c: c[1], default=(None, 0))
        if best_overlap == 0:
            return None

        numbers = re.findall(r"\d+", normalize_arabic(transcript))
        key = self.extra_key.get((device, action))
        kwargs = {key: int(numbers[0])} if key and numbers else {}

        return Action(device=device, action=action, **kwargs)
