import re
from collections import defaultdict
from pathlib import Path

import yaml

from arabic_assistant.nlu.base import (
    ACAction,
    ACCommand,
    Command,
    DeviceType,
    LightAction,
    LightCommand,
    ShadesAction,
    ShadesCommand,
)
from arabic_assistant.normalization import normalize_arabic

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "config.yml"

# Which action enum, and which Command class, applies to each device.
ACTIONS_BY_DEVICE = {
    DeviceType.AC: ACAction,
    DeviceType.LIGHT: LightAction,
    DeviceType.SHADES: ShadesAction,
}
COMMAND_BY_DEVICE = {
    DeviceType.AC: ACCommand,
    DeviceType.LIGHT: LightCommand,
    DeviceType.SHADES: ShadesCommand,
}

# ACCommand field that a trailing number in the transcript should fill,
# per action. Actions not listed here take no extra value.
EXTRA_KEY_BY_ACTION = {
    (DeviceType.AC, ACAction.SET): "value",
    (DeviceType.AC, ACAction.UP): "step",
    (DeviceType.AC, ACAction.DOWN): "step",
}


class KeywordMatcher:
    def __init__(self, config_path: Path | None = None):
        config_path = config_path or DEFAULT_CONFIG_PATH
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        prompts = cfg["prompts"]

        device_words: dict[DeviceType, set[str]] = defaultdict(set)
        action_words: dict[tuple[DeviceType, object], set[str]] = defaultdict(set)

        for p in prompts:
            act = p["action"]
            device = DeviceType(act["device"])
            action_enum = ACTIONS_BY_DEVICE[device]
            try:
                action = action_enum(act["action"])
            except ValueError:
                # Stale entry for an action this device no longer supports.
                continue

            words = set(normalize_arabic(p["text"]).split())
            device_words[device].update(words)
            action_words[(device, action)].update(words)

        # Words unique to one device (e.g. device noun) reliably identify it;
        # shared verbs do not, since they mean different things per device.
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
        self.devices = list(device_words)

    def match(self, transcript: str) -> Command | None:
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
        if action is None or best_overlap == 0:
            return None

        numbers = re.findall(r"\d+", normalize_arabic(transcript))
        key = EXTRA_KEY_BY_ACTION.get((device, action))
        kwargs = {key: int(numbers[0])} if key and numbers else {}

        return COMMAND_BY_DEVICE[device](action=action, **kwargs)
