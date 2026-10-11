import os
import re

from dotenv import load_dotenv
from openai import OpenAI

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

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-6-luna"

ACTIONS_BY_DEVICE = {
    DeviceType.AC: ACAction,
    DeviceType.LIGHT: LightAction,
    DeviceType.SHADES: ShadesAction,
}

ACTION_DESCRIPTIONS = {
    (DeviceType.AC, ACAction.ON): "The command turns the AC on.",
    (DeviceType.AC, ACAction.OFF): "The command turns the AC off.",
    (DeviceType.AC, ACAction.SET): (
        "The command sets the AC to a specific, absolute temperature value, "
        "e.g. 'set the AC to 22 degrees'."
    ),
    (DeviceType.AC, ACAction.UP): (
        "The command asks to make the room warmer / raise the temperature, "
        "without giving an absolute value, e.g. 'it's cold, raise the temperature'."
    ),
    (DeviceType.AC, ACAction.DOWN): (
        "The command asks to make the room cooler / lower the temperature, "
        "without giving an absolute value, e.g. 'it's hot, lower the temperature'."
    ),
    (DeviceType.LIGHT, LightAction.ON): "The command turns the light on.",
    (DeviceType.LIGHT, LightAction.OFF): "The command turns the light off.",
    (DeviceType.SHADES, ShadesAction.UP): "The command raises/opens the shades.",
    (DeviceType.SHADES, ShadesAction.DOWN): "The command lowers/closes the shades.",
}

STEP_DESCRIPTIONS = {
    "1": "A small, slight change, e.g. 'a little' or 'a bit'.",
    "2": "A moderate, regular change with no extra emphasis.",
    "3": "A large, drastic change, e.g. 'a lot' or 'way too hot/cold'.",
}

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_NUMBER_RE = re.compile(r"-?[0-9٠-٩]+")


def extract_temperature(transcript: str) -> int | None:
    match = _NUMBER_RE.search(transcript.translate(_ARABIC_INDIC_DIGITS))
    if match is None:
        return None
    return int(match.group())


class DecisionLLMMatcher:
    def __init__(self):
        pass

    def match(self, transcript: str) -> Command:
        device = self._match_device(transcript)
        action = self._match_action(transcript, device)

        if device is DeviceType.AC:
            if action is ACAction.SET:
                return ACCommand(action=action, value=extract_temperature(transcript))
            if action in (ACAction.UP, ACAction.DOWN):
                return ACCommand(action=action, step=self._match_step(transcript))
            return ACCommand(action=action)

        if device is DeviceType.LIGHT:
            return LightCommand(action=action)

        return ShadesCommand(action=action)

    def _match_device(self, transcript: str) -> DeviceType:
        decision = client.decisions.create(
            model=MODEL,
            input=transcript,
            questions=[
                {
                    "name": "device",
                    "instructions": "Determine which device the text refers to. Select the single best match.",
                    "type": "choice",
                    "choices": [
                        {
                            "value": device.value,
                            "description": f"The command refers to the {device.value}.",
                        }
                        for device in DeviceType
                    ],
                }
            ],
        )

        return DeviceType(decision.answers[0].choice)

    def _match_action(self, transcript: str, device: DeviceType):
        action_enum = ACTIONS_BY_DEVICE[device]

        decision = client.decisions.create(
            model=MODEL,
            input=transcript,
            questions=[
                {
                    "name": "action",
                    "instructions": "Determine what action the text is asking to perform. Select the single best match.",
                    "type": "choice",
                    "choices": [
                        {
                            "value": action.value,
                            "description": ACTION_DESCRIPTIONS[(device, action)],
                        }
                        for action in action_enum
                    ],
                }
            ],
        )

        return action_enum(decision.answers[0].choice)

    def _match_step(self, transcript: str) -> int:
        decision = client.decisions.create(
            model=MODEL,
            input=transcript,
            questions=[
                {
                    "name": "step",
                    "instructions": (
                        "The text asks to raise or lower the AC temperature by some "
                        "relative amount, without giving an absolute value. Determine "
                        "how big that change should be, regardless of whether it is an "
                        "increase or a decrease."
                    ),
                    "type": "choice",
                    "choices": [
                        {"value": value, "description": description}
                        for value, description in STEP_DESCRIPTIONS.items()
                    ],
                }
            ],
        )

        return int(decision.answers[0].choice)


if __name__ == "__main__":
    matcher = DecisionLLMMatcher()
    print("Type a transcript and press enter (Ctrl+D to quit).")
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        if text.strip():
            print(matcher.match(text))
