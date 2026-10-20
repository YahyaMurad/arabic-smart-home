from arabic_assistant.light import set_power_all
from arabic_assistant.nlu.base import Command, DeviceType, LightAction
from arabic_assistant.sinks.stdout_sink import StdoutSink


class LightSink:
    def __init__(self):
        self.fallback = StdoutSink()

    def send(self, action: Command) -> None:
        if action.action.device is not DeviceType.LIGHT:
            self.fallback.send(action)
            return

        if action.action in (LightAction.ON, LightAction.OFF):
            set_power_all(on=action.action is LightAction.ON)
        else:
            self.fallback.send(action)
