from arabic_assistant.nlu.base import Command, DeviceType
from arabic_assistant.sinks.ac_sink import AcSink
from arabic_assistant.sinks.light_sink import LightSink
from arabic_assistant.sinks.stdout_sink import StdoutSink


class CompositeSink:
    def __init__(self):
        self.sinks = {
            DeviceType.AC: AcSink(),
            DeviceType.LIGHT: LightSink(),
        }
        self.fallback = StdoutSink()

    def send(self, action: Command) -> None:
        sink = self.sinks.get(action.action.device, self.fallback)
        sink.send(action)
