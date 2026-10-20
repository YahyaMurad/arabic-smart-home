from arabic_assistant.ac import adjust_temperature_all, set_power_all, set_temperature_all
from arabic_assistant.nlu.base import ACAction, Command, DeviceType
from arabic_assistant.sinks.stdout_sink import StdoutSink

# Degrees to move per step size (1 = slight, 2 = regular, 3 = drastic).
STEP_DEGREES = {1: 1, 2: 2, 3: 4}
DEFAULT_STEP_DEGREES = STEP_DEGREES[2]


class AcSink:
    def __init__(self):
        self.fallback = StdoutSink()

    def send(self, action: Command) -> None:
        if action.action.device is not DeviceType.AC:
            self.fallback.send(action)
            return

        if action.action in (ACAction.ON, ACAction.OFF):
            set_power_all(on=action.action is ACAction.ON)
        elif action.action is ACAction.SET and action.value is not None:
            set_temperature_all(action.value)
        elif action.action in (ACAction.UP, ACAction.DOWN):
            degrees = STEP_DEGREES.get(action.step, DEFAULT_STEP_DEGREES)
            sign = 1 if action.action is ACAction.UP else -1
            adjust_temperature_all(sign * degrees)
        else:
            self.fallback.send(action)
