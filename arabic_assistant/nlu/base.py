from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class DeviceType(Enum):
    AC = "ac"
    LIGHT = "light"
    SHADES = "shades"


class ACAction(Enum):
    ON = "on"
    OFF = "off"
    SET = "set"
    UP = "up"
    DOWN = "down"

    @property
    def device(self) -> DeviceType:
        return DeviceType.AC


class LightAction(Enum):
    ON = "on"
    OFF = "off"

    @property
    def device(self) -> DeviceType:
        return DeviceType.LIGHT


class ShadesAction(Enum):
    UP = "up"
    DOWN = "down"

    @property
    def device(self) -> DeviceType:
        return DeviceType.SHADES


@dataclass(frozen=True)
class ACCommand:
    action: ACAction
    value: int | None = None
    step: int | None = None


@dataclass(frozen=True)
class LightCommand:
    action: LightAction


@dataclass(frozen=True)
class ShadesCommand:
    action: ShadesAction


Command = ACCommand | LightCommand | ShadesCommand


class Matcher(Protocol):
    def match(self, transcript: str) -> Command | None: ...
