import asyncio
import threading

from greeclimate.device import Device, DeviceInfo

# Temporary: hardcoded units for testing.
UNITS = {
    "192.168.1.31": "580d0d9292f8",
}

# One event loop for the whole process, so bound devices stay usable.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()

_devices: dict[str, Device] = {}


async def _get_device(ip: str, mac: str) -> Device:
    if ip not in _devices:
        dev = Device(DeviceInfo(ip, 7000, mac, "ac"))
        await dev.bind()
        _devices[ip] = dev
    return _devices[ip]


async def set_power(ip: str, mac: str, on: bool) -> bool | None:
    dev = await _get_device(ip, mac)
    dev.power = on
    await dev.push_state_update()
    await asyncio.sleep(3)
    await dev.update_state()
    return dev.power


async def set_temperature(ip: str, mac: str, value: int) -> int | None:
    dev = await _get_device(ip, mac)
    dev.target_temperature = value
    await dev.push_state_update()
    await asyncio.sleep(3)
    await dev.update_state()
    return dev.target_temperature


async def adjust_temperature(ip: str, mac: str, delta: int) -> int | None:
    dev = await _get_device(ip, mac)
    await dev.update_state()
    dev.target_temperature = dev.target_temperature + delta
    await dev.push_state_update()
    await asyncio.sleep(3)
    await dev.update_state()
    return dev.target_temperature


async def _run_all(fn, *args) -> None:
    for ip, mac in UNITS.items():
        try:
            result = await fn(ip, mac, *args)
            print(ip, "result:", result)
        except Exception as e:
            print(ip, "failed:", e)


def set_power_all(on: bool) -> None:
    asyncio.run_coroutine_threadsafe(_run_all(set_power, on), _loop).result()


def set_temperature_all(value: int) -> None:
    asyncio.run_coroutine_threadsafe(_run_all(set_temperature, value), _loop).result()


def adjust_temperature_all(delta: int) -> None:
    asyncio.run_coroutine_threadsafe(_run_all(adjust_temperature, delta), _loop).result()
