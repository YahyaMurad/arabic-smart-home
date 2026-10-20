import tinytuya

# Temporary: hardcoded unit for testing.
# Fill in via `python -m tinytuya scan` (ip, device id) and
# `python -m tinytuya wizard` (local_key).
# GANG is the dps index of the only switch that's actually wired up.
DEVICE_ID = ""
IP = ""
LOCAL_KEY = ""
GANG = "1"

_devices: dict[str, tinytuya.OutletDevice] = {}


def _get_device(device_id: str, ip: str, local_key: str) -> tinytuya.OutletDevice:
    if device_id not in _devices:
        dev = tinytuya.OutletDevice(device_id, ip, local_key, version=3.4)
        dev.set_version(3.4)
        _devices[device_id] = dev
    return _devices[device_id]


def set_power(device_id: str, ip: str, local_key: str, gang: str, on: bool) -> bool | None:
    dev = _get_device(device_id, ip, local_key)
    set_result = dev.set_status(on, switch=gang)
    print("set_status raw response:", set_result)
    status = dev.status()
    print("status raw response:", status)
    return status.get("dps", {}).get(gang)


def set_power_all(on: bool) -> None:
    try:
        result = set_power(DEVICE_ID, IP, LOCAL_KEY, GANG, on)
        print(IP, "result:", result)
    except Exception as e:
        print(IP, "failed:", e)
