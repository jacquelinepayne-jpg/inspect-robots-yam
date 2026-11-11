"""List Orbbec cameras with the serials, sensors, and stream profiles yam_arms needs.

Reads only the cameras through pyorbbecsdk. No CAN traffic, no torque, no robot
motion, so this is safe to run with the arms powered down. Install the binding
first (``--no-deps`` keeps its example-app dependencies off the rig):

    uv pip install --no-deps 'pyorbbecsdk2>=2.1.3'
    python scripts/orbbec_probe.py

For every device it prints the serial to put in ``{slot}_orbbec_serial``, every
sensor the device exposes, and the colour and depth profiles it offers. A device
without a ``COLOR_SENSOR`` (one that exposes only a left/right colour pair) is
flagged: the builtin reader cannot align depth to it yet.
"""

from __future__ import annotations

from typing import Any


def _profiles(pipeline: Any, sensor: Any) -> list[str]:
    """Return ``WxH@fps FORMAT`` for every video profile one sensor offers."""
    profiles = pipeline.get_stream_profile_list(sensor)
    entries = set()
    for index in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(index).as_video_stream_profile()
        fmt = str(profile.get_format()).split(".")[-1]
        entries.add((profile.get_width(), profile.get_height(), profile.get_fps(), fmt))
    return [f"{w}x{h}@{fps} {fmt}" for w, h, fps, fmt in sorted(entries)]


def main() -> int:
    """Print one block per connected Orbbec device; return 1 when none is found."""
    import pyorbbecsdk as ob  # type: ignore[import-not-found]

    # Keep the Context referenced: it owns the device manager, and a temporary
    # one is freed at once, failing later calls with a NULL "deviceMgr".
    context = ob.Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        print("no Orbbec devices found (check USB, udev rules, and VM passthrough)")
        return 1
    for index in range(devices.get_count()):
        serial = devices.get_device_serial_number_by_index(index)
        name = devices.get_device_name_by_index(index)
        device = devices.get_device_by_serial_number(serial)
        sensors = device.get_sensor_list()
        types = [sensors.get_type_by_index(i) for i in range(sensors.get_count())]
        print(f"== {name}  serial={serial}")
        print("   sensors: " + ", ".join(str(t).split(".")[-1] for t in types))
        if ob.OBSensorType.COLOR_SENSOR not in types:
            print("   WARNING: no COLOR_SENSOR; yam_arms cannot align depth on this device yet")
        pipeline = ob.Pipeline(device)
        for label, sensor in (
            ("colour", ob.OBSensorType.COLOR_SENSOR),
            ("left colour", ob.OBSensorType.LEFT_COLOR_SENSOR),
            ("depth", ob.OBSensorType.DEPTH_SENSOR),
        ):
            if sensor not in types:
                continue
            offered = _profiles(pipeline, sensor)
            print(f"   {label}: " + (", ".join(offered) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
