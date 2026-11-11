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

``--snap DIR`` also saves one colour photo per camera as ``DIR/<serial>.jpg``,
to tell which serial is the top, left, and right camera:

    python scripts/orbbec_probe.py --snap ~/orbbec-snaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
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


def _snap(ob: Any, device: Any, path: Path) -> None:
    """Save one 640x480 RGB colour frame from ``device`` as a JPEG at ``path``."""
    import cv2
    import numpy as np

    pipeline = ob.Pipeline(device)
    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    config = ob.Config()
    config.enable_stream(profiles.get_video_stream_profile(640, 480, ob.OBFormat.RGB, 30))
    pipeline.start(config)
    try:
        for _ in range(30):  # let auto-exposure settle and skip empty sets
            frames = pipeline.wait_for_frames(1000)
            colour = frames.get_color_frame() if frames is not None else None
        if colour is None:
            raise RuntimeError("no colour frame within 30 waits")
        rgb = np.asarray(colour.get_data(), dtype=np.uint8).reshape(
            colour.get_height(), colour.get_width(), 3
        )
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        pipeline.stop()


def main(argv: list[str] | None = None) -> int:
    """Print every Orbbec device; return 1 when none is found or any fails to open."""
    import pyorbbecsdk as ob  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snap", type=Path, help="save one colour JPEG per camera here")
    args = parser.parse_args(argv)
    if args.snap is not None:
        args.snap.expanduser().mkdir(parents=True, exist_ok=True)

    # Keep the Context referenced: it owns the device manager, and a temporary
    # one is freed at once, failing later calls with a NULL "deviceMgr".
    context = ob.Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        print("no Orbbec devices found (check USB, udev rules, and VM passthrough)")
        return 1
    serials = [
        (devices.get_device_serial_number_by_index(i), devices.get_device_name_by_index(i))
        for i in range(devices.get_count())
    ]
    # Serials come from enumeration, which needs no device access; print them
    # all before opening anything, so a camera that fails to open still shows.
    for serial, name in serials:
        print(f"found {name}  serial={serial}")
    if any(not serial for serial, _ in serials):
        print(
            "WARNING: blank serials mean this user cannot access the cameras over USB. "
            "Install Orbbec's udev rule (README, Orbbec depth) and replug, or compare "
            'with: sudo "$(which python)" scripts/orbbec_probe.py'
        )
    failures = 0
    for serial, name in serials:
        print(f"== {name}  serial={serial}")
        try:
            device = devices.get_device_by_serial_number(serial)
            _describe(ob, device)
            if args.snap is not None:
                path = args.snap.expanduser() / f"{serial}.jpg"
                _snap(ob, device, path)
                print(f"   saved {path}")
        except Exception as exc:
            failures += 1
            print(f"   ERROR opening device: {exc}")
            print(
                "   Is another process streaming it (a loopback bridge, OrbbecViewer, "
                "a running eval)? Otherwise check Orbbec's udev rule for USB vendor 2bc5."
            )
    return 1 if failures else 0


def _describe(ob: Any, device: Any) -> None:
    """Print one opened device's sensors and its colour and depth profiles."""
    sensors = device.get_sensor_list()
    types = [sensors.get_type_by_index(i) for i in range(sensors.get_count())]
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


if __name__ == "__main__":
    raise SystemExit(main())
