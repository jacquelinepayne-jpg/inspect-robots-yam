"""Orbbec (pyorbbecsdk v2) colour/depth reader, its config keys, and embodiment wiring.

The pyorbbecsdk surface is a small recording fake whose method names match the
real binding (checked against pyorbbecsdk2 2.1.3). Colour decoding uses the real
OpenCV module, so the YUYV/UYVY/MJPG conversion codes are exercised for real.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_yam.embodiment as embodiment_module
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import (
    ORBBEC_INSTALL_COMMAND,
    YAMEmbodiment,
    _CompositeCameraReader,
    _OrbbecBundle,
    _OrbbecCameraReader,
)
from inspect_robots_yam.health import HealthReport as HealthReportType
from inspect_robots_yam.health import _unchecked_depth_cameras, main
from inspect_robots_yam.operator import OperatorIO

SERIALS = {"top_cam": "OB1", "left_cam": "OB2", "right_cam": "OB3"}
SIZE = (8, 6)  # width, height of every fake stream unless a test says otherwise

_OPENED: list[_OrbbecCameraReader] = []


@pytest.fixture(autouse=True)
def close_readers() -> Iterator[None]:
    """Close every reader a test built, however the test ended."""
    yield
    while _OPENED:
        _OPENED.pop().close()


# --------------------------------------------------------------------------- fakes


class Enum(SimpleNamespace):
    """A namespace of distinct string members, standing in for a pybind11 enum."""


class FakeOb:
    """The pyorbbecsdk module surface the reader uses."""

    OBSensorType = Enum(
        COLOR_SENSOR="COLOR_SENSOR",
        LEFT_COLOR_SENSOR="LEFT_COLOR_SENSOR",
        RIGHT_COLOR_SENSOR="RIGHT_COLOR_SENSOR",
        DEPTH_SENSOR="DEPTH_SENSOR",
        IR_SENSOR="IR_SENSOR",
    )
    OBStreamType = Enum(COLOR_STREAM="COLOR_STREAM")
    OBFormat = Enum(
        UNKNOWN_FORMAT="UNKNOWN_FORMAT",
        RGB="RGB",
        BGR="BGR",
        YUYV="YUYV",
        YUY2="YUY2",
        UYVY="UYVY",
        MJPG="MJPG",
        Y16="Y16",
        NV12="NV12",
    )
    OBFrameAggregateOutputMode = Enum(FULL_FRAME_REQUIRE="FULL_FRAME_REQUIRE")

    def __init__(self, devices: list[FakeDevice] | None = None) -> None:
        self.devices = devices if devices is not None else [FakeDevice(s) for s in SERIALS.values()]
        self.pipelines: list[FakePipeline] = []
        self.configs: list[FakeConfig] = []
        self.aligns: list[FakeAlign] = []

    def Context(self) -> FakeContext:
        return FakeContext(self.devices)

    def Pipeline(self, device: FakeDevice) -> FakePipeline:
        pipeline = FakePipeline(device)
        self.pipelines.append(pipeline)
        return pipeline

    def Config(self) -> FakeConfig:
        config = FakeConfig()
        self.configs.append(config)
        return config

    def AlignFilter(self, align_to_stream: Any) -> FakeAlign:
        align = FakeAlign(align_to_stream)
        self.aligns.append(align)
        return align


class FakeContext:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self._devices = devices

    def query_devices(self) -> FakeDeviceList:
        return FakeDeviceList(self._devices)


class FakeDeviceList:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self._devices = devices

    def get_count(self) -> int:
        return len(self._devices)

    def get_device_serial_number_by_index(self, index: int) -> str:
        return self._devices[index].serial

    def get_device_name_by_index(self, index: int) -> str:
        return self._devices[index].name

    def get_device_by_serial_number(self, serial: str) -> FakeDevice:
        return next(device for device in self._devices if device.serial == serial)


class FakeSensorList:
    def __init__(self, types: list[str]) -> None:
        self._types = types

    def get_count(self) -> int:
        return len(self._types)

    def get_type_by_index(self, index: int) -> str:
        return self._types[index]


class FakeProfile:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width, self.height, self.fps = width, height, fps

    def as_video_stream_profile(self) -> FakeProfile:
        return self

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_fps(self) -> int:
        return self.fps

    def get_intrinsic(self) -> Any:
        return SimpleNamespace(fx=4.0, fy=5.0, cx=3.5, cy=2.5)


class FakeProfileList:
    def __init__(self, sensor: str, offered: list[tuple[int, int, int]]) -> None:
        self.sensor = sensor
        self._profiles = [FakeProfile(*entry) for entry in offered]
        self.requests: list[tuple[int, int, Any, int]] = []

    def get_video_stream_profile(self, width: int, height: int, fmt: Any, fps: int) -> FakeProfile:
        self.requests.append((width, height, fmt, fps))
        for profile in self._profiles:
            if (profile.width, profile.height, profile.fps) == (width, height, fps):
                return profile
        raise RuntimeError("Invalid input, No matched video stream profile found!")

    def get_count(self) -> int:
        return len(self._profiles)

    def get_stream_profile_by_index(self, index: int) -> FakeProfile:
        return self._profiles[index]


class ColourFrame:
    def __init__(self, fmt: str, data: npt.NDArray[np.uint8], width: int, height: int) -> None:
        self._fmt, self._data, self._w, self._h = fmt, data, width, height

    def get_format(self) -> str:
        return self._fmt

    def get_data(self) -> npt.NDArray[np.uint8]:
        return self._data

    def get_width(self) -> int:
        return self._w

    def get_height(self) -> int:
        return self._h


class DepthFrame:
    def __init__(self, depth: npt.NDArray[np.uint16], scale: float = 1.0) -> None:
        self._depth, self._scale = depth, scale

    def get_data(self) -> npt.NDArray[np.uint8]:
        return np.ascontiguousarray(self._depth).view(np.uint8).reshape(-1)

    def get_width(self) -> int:
        return int(self._depth.shape[1])

    def get_height(self) -> int:
        return int(self._depth.shape[0])

    def get_depth_scale(self) -> float:
        return self._scale


class FrameSet:
    """Both the raw set from the pipeline and the aligned set from the filter."""

    def __init__(
        self,
        colour: ColourFrame | None,
        depth: DepthFrame | None,
        *,
        unalignable: bool = False,
    ) -> None:
        self.colour, self.depth, self.unalignable = colour, depth, unalignable

    def as_frame_set(self) -> FrameSet:
        return self

    def get_color_frame(self) -> ColourFrame | None:
        return self.colour

    def get_depth_frame(self) -> DepthFrame | None:
        return self.depth


def rgb_frame(fill: int = 7, size: tuple[int, int] = SIZE) -> ColourFrame:
    width, height = size
    data = np.full((height, width, 3), fill, dtype=np.uint8)
    data[..., 0] = fill  # R
    data[..., 2] = 200  # B: lets channel-order mistakes show
    return ColourFrame("RGB", data.reshape(-1), width, height)


def frames(
    fill: int = 7,
    depth_mm: int = 1000,
    *,
    scale: float = 1.0,
    size: tuple[int, int] = SIZE,
) -> FrameSet:
    width, height = size
    depth = np.full((height, width), depth_mm, dtype=np.uint16)
    return FrameSet(rgb_frame(fill, size), DepthFrame(depth, scale))


Response = FrameSet | None | BaseException


class FakeDevice:
    """One Orbbec device: its sensors, offered profiles, and scripted frame responses."""

    def __init__(
        self,
        serial: str,
        name: str = "Orbbec Gemini 305",
        *,
        sensors: list[str] | None = None,
        offered: list[tuple[int, int, int]] | None = None,
        responses: list[Response] | None = None,
    ) -> None:
        self.serial = serial
        self.name = name
        self.sensors = sensors if sensors is not None else ["COLOR_SENSOR", "DEPTH_SENSOR"]
        self.offered = offered if offered is not None else [(640, 480, 30), (*SIZE, 30)]
        self.responses: list[Response] = list(responses or [])
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None

    def get_sensor_list(self) -> FakeSensorList:
        return FakeSensorList(self.sensors)


class FakeConfig:
    def __init__(self) -> None:
        self.streams: list[FakeProfile] = []
        self.aggregate: Any = None

    def enable_stream(self, profile: FakeProfile) -> None:
        self.streams.append(profile)

    def set_frame_aggregate_output_mode(self, mode: Any) -> None:
        self.aggregate = mode


class FakePipeline:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.lists: dict[str, FakeProfileList] = {}
        self.started: FakeConfig | None = None
        self.stopped = False
        self.waits = 0

    def get_stream_profile_list(self, sensor: str) -> FakeProfileList:
        profile_list = FakeProfileList(sensor, self.device.offered)
        self.lists[sensor] = profile_list
        return profile_list

    def start(self, config: FakeConfig) -> None:
        if self.device.start_error is not None:
            raise self.device.start_error
        self.started = config

    def wait_for_frames(self, timeout_ms: int) -> FrameSet | None:
        assert timeout_ms == 1000
        self.waits += 1
        if self.device.responses:
            response = self.device.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        time.sleep(0.002)  # an idle camera: keep the drain thread from spinning
        return frames()

    def stop(self) -> None:
        self.stopped = True
        if self.device.stop_error is not None:
            raise self.device.stop_error


class FakeAlign:
    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def process(self, source: FrameSet) -> FrameSet | None:
        return None if source.unalignable else source


class Clock:
    """A monotonic clock advanced only by tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def reader(
    ob: FakeOb | None = None,
    serials: dict[str, str] | None = None,
    *,
    clock: Clock | None = None,
    **kwargs: Any,
) -> tuple[_OrbbecCameraReader, FakeOb]:
    ob = ob if ob is not None else FakeOb()
    kwargs.setdefault("capture_size", SIZE)
    kwargs.setdefault("sleep_fn", lambda _: None)
    built = _OrbbecCameraReader(
        serials if serials is not None else dict(SERIALS),
        30,
        rs_module=ob,
        cv2_module=cv2,
        clock=clock if clock is not None else Clock(),
        **kwargs,
    )
    _OPENED.append(built)
    return built, ob


def cfg(**kwargs: Any) -> YamConfig:
    kwargs.setdefault("cam_width", 4)
    kwargs.setdefault("cam_height", 3)
    return YamConfig(**kwargs)


# --------------------------------------------------------------------------- reader


def test_images_depth_and_intrinsics_are_served_for_every_camera() -> None:
    camera, ob = reader()

    images = camera(cfg())
    extra = camera.extra(cfg())

    assert set(images) == set(SERIALS)
    for name in SERIALS:
        assert images[name].shape == (3, 4, 3)
        assert images[name].dtype == np.uint8
        assert images[name][0, 0].tolist() == [7, 7, 200]  # RGB order kept
        depth = extra[f"{name}_depth"]()
        assert depth.shape == (3, 4)
        assert depth.dtype == np.float32
        np.testing.assert_allclose(depth, 1.0)  # 1000 units x 1 mm/unit
        k_matrix = extra[f"{name}_intrinsics"]
        # Captured 8x6, published 4x3: K halves on both axes.
        np.testing.assert_allclose(
            k_matrix, [[2.0, 0.0, 1.75], [0.0, 2.5, 1.25], [0.0, 0.0, 1.0]], rtol=1e-6
        )
    for config in ob.configs:
        assert config.aggregate == "FULL_FRAME_REQUIRE"
        assert len(config.streams) == 2
    assert all(align.stream == "COLOR_STREAM" for align in ob.aligns)


def test_depth_uses_each_frames_own_scale() -> None:
    device = FakeDevice("OB1", responses=[frames(depth_mm=400, scale=0.5)])
    camera, _ = reader(FakeOb([device]), {"top_cam": "OB1"})

    camera(cfg())
    depth = camera.extra(cfg())["top_cam_depth"]()

    np.testing.assert_allclose(depth, 0.2)  # 400 x 0.5 mm = 0.2 m


def test_profiles_request_configured_sizes_and_fps() -> None:
    devices = [
        FakeDevice("OB1", offered=[(8, 6, 30), (4, 2, 30)]),
        FakeDevice("OB2", offered=[(16, 12, 30), (4, 2, 30)]),
    ]
    camera, ob = reader(
        FakeOb(devices),
        {"top_cam": "OB1", "left_cam": "OB2"},
        depth_capture_size=(4, 2),
        capture_sizes={"left_cam": (16, 12)},
    )

    camera(cfg())

    top, left = ob.pipelines
    assert top.lists["COLOR_SENSOR"].requests == [(8, 6, "UNKNOWN_FORMAT", 30)]
    assert top.lists["DEPTH_SENSOR"].requests == [(4, 2, "UNKNOWN_FORMAT", 30)]
    assert left.lists["COLOR_SENSOR"].requests == [(16, 12, "UNKNOWN_FORMAT", 30)]
    assert left.lists["DEPTH_SENSOR"].requests == [(4, 2, "UNKNOWN_FORMAT", 30)]


def test_depth_defaults_to_the_colour_size_and_per_camera_depth_override_wins() -> None:
    devices = [FakeDevice("OB1"), FakeDevice("OB2", offered=[(*SIZE, 30), (4, 2, 30)])]
    camera, ob = reader(
        FakeOb(devices),
        {"top_cam": "OB1", "left_cam": "OB2"},
        depth_capture_sizes={"left_cam": (4, 2)},
    )

    camera(cfg())

    top, left = ob.pipelines
    assert top.lists["DEPTH_SENSOR"].requests == [(*SIZE, "UNKNOWN_FORMAT", 30)]
    assert left.lists["DEPTH_SENSOR"].requests == [(4, 2, "UNKNOWN_FORMAT", 30)]


@pytest.mark.parametrize(
    ("sensors", "listing"),
    [
        (
            ["LEFT_COLOR_SENSOR", "RIGHT_COLOR_SENSOR", "DEPTH_SENSOR"],
            "LEFT_COLOR_SENSOR, RIGHT_COLOR_SENSOR, DEPTH_SENSOR",
        ),
        ([], "none"),
    ],
)
def test_device_without_colour_sensor_is_rejected(sensors: list[str], listing: str) -> None:
    camera, ob = reader(FakeOb([FakeDevice("OB1", sensors=sensors)]), {"top_cam": "OB1"})

    with pytest.raises(
        RuntimeError, match=f"no colour sensor to align depth to; sensors: {listing}"
    ) as info:
        camera(cfg())
    assert "left/right colour pair are not supported yet" in str(info.value)
    assert ob.pipelines == []  # rejected before any stream starts


def test_missing_serial_lists_visible_devices() -> None:
    camera, _ = reader(FakeOb([FakeDevice("CP1", "Orbbec Gemini 305")]), {"top_cam": "OB1"})

    with pytest.raises(
        RuntimeError,
        match=(
            r"cannot find Orbbec camera top_cam \(OB1\); "
            r"visible devices: Orbbec Gemini 305 / CP1"
        ),
    ):
        camera(cfg())


def test_missing_serial_with_no_devices_says_none() -> None:
    camera, _ = reader(FakeOb([]), {"top_cam": "OB1"})

    with pytest.raises(RuntimeError, match="visible devices: none"):
        camera(cfg())


def test_unoffered_size_lists_the_offered_profiles() -> None:
    device = FakeDevice("OB1", offered=[(640, 480, 30), (640, 480, 60), (320, 240, 30)])
    camera, _ = reader(FakeOb([device]), {"top_cam": "OB1"})

    with pytest.raises(
        RuntimeError,
        match=(
            r"offers no colour stream at 8x6@30; offered: "
            r"320x240@30, 640x480@30, 640x480@60"
        ),
    ) as info:
        camera(cfg())
    assert isinstance(info.value.__cause__, RuntimeError)


def test_device_offering_nothing_says_none() -> None:
    camera, _ = reader(FakeOb([FakeDevice("OB1", offered=[])]), {"top_cam": "OB1"})

    with pytest.raises(RuntimeError, match="offered: none"):
        camera(cfg())


def test_warm_up_retries_timeouts_and_incomplete_sets() -> None:
    device = FakeDevice(
        "OB1",
        responses=[
            None,
            FrameSet(rgb_frame(), DepthFrame(np.zeros((6, 8), np.uint16)), unalignable=True),
            FrameSet(None, DepthFrame(np.zeros((6, 8), np.uint16))),
            FrameSet(rgb_frame(), None),
            frames(fill=55),
        ],
    )
    camera, ob = reader(FakeOb([device]), {"top_cam": "OB1"})

    assert camera(cfg())["top_cam"][0, 0, 0] == 55
    assert ob.pipelines[0].waits >= 5


def test_drain_skips_timeouts_and_publishes_later_sets() -> None:
    device = FakeDevice("OB1")
    camera, ob = reader(FakeOb([device]), {"top_cam": "OB1"})
    camera(cfg())
    device.responses = [None, frames(fill=88)]

    deadline = time.monotonic() + 2.0
    while camera(cfg())["top_cam"][0, 0, 0] != 88:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert ob.pipelines[0].waits > 2


def test_drain_exception_is_latched_and_reraised() -> None:
    device = FakeDevice("OB1")
    camera, _ = reader(FakeOb([device]), {"top_cam": "OB1"})
    camera(cfg())
    boom = OSError("usb gone")
    device.responses = [boom]

    deadline = time.monotonic() + 2.0
    while "top_cam" not in camera._faults:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    with pytest.raises(RuntimeError, match=r"camera top_cam \(OB1\) stopped reading") as info:
        camera(cfg())
    assert info.value.__cause__ is boom


def test_depth_with_wrong_byte_count_fails_the_open_and_stops_the_pipeline() -> None:
    bad = FrameSet(rgb_frame(), DepthFrame(np.zeros((6, 8), np.uint16)))
    bad.depth.get_data = lambda: np.zeros(7, dtype=np.uint8)  # type: ignore[union-attr,method-assign]
    device = FakeDevice("OB1", responses=[bad])
    camera, ob = reader(FakeOb([device]), {"top_cam": "OB1"})

    with pytest.raises(ValueError, match="has 7 bytes, expected 96 for 8x6 uint16"):
        camera(cfg())
    assert ob.pipelines[0].stopped


def test_open_failure_rolls_back_pipelines_that_started() -> None:
    devices = [FakeDevice("OB1"), FakeDevice("OB2")]
    devices[1].start_error = RuntimeError("busy")
    camera, ob = reader(FakeOb(devices), {"top_cam": "OB1", "left_cam": "OB2"})

    with pytest.raises(RuntimeError, match="busy"):
        camera(cfg())
    assert ob.pipelines[0].stopped
    assert camera._bundles == {}
    assert camera._published == {}


def test_retired_generation_does_not_publish() -> None:
    camera, _ = reader(FakeOb([FakeDevice("OB1")]), {"top_cam": "OB1"})
    camera(cfg())
    bundle = camera._bundles["top_cam"]
    assert isinstance(bundle, _OrbbecBundle)
    before = camera._published["top_cam"]

    assert camera._publish_set("top_cam", bundle, frames(fill=99), camera._generation - 1)
    assert camera._published["top_cam"] is before


def test_close_stops_pipelines_and_logs_stop_errors(caplog: pytest.LogCaptureFixture) -> None:
    devices = [FakeDevice("OB1"), FakeDevice("OB2")]
    devices[1].stop_error = RuntimeError("stop failed")
    camera, ob = reader(FakeOb(devices), {"top_cam": "OB1", "left_cam": "OB2"})
    camera(cfg())

    with caplog.at_level(logging.ERROR):
        camera.close()
    camera.close()

    assert all(pipeline.stopped for pipeline in ob.pipelines)
    assert "stopping Orbbec pipeline for left_cam failed" in caplog.text


def test_close_leaves_a_still_reading_pipeline_open(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera, ob = reader(FakeOb([FakeDevice("OB1")]), {"top_cam": "OB1"})
    camera(cfg())
    camera._stop.set()
    camera._threads["top_cam"].join(timeout=2.0)
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, daemon=True)
    stuck.start()
    camera._threads["top_cam"] = stuck
    monkeypatch.setattr(embodiment_module, "JOIN_TIMEOUT_S", 0.01)
    try:
        with caplog.at_level(logging.WARNING):
            camera.close()
    finally:
        release.set()

    assert not ob.pipelines[0].stopped
    assert "leaving the Orbbec pipeline open" in caplog.text


# --------------------------------------------------------------------------- colour formats


def decoder() -> _OrbbecCameraReader:
    camera, _ = reader()
    return camera


RGB_IMAGE = np.array(
    [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]] * 2, dtype=np.uint8
)  # 2 rows x 4 columns: red, green, blue, white


def test_bgr_frames_are_swapped_to_rgb() -> None:
    bgr = RGB_IMAGE[..., ::-1].copy()
    out = decoder()._colour_rgb(ColourFrame("BGR", bgr.reshape(-1), 4, 2))

    np.testing.assert_array_equal(out, RGB_IMAGE)


@pytest.mark.parametrize("fmt", ["YUYV", "YUY2"])
def test_yuyv_frames_decode_to_rgb(fmt: str) -> None:
    grey = np.full((2, 4, 3), 128, dtype=np.uint8)
    # YUYV for mid-grey: Y=128 and U=V=128 on every byte.
    packed = np.full((2, 4, 2), 128, dtype=np.uint8)
    out = decoder()._colour_rgb(ColourFrame(fmt, packed.reshape(-1), 4, 2))

    assert out.shape == (2, 4, 3)
    np.testing.assert_allclose(out.astype(int), grey.astype(int), atol=2)


def test_uyvy_frames_decode_to_rgb() -> None:
    packed = np.full((2, 4, 2), 128, dtype=np.uint8)
    packed[..., 1] = 235  # UYVY: luma is the second byte of each pair, so this is bright
    out = decoder()._colour_rgb(ColourFrame("UYVY", packed.reshape(-1), 4, 2))

    assert out.shape == (2, 4, 3)
    assert out.min() > 200


def test_mjpg_frames_decode_to_rgb() -> None:
    solid = np.zeros((16, 16, 3), dtype=np.uint8)
    solid[..., 0] = 220  # red in RGB
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(solid, cv2.COLOR_RGB2BGR))
    assert ok

    out = decoder()._colour_rgb(ColourFrame("MJPG", encoded.reshape(-1), 16, 16))

    assert out.shape == (16, 16, 3)
    assert out[8, 8, 0] > 180
    assert out[8, 8, 2] < 40


def test_undecodable_mjpg_is_an_error() -> None:
    junk = np.zeros(32, dtype=np.uint8)

    with pytest.raises(ValueError, match="MJPG colour frame could not be decoded"):
        decoder()._colour_rgb(ColourFrame("MJPG", junk, 4, 2))


def test_unsupported_colour_format_names_the_supported_ones() -> None:
    with pytest.raises(ValueError, match="unsupported Orbbec colour format NV12; expected RGB"):
        decoder()._colour_rgb(ColourFrame("NV12", np.zeros(12, np.uint8), 4, 2))


def test_cv2_resolves_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    camera, _ = reader()
    camera._cv2 = None
    monkeypatch.setattr(embodiment_module, "_import_cv2", lambda: cv2)

    camera._colour_rgb(ColourFrame("BGR", np.zeros(24, np.uint8), 4, 2))

    assert camera._cv2 is cv2


def test_install_command_skips_the_bindings_example_dependencies() -> None:
    assert "--no-deps" in ORBBEC_INSTALL_COMMAND
    assert "pyorbbecsdk2" in ORBBEC_INSTALL_COMMAND


# --------------------------------------------------------------------------- config


def test_orbbec_serials_are_a_camera_source() -> None:
    config = YamConfig(
        top_orbbec_serial="OB1", left_orbbec_serial="OB2", right_cam_device="/dev/right"
    )

    assert config.top_orbbec_serial == "OB1"
    assert config.left_orbbec_serial == "OB2"


@pytest.mark.parametrize("other", ["cam_device", "depth_serial"])
def test_orbbec_serial_excludes_other_sources_for_the_slot(other: str) -> None:
    with pytest.raises(ValueError, match=f"top_{other} and top_orbbec_serial are mutually"):
        YamConfig(
            top_orbbec_serial="OB1",
            left_orbbec_serial="OB2",
            right_orbbec_serial="OB3",
            **{f"top_{other}": "X"},
        )


def test_duplicate_orbbec_serials_are_rejected() -> None:
    with pytest.raises(
        ValueError, match="top_orbbec_serial and left_orbbec_serial must be different; duplicate"
    ):
        YamConfig(top_orbbec_serial="OB1", left_orbbec_serial="OB1", right_orbbec_serial="OB3")


def test_empty_orbbec_serial_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_orbbec_serial must be a non-empty string"):
        YamConfig(top_orbbec_serial=" ")


def test_partial_orbbec_rig_must_source_every_slot() -> None:
    with pytest.raises(ValueError, match="unsourced slots: left, right"):
        YamConfig(top_orbbec_serial="OB1")


def test_depth_capture_override_applies_to_an_orbbec_slot() -> None:
    config = YamConfig(
        top_orbbec_serial="OB1",
        left_orbbec_serial="OB2",
        right_orbbec_serial="OB3",
        top_depth_capture_width=320,
        top_depth_capture_height=240,
    )

    assert config.depth_capture_size_for("top_cam") == (320, 240)


def test_depth_capture_override_on_a_v4l2_slot_names_both_depth_sources() -> None:
    with pytest.raises(ValueError, match="set top_depth_serial or top_orbbec_serial"):
        YamConfig(top_depth_capture_width=320, top_depth_capture_height=240)


@pytest.mark.parametrize("field", ["top_orbbec_serial", "right_orbbec_serial"])
def test_from_kwargs_guides_int_orbbec_serials(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a string; quote the serial"):
        YamConfig.from_kwargs(**{field: 1234})


# --------------------------------------------------------------------------- embodiment


def driver_factory(_cfg: YamConfig) -> Any:
    class Driver:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(14)

        def get_joint_eff(self) -> np.ndarray:
            return np.zeros(14)

        def get_motor_temps(self) -> np.ndarray:
            return np.full(14, 30.0)

        def command_joint_pos(self, target: np.ndarray) -> None:
            del target

        def close(self) -> None:
            pass

    return Driver()


class FakeV4l2Capture:
    def isOpened(self) -> bool:
        return True

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def read(self) -> tuple[bool, npt.NDArray[np.uint8]]:
        time.sleep(0.002)
        return True, np.full((480, 640, 3), 9, dtype=np.uint8)

    def release(self) -> None:
        pass


class V4l2Cv2:
    """Real OpenCV for conversions, with V4L2 capture replaced by a fake device."""

    def __getattr__(self, name: str) -> Any:
        return getattr(cv2, name)

    def VideoCapture(self, _device: str, _api: int) -> FakeV4l2Capture:
        return FakeV4l2Capture()


def builtin_embodiment(
    config: YamConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[YAMEmbodiment, FakeOb]:
    ob = FakeOb()
    monkeypatch.setattr(embodiment_module, "_import_ob", lambda: ob)
    monkeypatch.setattr(embodiment_module, "_import_cv2", lambda: V4l2Cv2())
    emb = YAMEmbodiment(
        config,
        driver_factory=driver_factory,
        operator=OperatorIO(input_fn=lambda _: "", output_fn=lambda _: None),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )
    return emb, ob


def observe_once(emb: YAMEmbodiment) -> Any:
    emb._driver = driver_factory(emb._cfg)
    try:
        return emb._observe("task")
    finally:
        emb._driver = None
        emb.close()


def test_orbbec_only_rig_observes_images_depth_and_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cfg(
        top_orbbec_serial="OB1",
        left_orbbec_serial="OB2",
        right_orbbec_serial="OB3",
    )
    emb, ob = builtin_embodiment(config, monkeypatch)
    assert isinstance(emb._builtin_orbbec_reader, _OrbbecCameraReader)
    assert emb._builtin_realsense_reader is None

    observation = observe_once(emb)

    assert set(observation.images) == set(SERIALS)
    assert observation.extra is not None
    for name in SERIALS:
        assert observation.images[name].shape == (3, 4, 3)
        # Lazy depth is bound to the open cycle; observe_once has closed it.
        assert callable(observation.extra[f"{name}_depth"])
        assert observation.extra[f"{name}_intrinsics"].shape == (3, 3)
    assert all(pipeline.stopped for pipeline in ob.pipelines)


def test_mixed_v4l2_and_orbbec_rig_serves_depth_only_for_orbbec_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cfg(
        top_cam_device="/dev/top",
        left_orbbec_serial="OB2",
        right_orbbec_serial="OB3",
    )
    emb, _ = builtin_embodiment(config, monkeypatch)
    assert isinstance(emb._camera_reader, _CompositeCameraReader)

    observation = observe_once(emb)

    assert set(observation.images) == set(SERIALS)
    assert observation.extra is not None
    assert "top_cam_depth" not in observation.extra
    assert callable(observation.extra["left_cam_depth"])
    assert "Depth: for each serial-configured camera (left_cam, right_cam)" in emb.info.docs


def test_custom_camera_reader_conflicts_with_orbbec_serials() -> None:
    config = YamConfig(top_orbbec_serial="OB1", left_orbbec_serial="OB2", right_orbbec_serial="OB3")

    with pytest.raises(ValueError, match="conflicts with configured \\*_depth_serial or"):
        YAMEmbodiment(config, camera_reader=lambda _cfg: {})


def test_unconfigured_cameras_error_names_the_orbbec_source() -> None:
    emb = YAMEmbodiment(
        YamConfig(),
        driver_factory=driver_factory,
        operator=OperatorIO(input_fn=lambda _: "", output_fn=lambda _: None),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )
    from inspect_robots.errors import ConfigError
    from inspect_robots.scene import Scene

    with pytest.raises(ConfigError, match=r"\*_orbbec_serial for Orbbec"):
        emb.reset(Scene(id="s", instruction="task"))


# --------------------------------------------------------------------------- health


def test_health_reports_orbbec_slots_as_unchecked() -> None:
    config = YamConfig(
        top_cam_device="/dev/top", left_orbbec_serial="OB2", right_depth_serial="RS3"
    )

    assert [camera.name for camera in _unchecked_depth_cameras(config)] == [
        "left_cam",
        "right_cam",
    ]


def test_health_keeps_an_all_digit_orbbec_serial_a_string() -> None:
    seen: list[YamConfig] = []

    def capture(cfg_value: YamConfig, **_kwargs: object) -> HealthReportType:
        seen.append(cfg_value)
        return HealthReportType((), False, (), True, None)

    args = [
        "-E",
        "top_orbbec_serial=0123",
        "--left-cam",
        "/dev/left",
        "--right-cam",
        "/dev/right",
        "--skip-motors",
    ]
    assert main(args, run=capture) == 0
    assert seen[0].top_orbbec_serial == "0123"


def test_camera_that_misses_every_warm_up_wait_is_served_by_its_drain() -> None:
    device = FakeDevice("OB1", responses=[None] * 10)
    # A real sleep: the consumer waits for the drain thread's first publish.
    camera, ob = reader(FakeOb([device]), {"top_cam": "OB1"}, sleep_fn=time.sleep)

    image = camera(cfg())["top_cam"]

    assert image.shape == (3, 4, 3)
    assert ob.pipelines[0].waits > 10


def test_retired_drain_does_not_fault_the_current_cycle() -> None:
    camera, _ = reader(FakeOb([]), {"top_cam": "OB1"})
    bundle = _OrbbecBundle(
        pipeline=FakePipeline(FakeDevice("OB1", responses=[OSError("old cycle")])),
        align=FakeAlign("COLOR_STREAM"),
        depth_scale=0.0,
        intrinsics=np.eye(3, dtype=np.float32),
    )

    camera._drain("top_cam", bundle, threading.Event(), camera._generation - 1)

    assert camera._faults == {}
