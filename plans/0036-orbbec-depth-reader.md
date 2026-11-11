# Orbbec depth reader

Status: implemented. Enumeration and profiles verified on three Gemini 305s
(every unit exposes `COLOR_SENSOR`, `DEPTH_SENSOR`, `LEFT_IR_SENSOR`,
`RIGHT_IR_SENSOR`, with colour and depth both at 640x480@30); streaming in an
eval not yet verified.

## Problem

`yam_arms` serves aligned depth only through librealsense (`*_depth_serial`).
Rigs built with Orbbec cameras, such as three Gemini 305 units, can reach the
plugin only through `*_cam_device`, which reads the colour node over V4L2 and
drops depth. Policies that read depth (the agent policy renders
`extra["{cam}_depth"]`, CaP-X grasp planning needs depth and intrinsics) get
nothing from those rigs.

Reading the raw V4L2 nodes is not an answer. A Gemini exposes depth (`Z16`),
IR (`GREY`), raw Bayer and colour as separate nodes, and the depth image is in
the depth sensor's frame at its own resolution. Pixel alignment to the colour
image needs the device calibration, which only the SDK applies.

## Design

### Configuration

`top_orbbec_serial`, `left_orbbec_serial`, `right_orbbec_serial`: plain
strings, default `None`. Validation mirrors `*_depth_serial`:

- non-empty; all-digit values arriving as ints from config.ini are rejected
  with the "quote the serial" hint;
- no two slots share a serial;
- a slot takes exactly one of `*_cam_device`, `*_depth_serial`,
  `*_orbbec_serial`;
- an Orbbec slot counts as sourced for the all-or-none rule;
- `{slot}_depth_capture_*` overrides are allowed on Orbbec slots.

`capture_*`, `depth_capture_*` and `depth_fps` keep their meaning. No new size
or rate keys.

### Reader

`_OrbbecCameraReader` subclasses the inline `_RealsenseCameraReader`. The
parent already owns the parts that are device independent and tested:
per-camera published slots under one lock, generation fencing, the staleness
wait in `_latest`, intrinsics scaling to the published size in `extra()`, lazy
nearest-neighbour depth thunks, and the join-before-stop `close()`. The
subclass replaces only:

- `_open_all`: enumerate with `Context().query_devices()`, match serials,
  list visible `name / serial` pairs when one is missing, all-or-nothing open.
- `_open_one`: require a `COLOR_SENSOR`, select colour and depth profiles at
  the configured size and `depth_fps` in explicit formats (colour RGB, then
  BGR, YUYV, MJPG; depth Y16), since the Gemini colour sensor also offers
  Y8/Y16 greyscale at every size (listing offered profiles on failure),
  start with `FULL_FRAME_REQUIRE` aggregation, read K from the colour profile,
  build `AlignFilter(COLOR_STREAM)`, warm up for ten waits.
- `_drain` / `_publish_set`: `wait_for_frames(1000)`, align, take colour and
  depth, convert depth bytes to uint16 with a size check, store the per-frame
  scale converted from millimetres to metres.
- `_colour_rgb`: RGB, BGR, YUYV/YUY2, UYVY and MJPG to RGB through OpenCV.

Two small hooks were added to the parent so the subclass needs no copy of its
logic: `_import_sdk()` (pragma'd seam) and `_SDK_NAME` for log messages.

The embodiment builds the reader when any Orbbec serial is set, composes it
with the V4L2 and RealSense readers, merges its `extra()` into observations,
and includes Orbbec cameras in the depth docs paragraph. A custom
`camera_reader` conflicts with Orbbec serials, as with RealSense serials.

### Dependency

No pip extra. pyorbbecsdk2 2.1.3 declares opencv-python, open3d, pygame,
pynput, and `numpy<2` on Python 3.11 as requirements. As an extra it would
install a second `cv2` next to opencv-python-headless and pin numpy on the rig.
The binding imports and enumerates devices with only numpy 2 installed
(checked on 2.1.3), so the documented install is
`uv pip install --no-deps 'pyorbbecsdk2>=2.1.3'`, held in
`ORBBEC_INSTALL_COMMAND` and printed by the import error.

## Decisions and limits

- **In-process capture.** RealSense defaults to a capture child to keep frame
  work out of the motor-control interpreter. Porting that to Orbbec doubles the
  scope. Whether pyorbbecsdk releases the GIL during `wait_for_frames` is not
  known; if hardware runs show control-loop jitter, a capture child is the
  follow-up.
- **No stereo-RGB fallback.** `AlignFilter` in 2.1.3 rejects
  `LEFT_COLOR_STREAM` ("AlignType value 11 out of range [1, 7]"). A device
  without a single colour sensor is rejected at open with its sensor list
  rather than served unaligned depth. Whether the Gemini 305 needs this path
  is to be confirmed with `scripts/orbbec_probe.py`.
- **Health and setup.** The health tool reports Orbbec slots as unchecked,
  as it does RealSense slots; the setup wizard does not enumerate them.

## Verification

Unit tests (`tests/test_orbbec_reader.py`) drive a recording fake whose method
names were checked against the real pyorbbecsdk2 2.1.3 binding, with real
OpenCV for colour decoding. Every SDK call the reader makes was also checked to
exist, and the enum and filter constructions to succeed, on the real binding
without a camera attached. Not yet run against a physical Gemini 305.
