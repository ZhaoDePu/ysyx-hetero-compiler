"""Offline ASan/UBSan loopback using the C8 model and frozen image fixtures."""

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image


HEADER = struct.Struct("<4sHHQ8I5Q64s")
TIMING = struct.Struct("<4sQ6I2H2B")
FEED = struct.Struct("<4sIII")
DETS = struct.Struct("<6f")
HOST = "127.0.0.1"
PORT_OUT, PORT_IN = 9000, 9001
SANITIZERS = {
    "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
    "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
}


def exact(conn, size):
    result = bytearray(size)
    view = memoryview(result)
    while view:
        n = conn.recv_into(view)
        if not n:
            raise EOFError("Disconnected")
        view = view[n:]
    return result


def receive(conn, seq, w, h, codec, fault=False):
    header = bytes(exact(conn, HEADER.size))
    magic, version, size, number, width, height, kind, length, count, errors, error, flags, *extra = HEADER.unpack(header)
    assert (magic, version, size, number, width, height, kind) == (b"Z15F", 1, 152, seq, w, h, codec)
    assert count <= 300 and 0 < length <= 1920 * 1080 * 3
    assert (errors, error, flags) == (int(fault), 0, 4 | (2 if fault else 0)), (errors, error, flags)
    assert extra[:4] == [0] * 4
    name = extra[-1]
    assert name.split(b"\0", 1)[0] == b"YOLO26n-C8"
    assert name[:16] == b"YOLO26n-C8" + b"\0" * 6
    marker, previous, capture, h2d, pl, d2h, tx, e2e, npu_clock, gpu_clock, mode, v = TIMING.unpack_from(name, 16)
    assert (marker, previous, npu_clock, gpu_clock, v) == (b"C10T", seq - 1, 125, 100, 1)
    assert pl == 0xFFFFFFFF
    if seq == 1:
        assert capture == h2d == d2h == tx == e2e == 0xFFFFFFFF
    else:
        assert max(capture, h2d, d2h, tx) <= e2e < 0xFFFFFFFF
    payload = bytes(exact(conn, length))
    boxes = bytes(exact(conn, count * DETS.size))
    for x1, y1, x2, y2, score, cl in DETS.iter_unpack(boxes):
        assert all(math.isfinite(v) for v in (x1, y1, x2, y2, score, cl))
        assert 0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1 and 0.39 < score <= 1 and cl == int(cl) and 0 <= cl < 80
    return payload, list(DETS.iter_unpack(boxes)), mode, header + payload + boxes


def connect(port):
    stop = time.monotonic() + 12
    while True:
        try:
            conn = socket.create_connection((HOST, port), timeout=0.5)
            conn.settimeout(90)
            return conn
        except ConnectionRefusedError:
            if time.monotonic() >= stop:
                raise
            time.sleep(.05)


def launch(args, name, *options):
    log = args.out / f"{name}.service.log"
    with log.open("w") as f:
        return subprocess.Popen(
            [str(args.binary), "--model", str(args.model), "--sim", "--bind", HOST,
             "--port", str(PORT_OUT), "--in-port", str(PORT_IN),
             "--csv", str(args.out / f"{name}.csv"), *options], stdout=f, stderr=f,
            env=dict(os.environ, **SANITIZERS))


@contextmanager
def running(args, name, *options, status=0):
    proc = launch(args, name, *options)
    try:
        yield
        assert proc.wait(timeout=15) == status, name
        log = (args.out / f"{name}.service.log").read_text()
        assert "runtime error:" not in log and "AddressSanitizer" not in log, log
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def check_csv(args, name, count, verify="pass", fault=0, source="tcp"):
    with (args.out / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count
    for i, row in enumerate(rows, 1):
        assert None not in row and all(v is not None for v in row.values()), row
        assert row["backend"] == "fixed_point" and int(row["sequence"]) == i and row["source"] == source
        assert row["invalid_dma"] == row["abi_error"] == "0"
        assert row["verify"] == verify and row["fault"] == str(fault)
        assert row["network_complete"] == "1"
        assert all(row[field] == "" for field in ("t_pl_hw_ms", "total_cycles", "npu_busy", "vpu_busy",
                                                  "gpu_busy", "native_npu_vpu_busy", "dma_busy", "overlap"))
        t = float(row["t_e2e_ms"])
        assert all(0 <= float(row[field]) <= t for field in ("t_capture_preproc_ms", "t_h2d_ms",
                   "t_d2h_post_ms", "t_net_tx_ms", "t_execute_ms"))
        assert int(row["dynamic_h2d_bytes"]) > 0
    if count > 1:
        assert int(rows[0]["static_h2d_bytes"]) > 0 and rows[1]["static_h2d_bytes"] == "0"
    return rows


def load_viewer(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viewers(args, packet, n):
    """Exercise the original parsers with real frames; use Pillow for image conversion if OpenCV is absent."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_RGB2BGR, cv2.IMREAD_COLOR = 1, 1
        cv2.cvtColor = lambda a, _: a[..., ::-1].copy()
        cv2.imdecode = lambda buf, _: np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))[..., ::-1].copy()
        sys.modules["cv2"] = cv2

    class Socket:
        def __init__(self, packet):
            self.data = io.BytesIO(packet)

        def recv_into(self, dest):
            return self.data.readinto(dest)

    legacy = load_viewer(args.legacy_viewer).read_frame(Socket(packet))
    enhanced = load_viewer(args.viewer).read_frame(Socket(packet))
    assert legacy[0] == enhanced[0] == n
    assert legacy[2:] == enhanced[2:12]
    assert np.array_equal(legacy[1], enhanced[1])
    assert enhanced[12]["sequence"] == n - 1
    assert enhanced[12]["source"] in ("TCP", "recorded YUYV")


def packing_yuyv(image):
    data = np.asarray(image, dtype=np.int32)
    r, g, b = (data[..., i] for i in range(3))
    y = np.clip(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16, 16, 235)
    u = np.clip(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 16, 240)
    v = np.clip(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 16, 240)
    return np.stack((y[:, ::2], (u[:, ::2] + u[:, 1::2] + 1) // 2,
                     y[:, 1::2], (v[:, ::2] + v[:, 1::2] + 1) // 2), axis=-1).astype("u1").tobytes()


def prepare(args):
    gold = args.out / "gold"
    gold.mkdir(exist_ok=True)
    raw = []
    recording = args.out / "bus-zidane-640x480.yuyv"
    with recording.open("wb") as f:
        for seq, stem in enumerate(("bus", "zidane"), 1):
            raw.append((args.vectors / f"{stem}.rgb_nhwc_u8.bin").read_bytes())
            heads = b"".join((args.vectors / f"{stem}.pl_out_{i}.bin").read_bytes() for i in range(6))
            assert len(raw[-1]) == 320 * 320 * 3 and len(heads) == 176400
            (gold / f"{seq}.heads.bin").write_bytes(heads)
            with Image.open(args.images / f"{stem}.jpg") as original:
                image = original.convert("RGB")
                image.thumbnail((640, 480), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (640, 480), (114, 114, 114))
                canvas.paste(image, ((640 - image.width) // 2, (480 - image.height) // 2))
                f.write(packing_yuyv(canvas))
    return raw, gold, recording


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "vectors", "images", "legacy-viewer", "viewer", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw, gold, recording = prepare(a)
    evidence = {}
    with running(a, "tcp", "--source", "tcp", "--verify", "--golden", str(gold), "--frames", "2"):
        # A viewer may disconnect before opening its feeder.
        connect(PORT_OUT).close()
        with connect(PORT_OUT) as viewer, connect(PORT_IN) as feeder:
            # A benign bad extent is rejected before allocation; the next client may reconnect.
            feeder.sendall(FEED.pack(b"Z15I", 1921, 320, 1))
            assert viewer.recv(1) == b""
        counts = []
        for i, frame in enumerate(raw, 1):
            with connect(PORT_OUT) as viewer, connect(PORT_IN) as feeder:
                feeder.sendall(FEED.pack(b"Z15I", 320, 320, len(frame)))
                feeder.sendall(frame)
                payload, boxes, mode, packet = receive(viewer, i, 320, 320, 1)
                assert payload == frame and boxes and mode == 5
                counts.append(len(boxes))
                test_viewers(a, packet, i)
    evidence["tcp"] = dict(frames=len(raw), detection_counts=counts, byte_exact_heads=176400 * len(raw),
                           csv=check_csv(a, "tcp", len(raw)), legacy_parser=True, enhanced_parser=True)
    print("C10 RGB TCP 9001/9000, reject/reconnect, two 176400-byte golden checks, legacy/enhanced parsers PASS",
          flush=True)

    with running(a, "yuyv", "--source", "v4l2", "--live", "--recorded-yuyv", str(recording),
                 "--width", "640", "--height", "480", "--frames", "2"):
        frames = []
        with connect(PORT_OUT) as viewer:
            for i in (1, 2):
                pixels, boxes, mode, packet = receive(viewer, i, 320, 320, 1)
                assert len(pixels) == 320 * 320 * 3 and boxes and mode == 14
                test_viewers(a, packet, i)
                frames.append((pixels, len(boxes)))
        assert frames[0][0] != frames[1][0]
    evidence["recorded_yuyv"] = dict(frames=2, detection_counts=[f[1] for f in frames],
                                    csv=check_csv(a, "yuyv", 2, verify="skipped", source="recorded_yuyv"))
    print("C10 bus/zidane 640x480 YUYV -> 320x320 RGB and distinct detection frames PASS", flush=True)

    jpeg = (a.images / "zidane.jpg").read_bytes()
    w, h = Image.open(io.BytesIO(jpeg)).size
    with running(a, "jpeg", "--source", "tcp", "--live", "--frames", "1"):
        with connect(PORT_OUT) as viewer, connect(PORT_IN) as feeder:
            feeder.sendall(FEED.pack(b"Z15J", w, h, len(jpeg)) + jpeg)
            payload, boxes, mode, packet = receive(viewer, 1, w, h, 2)
            assert payload == jpeg and boxes and mode == 4
            test_viewers(a, packet, 1)
    evidence["baseline_jpeg"] = dict(frames=1, detection_count=len(boxes),
                                    csv=check_csv(a, "jpeg", 1, verify="skipped"))
    print("C10 original baseline JPEG bytes, C99 decode, Z15J and Z15F JPEG echo PASS", flush=True)

    tampered = a.out / "tampered"
    tampered.mkdir(exist_ok=True)
    invalid = bytearray((gold / "1.heads.bin").read_bytes())
    invalid[0] ^= 1
    (tampered / "1.heads.bin").write_bytes(invalid)
    with running(a, "mismatch", "--source", "tcp", "--verify", "--golden", str(tampered),
                 "--frames", "1", status=1):
        with connect(PORT_OUT) as viewer, connect(PORT_IN) as feeder:
            feeder.sendall(FEED.pack(b"Z15I", 320, 320, len(raw[0])) + raw[0])
            _, boxes, _, _ = receive(viewer, 1, 320, 320, 1, fault=True)
            assert not boxes
    evidence["golden_mismatch"] = dict(csv=check_csv(a, "mismatch", 1, verify="fail", fault=1))
    print("C10 byte mismatch detected, fault sent, process exit 1 PASS", flush=True)
    truncated = a.out / "truncated.yuyv"
    truncated.write_bytes(recording.read_bytes()[:1024])
    with running(a, "truncated", "--source", "v4l2", "--recorded-yuyv", str(truncated),
                 "--width", "640", "--height", "480", "--frames", "1", status=1):
        with connect(PORT_OUT) as viewer:
            assert viewer.recv(1) == b""
    evidence["truncated_yuyv"] = dict(rejected=True, csv=check_csv(a, "truncated", 0))
    print("C10 truncated recorded YUYV, no emitted frame, process exit 1 PASS", flush=True)
    invalid_cli = [("--verify",), ("--live", "--verify"), ("--source", "usb"),
                   ("--source", "v4l2", "--recorded-yuyv", str(recording), "--width", "319"),
                   ("--freq-profile", "199_149"), ("--fps", "0"), ("--port", "9001")]
    for flags in invalid_cli:
        proc = subprocess.run([str(a.binary), "--model", str(a.model), *flags],
                              capture_output=True, env=dict(os.environ, **SANITIZERS))
        assert proc.returncode == 2 and b"z15_yolo_live_c10 --model" in proc.stderr
    evidence["invalid_cli"] = dict(rejected=len(invalid_cli))
    print("C10 CLI 7 invalid option combinations rejected before setup PASS", flush=True)
    (a.out / "loopback.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
