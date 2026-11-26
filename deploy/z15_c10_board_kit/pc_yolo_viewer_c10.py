"""Z15 C10 viewer and RGB/JPEG feeder. Requires Python 3, NumPy, OpenCV and Pillow for CJK labels."""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time

import cv2
import numpy as np

COCO_EN = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",")
COCO_ZH = "人,自行车,汽车,摩托车,飞机,公交车,火车,卡车,船,交通信号灯,消防栓,停车标志,停车计时器,长凳,鸟,猫,狗,马,羊,牛,大象,熊,斑马,长颈鹿,背包,雨伞,手提包,领带,行李箱,飞盘,双板滑雪板,单板滑雪板,球,风筝,棒球棒,棒球手套,滑板,冲浪板,网球拍,瓶子,葡萄酒杯,杯子,叉子,刀,勺子,碗,香蕉,苹果,三明治,橙子,西兰花,胡萝卜,热狗,披萨,甜甜圈,蛋糕,椅子,沙发,盆栽,床,餐桌,马桶,电视,笔记本电脑,鼠标,遥控器,键盘,手机,微波炉,烤箱,烤面包机,水槽,冰箱,书,钟,花瓶,剪刀,泰迪熊,吹风机,牙刷".split(",")
HEADER = struct.Struct("<4sHHQ8I5Q64s")
DETECTION = struct.Struct("<6f")
FEED = struct.Struct("<4sIII")
RGB, JPEG = 1, 2
SIMULATED, FAULT, COUNTERS_UNAVAILABLE = 1, 2, 4
MAX_BOXES = 512
C10_TIMING = struct.Struct("<4sQ6I2H2B")
TIMING_NAMES = (
    "t_capture_preproc_ms", "t_h2d_ms", "t_pl_hw_ms",
    "t_d2h_post_ms", "t_net_tx_ms", "t_e2e_ms",
)


def read_timing(name, sequence):
    """The extension lies after the first NUL in the original name field."""
    if b"\0" not in name[:16]:
        return None
    marker, sample, *suffix = C10_TIMING.unpack_from(name, 16)
    times = suffix[:6]
    native_mhz, gpu_mhz, mode, version = suffix[6:]
    if marker != b"C10T" or version != 1:
        return None
    if sample >= sequence or (native_mhz, gpu_mhz) not in {(125, 100), (200, 150)} or mode & ~15:
        raise ValueError("invalid C10 timing extension")
    timing = dict(zip(TIMING_NAMES, (None if t == 0xffffffff else t / 1000 for t in times)))
    source = "recorded YUYV" if mode & 8 else "V4L2" if mode & 2 else "TCP"
    return dict(timing, sequence=sample, native_mhz=native_mhz, gpu_mhz=gpu_mhz,
                mode="verify" if mode & 1 else "live", source=source, simulation=bool(mode & 4))

def read_exact(sock, size):
    buf = bytearray(size)
    view = memoryview(buf)
    while view:
        got = sock.recv_into(view)
        if not got:
            raise EOFError("stream closed")
        view = view[got:]
    return buf

def jpeg_dimensions(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("JPEG SOI missing")
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 255:
            break
        marker = data[pos + 1]
        pos += 2
        if marker == 255:
            pos -= 1
            continue
        size = int.from_bytes(data[pos:pos + 2], "big")
        if size < 2 or pos + size > len(data):
            break
        if marker in (192, 193, 194) and size >= 8:
            height, width = struct.unpack_from(">HH", data, pos + 3)
            return width, height
        if marker == 218:
            break
        pos += size
    raise ValueError("JPEG SOF missing or unsupported")

def read_frame(sock):
    (magic, version, header_size, sequence, width, height, codec, size, count,
     error_count, error, flags, npu, gpu, wall, overlap, duration, name) = HEADER.unpack(
        read_exact(sock, HEADER.size))
    if (magic != b"Z15F" or version != 1 or header_size != HEADER.size or not sequence
            or not 0 < width <= 1920 or not 0 < height <= 1080
            or not 0 < size <= 1920 * 1080 * 3 or count > MAX_BOXES
            or flags & ~(SIMULATED | FAULT | COUNTERS_UNAVAILABLE) or codec not in (RGB, JPEG)
            or codec == RGB and size != width * height * 3):
        raise ValueError("invalid Z15 header")
    payload = read_exact(sock, size)
    if codec == RGB:
        rgb = np.frombuffer(payload, np.uint8).reshape(height, width, 3)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        if jpeg_dimensions(payload) != (width, height):
            raise ValueError("JPEG dimensions disagree with stream")
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError("invalid JPEG payload")
    boxes = list(DETECTION.iter_unpack(read_exact(sock, count * DETECTION.size)))
    for x1, y1, x2, y2, score, cls in boxes:
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, cls))
                or not 0 <= x1 <= x2 <= 1 or not 0 <= y1 <= y2 <= 1
                or not 0 <= score <= 1 or not 0 <= cls < 80 or cls != int(cls)):
            raise ValueError("invalid detection")
    return (sequence, image, boxes, npu, gpu, wall, overlap, duration, error_count,
            error, flags, name.split(b"\0", 1)[0].decode("utf-8", "replace"), read_timing(name, sequence))

@lru_cache(maxsize=8)
def label_font(size, explicit):
    from PIL import ImageFont
    fonts = [explicit] if explicit else [
        str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for font in fonts:
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    raise RuntimeError("CJK font missing. Supply --font or use --label-lang en.")


def draw(frame, threshold, fps, lang="both", font_path=None):
    _, img, boxes, npu, gpu, wall, overlap, duration, errors, fault, flags, name, timing = frame
    h, w = img.shape[:2]
    labels = []
    for x1, y1, x2, y2, score, cls_id in boxes:
        if score < threshold:
            continue
        cls = int(cls_id)
        color = (60 + cls * 73 % 195, 170 - cls * 17 % 100, 255 - cls * 43 % 195)
        p = (round(x1 * (w - 1)), round(y1 * (h - 1)))
        q = (round(x2 * (w - 1)), round(y2 * (h - 1)))
        cv2.rectangle(img, p, q, color, 2, cv2.LINE_AA)
        label = COCO_EN[cls]
        if lang == "zh":
            label = COCO_ZH[cls]
        elif lang == "both":
            label = f"{COCO_ZH[cls]} / {label}"
        labels.append((p, color, f"{label} {score:.2f}"))

    panel = [f"End-to-end FPS  {fps:.1f}" if fps else "End-to-end FPS  -", f"Model  {name[:24]}"]
    if flags & SIMULATED:
        panel.append("DRY RUN | SYNTHETIC BOXES")
    elif flags & COUNTERS_UNAVAILABLE:
        panel.append("FIXED-POINT CPU REFERENCE")
    if timing is None:
        if not flags & COUNTERS_UNAVAILABLE:
            panel.append(f"PL wall {wall:,} ticks")
        if duration:
            panel.append(f"Inference {duration / 1e6:.2f} ms")
    else:
        clocks = f"{timing['native_mhz']}/{timing['gpu_mhz']} MHz"
        panel.append(f"{timing['mode'].upper()} | {timing['source']} | {clocks}")
        pl_ms = None if flags & COUNTERS_UNAVAILABLE else wall / (timing["native_mhz"] * 1000)
        ps_ms = None
        if timing["t_pl_hw_ms"] is not None and timing["t_e2e_ms"] is not None:
            ps_ms = timing["t_e2e_ms"] - timing["t_pl_hw_ms"]
        def ms(value):
            return "-" if value is None else f"{value:.2f}"
        panel.append(f"PL {ms(pl_ms)} ms | PS (prev) {ms(ps_ms)} ms")
        if timing["sequence"]:
            panel.append(f"#{timing['sequence']} cap {ms(timing['t_capture_preproc_ms'])}"
                         f" H2D {ms(timing['t_h2d_ms'])} ms")
            panel.append(f"D2H {ms(timing['t_d2h_post_ms'])}"
                         f" TX {ms(timing['t_net_tx_ms'])}"
                         f" E2E {ms(timing['t_e2e_ms'])} ms")
    if flags & FAULT:
        panel.append(f"INFERENCE STOPPED | 0x{fault:08x}")
    font_scale = max(.36, min(.55, w / 1020))
    panel_width = 18 + max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for line in panel)
    x = max(0, w - panel_width - 5)
    spacing = 19 if font_scale >= .46 else 16
    panel_bottom = min(h, 13 + spacing * len(panel))
    area = img[5:panel_bottom, x:max(x + 1, w - 5)]
    if area.size:
        cv2.addWeighted(area, .26, np.zeros_like(area), .74, 0, dst=area)
    for i, line in enumerate(panel):
        cv2.putText(img, line, (x + 7, 21 + spacing * i), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (241, 247, 251), 1, cv2.LINE_AA)
    if labels:
        img = draw_labels(img, labels, lang, font_path, (x, 5, w, panel_bottom))
    return img


def draw_labels(img, labels, lang, font_path, panel):
    h, w = img.shape[:2]
    font_size = max(14, min(22, w // 28))
    font = overlay = rgb = None
    if lang != "en":
        from PIL import Image, ImageDraw
        font = label_font(font_size, font_path)
        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        overlay = ImageDraw.Draw(rgb)
    obstacles = [panel]

    def overlaps(rect, other):
        return (rect[0] < other[2] and other[0] < rect[2] and
                rect[1] < other[3] and other[1] < rect[3])

    for (px, py), color, text in labels:
        if overlay is None:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0] + 10
        else:
            tw = round(overlay.textlength(text, font=font)) + 10
        th = font_size + 6
        x = max(0, min(px, w - tw - 1))
        y0 = py - th
        ys = (y0, py + 4, py + th + 5, panel[3] + 3, y0 - th - 2, py + 2 * th + 7)
        candidates = [max(0, min(y, h - th - 1)) for y in ys]
        y = next((n for n in candidates if not any(overlaps((x, n, x + tw, n + th), other)
                                                    for other in obstacles)), candidates[0])
        obstacles.append((x, y, x + tw, y + th))
        if overlay is None:
            cv2.rectangle(img, (x, y), (x + tw, y + th), (36, 24, 16), -1)
            cv2.putText(img, text, (x + 4, y + font_size), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        else:
            overlay.rounded_rectangle((x, y, x + tw, y + th - 1), radius=3, fill=(16, 24, 36))
            overlay.text((x + 4, y + 2), text, font=font, fill=color[::-1], anchor="lt")
    return img if rgb is None else cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


class StillImage:
    def __init__(self, path):
        self.image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Cannot decode image {path}")

    def read(self):
        image, self.image = self.image, None
        return image is not None, image

    def release(self):
        pass


def open_feed(args):
    if Path(args.feed).suffix.lower() in {".ppm", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return StillImage(args.feed)
    source = int(args.feed) if args.feed.isdecimal() else args.feed
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input {args.feed}")
    return capture


def send_input(input_sock, capture, size, encoding):
    ok, image = capture.read()
    if not ok:
        return False
    height, width = image.shape[:2]
    bound_w, bound_h = size
    ratio = min(1.0, bound_w / width, bound_h / height)
    if ratio < 1:
        width, height = max(1, round(width * ratio)), max(1, round(height * ratio))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if encoding == "jpeg":
        opts = [cv2.IMWRITE_JPEG_QUALITY, 100]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
            opts.extend((cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444))
        ok, payload = cv2.imencode(".jpg", image, opts)
        if not ok or payload.nbytes > 1920 * 1080 * 3:
            raise RuntimeError("Cannot encode frame inside the JPEG protocol bounds")
        magic = b"Z15J"
    else:
        payload = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        magic = b"Z15I"
    input_sock.sendall(FEED.pack(magic, width, height, payload.nbytes))
    input_sock.sendall(payload.tobytes())
    return True


def dump_frame(frame, stream):
    seq, image, boxes, npu, gpu, wall, overlap, duration, errors, fault, flags, model, timing = frame
    h, w = image.shape[:2]
    data = dict(frame=seq, width=w, height=h, model=model, flags=flags,
                npu_ticks=npu, gpu_ticks=gpu, wall_ticks=wall, overlap_ticks=overlap,
                inference_ms=duration / 1e6, error_count=errors, abi_error=fault, c10_timing=timing,
                detections=[dict(class_id=int(c), class_en=COCO_EN[int(c)], class_zh=COCO_ZH[int(c)],
                                 score=s, xyxy_normalized=[x1, y1, x2, y2])
                            for x1, y1, x2, y2, s, c in boxes])
    stream.write(json.dumps(data, ensure_ascii=False) + "\n")
    stream.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--conf-thresh", type=float, default=.25)
    p.add_argument("--record", help="Write rendered frames to an MP4")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int)
    p.add_argument("--feed", help="PC camera index, still image or video file for a board in --source tcp mode")
    p.add_argument("--in-port", type=int, default=9001)
    p.add_argument("--feed-size", default="native", help="native or a WxH maximum, preserving aspect ratio")
    p.add_argument("--feed-format", choices=("rgb", "jpeg"), default="rgb")
    p.add_argument("--label-lang", choices=("both", "en", "zh"), default="both")
    p.add_argument("--font", help="CJK font path for labels, default Windows Microsoft YaHei")
    p.add_argument("--snapshot", help="Save the final rendered frame as PNG")
    p.add_argument("--detections-jsonl", help="Write frame data and pre-overlay boxes to JSONL")
    p.add_argument("--timeout", type=float, default=5.0)
    args = p.parse_args()
    if not 0 <= args.conf_thresh <= 1 or args.frames is not None and args.frames <= 0:
        p.error("invalid threshold or frame limit")

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("invalid timeout")
    size = (1920, 1080)
    if args.feed_size != "native":
        try:
            w, h = map(int, args.feed_size.lower().split("x"))
            if not 1 <= w <= 1920 or not 1 <= h <= 1080:
                raise ValueError
            size = (w, h)
        except ValueError:
            p.error("--feed-size must be native or WxH (maximum 1920x1080)")

    count = dropped = bad = boxes = 0
    writer = capture = detections = image = None
    writer_shape = None
    started = time.monotonic()
    deadline = started + max(20, 5 + args.frames * args.timeout) if args.frames else math.inf
    done = False
    try:
        if args.detections_jsonl:
            detections = open(args.detections_jsonl, "w", encoding="utf-8")
        if args.feed:
            capture = open_feed(args)
        while not done and time.monotonic() < deadline:
            previous = None
            viewer = feeder = None
            try:
                viewer = socket.create_connection((args.host, args.port), timeout=args.timeout)
                if capture is not None:
                    feeder = socket.create_connection((args.host, args.in_port), timeout=args.timeout)
                while time.monotonic() < deadline:
                    if feeder is not None:
                        if not send_input(feeder, capture, size, args.feed_format):
                            done = True
                            break
                    frame = read_frame(viewer)
                    sequence = frame[0]
                    if previous is not None:
                        if sequence <= previous:
                            raise ValueError("non-monotonic frame sequence")
                        dropped += sequence - previous - 1
                    previous = sequence
                    count += 1
                    boxes += len(frame[2])
                    if count == 1:
                        started = time.monotonic()
                    fps = (count - 1) / max(.001, time.monotonic() - started)
                    if detections is not None:
                        dump_frame(frame, detections)
                    image = draw(frame, args.conf_thresh, fps, args.label_lang, args.font)
                    if args.record and writer is None:
                        writer_shape = (image.shape[1], image.shape[0])
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 30.0, writer_shape)
                        if not writer.isOpened():
                            raise RuntimeError("Could not open MP4 writer")
                    if writer is not None:
                        if (image.shape[1], image.shape[0]) != writer_shape:
                            raise RuntimeError("Recorder requires a consistent frame size")
                        writer.write(image)
                    if not args.headless:
                        cv2.imshow("Z15 C10 YOLO", image)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            done = True
                    if frame[10] & FAULT:
                        print(f"Board reported a hardware fault 0x{frame[9]:08x}", file=sys.stderr)
                        return 1
                    if args.frames is not None and count >= args.frames:
                        done = True
                    if done:
                        break
            except ValueError as exc:
                bad += 1
                print(f"Protocol error: {exc}", file=sys.stderr)
            except (OSError, EOFError) as exc:
                print(f"TCP reconnect: {exc}", file=sys.stderr)
            finally:
                if feeder is not None:
                    feeder.close()
                if viewer is not None:
                    viewer.close()
            if not done:
                time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        if detections is not None:
            detections.close()
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if args.snapshot and image is not None:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Cannot encode final snapshot")
        Path(args.snapshot).write_bytes(buf.tobytes())
    print(f"frames={count} dropped={dropped} boxes={boxes} protocol_errors={bad}")
    return int(bool(bad or dropped or args.frames is not None and count != args.frames))

if __name__ == "__main__":
    sys.exit(main())
