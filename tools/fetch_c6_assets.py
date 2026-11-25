"""Fetch the exact Phase C6 public artifacts on a host with HTTPS egress."""

from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

ASSETS = {
    "yolo26n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        8 << 20,
    ),
    "bus.jpg": (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
        5 << 20,
    ),
    "zidane.jpg": (
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
        "16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb",
        5 << 20,
    ),
    "coco128.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e",
        16 << 20,
    ),
}


def main():
    found = {}
    for name, (url, expected, limit) in ASSETS.items():
        path = Path(name)
        if path.is_file() and path.stat().st_size <= limit:
            data = path.read_bytes()
        else:
            request = Request(url, headers={"User-Agent": "ysyx-phase-c6"})
            with urlopen(request, timeout=30) as response:
                data = response.read(limit + 1)
        digest = sha256(data).hexdigest()
        if digest != expected or len(data) > limit:
            raise ValueError(f"{name}: expected SHA-256 {expected}, got {digest}")
        path.write_bytes(data)
        found[name] = {"source": url, "bytes": len(data), "sha256": digest}
    Path("downloads.json").write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
