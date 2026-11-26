"""Prepare Qwen Byte-Level BPE and Unicode NFC tables for the C99 runner."""

import argparse
from array import array
import json
from pathlib import Path
import struct
import unicodedata

from tools.c11_qwen_export import sha256


TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
WHITE_SPACE = (set(range(9, 14)) | {32, 133, 160, 5760, 8232, 8233, 8239, 8287, 12288}
               | set(range(8192, 8203)))


def bytes_to_chars():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {i: chr(i) for i in values}
    result.update((byte, chr(256 + i)) for i, byte in enumerate(
        byte for byte in range(256) if byte not in result))
    return result


def unicode_tables():
    dfd, flattened, compose, ccc, ranges = [], [], [], [], []
    previous = 0
    begin = 0
    for cp in range(0x110000):
        value = chr(cp)
        category = unicodedata.category(value)[0]
        flags = int(category == "L") | (int(category == "N") << 1) | (int(cp in WHITE_SPACE) << 2)
        if flags != previous:
            if previous:
                ranges.append((begin, cp - 1, previous))
            previous, begin = flags, cp
        combining = unicodedata.combining(value)
        if combining:
            ccc.append((cp, combining))
        nfd = unicodedata.normalize("NFD", value)
        if nfd != value:
            dfd.append((cp, len(flattened), len(nfd)))
            flattened.extend(map(ord, nfd))
        pair = unicodedata.decomposition(value).split()
        if len(pair) == 2 and not pair[0].startswith("<"):
            a, b = (int(x, 16) for x in pair)
            if unicodedata.normalize("NFC", chr(a) + chr(b)) == value:
                compose.append((a, b, cp))
    if previous:
        ranges.append((begin, 0x10ffff, previous))
    # Python's Unicode database supplies Hangul decompositions via an algorithm.
    for s in range(11172):
        cp = 0xAC00 + s
        trail = s % 28
        pair = ((cp - trail, 0x11A7 + trail) if trail else
                (0x1100 + s // 588, 0x1161 + (s % 588) // 28))
        compose.append((*pair, cp))
    return ranges, dfd, flattened, sorted(compose), ccc


def words(file, items):
    values = array("I", items)
    if values.itemsize != 4:
        raise ValueError("the BPE exporter needs uint32 arrays")
    if struct.pack("=I", 1)[0] != 1:
        values.byteswap()
    file.write(values.tobytes())


def convert(source, output):
    if sha256(source) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match Qwen's pinned tokenizer.json")
    doc = json.loads(source.read_text(encoding="utf-8"))
    model = doc["model"]
    if doc["normalizer"] != {"type": "NFC"} or model["type"] != "BPE":
        raise ValueError("Unrecognized Qwen tokenizer pipeline")
    chars = bytes_to_chars()
    invert = {char: byte for byte, char in chars.items()}
    base = model["vocab"]
    added = {row["content"]: row["id"] for row in doc["added_tokens"]}
    ids = {**base, **added}
    total = len(ids)
    if set(ids.values()) != set(range(total)):
        raise ValueError("Sparse or duplicate official tokenizer IDs")
    tokens = [b""] * total
    for name, idx in base.items():
        tokens[idx] = bytes(invert[char] for char in name)
    for name, idx in added.items():
        tokens[idx] = name.encode("utf-8")
    merges = []
    for rank, entry in enumerate(model["merges"]):
        a, b = entry.split(" ") if isinstance(entry, str) else entry
        merges.append((base[a], base[b], rank, base[a + b]))
    merges.sort()
    ranges, dfd, nfd, compose, ccc = unicode_tables()
    offsets = [0]
    for data in tokens:
        offsets.append(offsets[-1] + len(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        file.write(struct.pack("<8s10I", b"YSQBPE1\0", len(base), total, len(merges),
                               len(ranges), len(dfd), len(nfd), len(compose), len(ccc), offsets[-1], 0))
        words(file, offsets)
        for data in tokens:
            file.write(data)
        file.write(bytes(-file.tell() % 4))
        words(file, (base[chars[i]] for i in range(256)))
        for table in (merges, ranges, dfd):
            words(file, (word for row in table for word in row))
        words(file, nfd)
        for table in (compose, ccc):
            words(file, (word for row in table for word in row))
    print(f"C11_BPE_EXPORT_PASS symbols={total} merges={len(merges)} unicode={unicodedata.unidata_version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.tokenizer, args.out)


if __name__ == "__main__":
    main()
