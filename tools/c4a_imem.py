"""Optional replay-only cache for position-independent frozen Z15 GPU programs."""

from collections import Counter


_CAPACITY = 0x4000
_RELATIVE_OPS = {0x03, 0x0F, 0x13, 0x23, 0x33, 0x37, 0x63, 0x6F}
_GID = 0xCC3022F3
_EBREAK = 0x00100073


def _transfer_delta(word):
    opcode = word & 127
    if opcode == 0x63:
        bits = 13
        imm = ((word >> 31) << 12) | (((word >> 7) & 1) << 11)
        imm |= (((word >> 25) & 63) << 5) | (((word >> 8) & 15) << 1)
    elif opcode == 0x6F:
        bits = 21
        imm = ((word >> 31) << 20) | (((word >> 12) & 255) << 12)
        imm |= (((word >> 20) & 1) << 11) | (((word >> 21) & 1023) << 1)
    else:
        return None
    sign = 1 << (bits - 1)
    return (imm ^ sign) - sign


def _can_relocate(words):
    """Require the compiler builder's whitelist and no branch out of the image."""
    if not words or words[-1] != _EBREAK:
        return False
    for i, word in enumerate(words):
        opcode = word & 127
        if word == _GID or (word == _EBREAK and i == len(words) - 1):
            continue
        if opcode not in _RELATIVE_OPS or (opcode == 0x6F and (word >> 7) & 31):
            return False
        delta = _transfer_delta(word)
        if delta is not None and (delta % 4 or not 0 <= i + delta // 4 < len(words)):
            return False
    return True


def _upload(words, base):
    return [f"write {0x4000 + base + 4*i} {word}" for i, word in enumerate(words)]


class ReplayPrograms:
    def __init__(self, images):
        """Select resident images, leaving a base-zero replacement slot if needed."""
        self.images = {key: tuple(words) for key, words in images.items()}
        if any(not 0 < 4 * len(words) <= _CAPACITY for words in self.images.values()):
            raise ValueError("GPU image size is outside the IMEM aperture")
        frequencies = Counter(self.images.values())
        ordered = sorted(frequencies, key=lambda w: (-len(w) * frequencies[w], w))
        total = 4 * sum(len(words) for words in ordered)
        fully_resident = total <= _CAPACITY and all(map(_can_relocate, ordered))
        replace_size = 0 if fully_resident else max(map(len, ordered)) * 4
        base = replace_size
        self.address = {}
        self.preload = []
        self.replacement = None
        self.reloads = 0
        for words in ordered:
            size = 4 * len(words)
            if base + size <= _CAPACITY and _can_relocate(words):
                self.address[words] = base
                self.preload.extend(_upload(words, base))
                base += size

    def commands(self, key, jobs):
        words = self.images[key]
        lines = []
        if words in self.address:
            base = self.address[words]
        else:
            base = 0
            if words != self.replacement:
                lines = _upload(words, base)
                self.replacement = words
                self.reloads += 1
        for job in jobs:
            entry = job["entry"]
            if entry % 4 or not 0 <= entry < 4 * len(words):
                raise ValueError("GPU entry is outside its cached image")
            lines.append(f"submit_gpu {base+entry} {job['gpu_job_reg']}")
        return lines
