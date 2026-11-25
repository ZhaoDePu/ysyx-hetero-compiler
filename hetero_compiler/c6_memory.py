"""Two-level DDR4 and native Z15 scratch allocation for C6 graphs."""

from dataclasses import dataclass

from .yhb import POOL_BYTES

ALIGNMENT = 32
HALF_BYTES = 1 << 20


def align_dma(nbytes):
    return (nbytes + ALIGNMENT - 1) & -ALIGNMENT


@dataclass(frozen=True)
class Lifetime:
    """Storage is live from first through last, including both stages."""

    name: str
    size: int
    first: int
    last: int


@dataclass(frozen=True)
class Allocation:
    offset: int
    size: int

    @property
    def capacity(self):
        return align_dma(self.size)


def reserve_ddr(lifetimes, limit=POOL_BYTES, *, base=0):
    """Allocate statically, reusing a range after its last consumer drains."""
    if (type(limit) is not int or type(base) is not int or not 0 <= base <= limit <= POOL_BYTES
            or base % ALIGNMENT):
        raise ValueError("invalid DDR capacity")
    slots = {}
    occupied = []
    high_water = base
    for life in sorted(lifetimes, key=lambda life: (life.first, -life.size, life.name)):
        if (not life.name or life.name in slots or type(life.size) is not int
                or life.size <= 0 or not 0 <= life.first <= life.last):
            raise ValueError(f"invalid or duplicate DDR lifetime: {life}")
        occupied = [block for block in occupied if block[2] >= life.first]
        capacity = align_dma(life.size)
        offset = base
        for lo, hi, _ in occupied:
            if offset + capacity <= lo:
                break
            offset = hi
        end = offset + capacity
        if end > limit:
            raise ValueError(f"{life.name}: DDR allocation needs {end} bytes, capacity is {limit}")
        slots[life.name] = Allocation(offset, life.size)
        occupied.append((offset, end, life.last))
        occupied.sort()
        high_water = max(high_water, end)
    return slots, high_water


class Scratch:
    """Bump allocator for independent tile working sets in one ownership half."""

    def __init__(self, zone):
        if zone not in (0, 1):
            raise ValueError("invalid Z15 scratch ownership half")
        self.base = zone * HALF_BYTES
        self.used = 0

    def reserve(self, nbytes):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Z15 scratch allocation must have a positive byte count")
        offset = self.base + self.used
        if self.used + align_dma(nbytes) > HALF_BYTES:
            raise ValueError("tile exceeds its Z15 1 MiB scratch ownership half")
        self.used += align_dma(nbytes)
        return Allocation(offset, nbytes)
