"""C6 vector and layout planning tests."""
import unittest

from hetero_compiler.c6_memory import HALF_BYTES
from hetero_compiler.c6_vector import vector_tiles


def plan(op, sources, results, **kwargs):
    inputs = [i << 22 for i in range(len(sources))]
    outputs = [(i + len(inputs)) << 22 for i in range(len(results))]
    return vector_tiles("test_" + op, op, sources, results, inputs, outputs, **kwargs)


class VectorTiles(unittest.TestCase):
    def test_three_way_concat_preserves_order_and_independent_halves(self):
        tiles = plan("Concat", [(1, 17, 8, n) for n in (8, 12, 16)], [(1, 17, 8, 36)],
                     axis=-1, tile_rows=5)
        self.assertEqual([t.zone for t in tiles], [0, 1, 0, 1])
        self.assertTrue(all(len(t.stages) == 2 and len(t.loads) == 3 for t in tiles))
        self.assertEqual(sum(t.stores[0].size for t in tiles), 17 * 8 * 36)

    def test_pool_tile_has_halo_but_stores_only_its_core_rows(self):
        tiles = plan("MaxPool", [(1, 8, 8, 16)], [(1, 8, 8, 16)], tile_rows=2)
        self.assertEqual([t.loads[0].size for t in tiles], [512, 768, 768, 512])
        core = tiles[1].stores[0]
        self.assertEqual(core.size, 256)
        self.assertEqual(core.scratch_offset - dict(tiles[1].stages[0].writes)[0xA8], 256)

    def test_add_segments_never_cross_half(self):
        tiles = plan("Add", [(1_048_608,), (1_048_608,)], [(1_048_608,)])
        self.assertGreater(len(tiles), 1)
        for tile in tiles:
            self.assertLessEqual(tile.scratch_bytes, HALF_BYTES)
            for move in (*tile.loads, *tile.stores):
                self.assertEqual(move.ddr_offset % 32, 0)
                self.assertEqual(move.scratch_offset // HALF_BYTES, tile.zone)
                self.assertEqual((move.scratch_offset + move.size - 1) // HALF_BYTES, tile.zone)

    def test_non_unit_resize_and_unaligned_requested_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "nearest"):
            plan("Resize", [(1, 2, 16, 4)], [(1, 6, 48, 4)])
        with self.assertRaisesRegex(ValueError, "aligned"):
            plan("Concat", [(1, 8, 3, 5), (1, 8, 3, 7)], [(1, 8, 3, 12)],
                 axis=3, tile_rows=1)

    def test_lookup_table_loaded_once_across_tiles(self):
        lut = tuple(range(128)) + tuple(range(-128, 0))
        case = [(1, 4, 8, 16)]
        tiles = plan("RequantLUT", case, case, table=lut, tile_rows=1)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(sum(reg == 0xCC for t in tiles for s in t.stages
                             for reg, _ in s.writes), 256)


if __name__ == "__main__":
    unittest.main()
