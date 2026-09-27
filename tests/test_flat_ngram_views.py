import json
import struct
import tempfile
import unittest
from pathlib import Path

from omlx.quantization.ngram import flat_ngram_views


class FlatNgramTests(unittest.TestCase):
    def write(self, path, mutate=lambda h: None):
        h = {
            "__metadata__": {
                "format": "mlx-serve-ngram",
                "bits": "4",
                "group_size": "32",
            }
        }
        offset = 0
        for part, dtype, width, itemsize in [
            ("weight", "U32", 16, 4),
            ("scales", "BF16", 4, 2),
            ("biases", "BF16", 4, 2),
        ]:
            size = 5 * width * itemsize
            h[part] = {
                "dtype": dtype,
                "shape": [5, width],
                "data_offsets": [offset, offset + size],
            }
            offset += size
        mutate(h)
        raw = json.dumps(h).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))

    def check(self, path):
        return flat_ngram_views(
            path, "embedding", (3, 2), 128, {"bits": 4, "group_size": 32}
        )

    def test_uneven_shards_share_regions_without_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "table.bin"
            self.write(p)
            before = p.read_bytes()
            v = self.check(p)
            self.assertEqual(v["embedding.shard_1.weight"]["shape"], [2, 16])
            self.assertEqual(v["embedding.shard_1.weight"]["data_offsets"], [192, 320])
            self.assertEqual(v["embedding.shard_1.biases"]["data_offsets"], [384, 400])
            self.assertEqual(p.read_bytes(), before)

    def test_reject_bad_regions_geometry_and_format(self):
        mutations = [
            lambda h: h["weight"].update(data_offsets=[0, 999]),
            lambda h: h["scales"].update(data_offsets=[0, 40]),
            lambda h: h["weight"].update(shape=[6, 16]),
            lambda h: h["__metadata__"].update(format="unknown"),
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "table.bin"
            for mutation in mutations:
                self.write(p, mutation)
                with self.assertRaises(ValueError):
                    self.check(p)

    def test_truncated_and_bad_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "table.bin"
            for raw in (b"", struct.pack("<Q", 999999999)):
                p.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.check(p)

    def test_config_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "table.bin"
            self.write(p)
            with self.assertRaises(ValueError):
                flat_ngram_views(p, "x", (3, 2), 128, {"bits": 8, "group_size": 32})


if __name__ == "__main__":
    unittest.main()
