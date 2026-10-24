"""Regression tests for real OS descriptor exhaustion in PLE v3.

Fixtures use actual safetensors files and NumPy reads, not mmap test doubles.
The soft descriptor limit is lowered only in a child process.
"""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jangh2_bridge.checkpoint import Checkpoint
from jangh2_bridge.ple import IndexedMMapEmbedding


def many_shards(root, *, shards=512, files=1, split_tensors=False, bits=8, dtype='F16'):
    """512 real logical shards can share a single physical safetensors file."""
    root = Path(root)
    groups = {}
    weights = {}
    quant = {'bits': bits, 'group_size': 32, 'mode': 'affine'}
    dense = []
    for shard in range(shards):
        codes = (np.arange(64, dtype=np.uint32).reshape(2, 32) + shard * 7) % (1 << bits)
        packed = np.zeros((2, bits), np.uint32)
        for col in range(32):
            packed[:, col * bits // 32] |= codes[:, col] << ((col * bits) % 32)
        scales = np.full((2, 1), .5 + (shard % 3) * .25, np.float32)
        biases = np.full((2, 1), -2.0, np.float32)
        def aux(a):
            return ((a.view(np.uint32) >> 16).astype('<u2') if dtype == 'BF16'
                    else a.astype({'F16': '<f2', 'F32': '<f4'}[dtype]))
        arrays = {'weight': ('U32', packed), 'scales': (dtype, aux(scales)),
                  'biases': (dtype, aux(biases))}
        prefix = f'language_model.layers.1.ple.ngram_embedding.shards.{shard}'
        for ordinal, (name, array) in enumerate(arrays.items()):
            file_index = (3 * shard + ordinal if split_tensors else shard) % files
            filename = f'part-{file_index}.safetensors'
            key = prefix + '.' + name
            weights[key] = filename
            groups.setdefault(filename, []).append((key, *array))
        dense.append(codes.astype(np.float32) * scales + biases)
    for filename, tensors in groups.items():
        header, payload = {}, bytearray()
        for key, dt, array in tensors:
            start = len(payload)
            payload.extend(array.tobytes())
            header[key] = {'dtype': dt, 'shape': list(array.shape),
                           'data_offsets': [start, len(payload)]}
        encoded = json.dumps(header).encode()
        encoded += b' ' * (-len(encoded) % 8)
        (root / filename).write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    checkpoint = Checkpoint(root, {'quantization': quant}, weights, {}, 'many-shards')
    return checkpoint, np.concatenate(dense), shards * 2


def decode(reader, ids):
    p, s, b = reader.read_rows(ids)
    bits = reader.bits
    q = np.stack([(p[:, j * bits // 32] >> (j * bits % 32)) & ((1 << bits) - 1)
                  for j in range(32)], axis=1)
    return q * s + b


def fd_count():
    # Linux and macOS. Used only for observation, never for changing limits.
    directory = Path('/proc/self/fd') if Path('/proc/self/fd').is_dir() else Path('/dev/fd')
    return len(list(directory.iterdir()))


def stress_child(files):
    import resource
    with tempfile.TemporaryDirectory() as directory:
        ckpt, expected, rows = many_shards(directory, files=files)
        baseline = fd_count()
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = 64 if hard == resource.RLIM_INFINITY else min(64, hard)
        if baseline + 12 >= limit:
            raise RuntimeError(f'Test process already uses too many descriptors: {baseline}/{limit}')
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
        peak = baseline
        for cycle in range(3):
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 512)
            try:
                ids = np.concatenate([np.arange(rows)[::-1], [0, rows-1, 3, 3]])
                np.testing.assert_array_equal(decode(reader, ids), expected[ids])
                peak = max(peak, fd_count())
                stats = reader.mapping_stats()
                assert stats['peak_open_files'] <= 8, stats
                assert stats['open_files'] <= 8, stats
                assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == limit
            finally:
                reader.close()
            assert fd_count() <= baseline, (fd_count(), baseline)
        print(json.dumps({'status': 'PASS', 'logical_shards': 512, 'tensors': 1536,
                          'physical_files': files, 'soft_limit': limit,
                          'baseline_fds': baseline, 'observed_fds_after_lookup': peak,
                          'max_cached_mappings': stats['peak_open_files'],
                          'final_fds': fd_count(), 'load_read_close_cycles': 3}))


class FileLimitTests(unittest.TestCase):
    def run_child(self, files):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'tests')))
        env['OPENBLAS_NUM_THREADS'] = '1'
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stress', str(files)],
                                text=True, capture_output=True, env=env, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        result = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(result['status'], 'PASS')
        self.assertLessEqual(result['soft_limit'], 64)
        return result

    def test_1536_tensors_share_one_file_under_64_fd_limit(self):
        result = self.run_child(1)
        self.assertEqual(result['max_cached_mappings'], 1)

    def test_80_physical_files_under_64_fd_limit_with_no_leaks(self):
        result = self.run_child(80)
        self.assertLessEqual(result['max_cached_mappings'], 8)

    def test_constructor_is_metadata_only_then_one_map_per_file(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, expected, rows = many_shards(d, shards=16)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 16)
            try:
                self.assertEqual(reader.mapping_stats()['open_files'], 0)
                np.testing.assert_array_equal(decode(reader, [0, 31, 2, 31]), expected[[0, 31, 2, 31]])
                stats = reader.mapping_stats()
                self.assertEqual(stats['file_opens'], 1)
                self.assertEqual(stats['open_files'], 1)
                self.assertEqual(stats['tensor_count'], 48)
                self.assertEqual(stats['unique_files'], 1)
            finally:
                reader.close()

    def test_eviction_reopens_correct_file_and_detaches_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, expected, rows = many_shards(d, shards=18, files=18)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 18)
            p, s, b = reader.read_rows([0, 1])
            snapshots = [x.copy() for x in (p, s, b)]
            np.testing.assert_array_equal(decode(reader, np.arange(rows)), expected)
            np.testing.assert_array_equal(decode(reader, [0, rows-1, 0]), expected[[0, rows-1, 0]])
            stats = reader.mapping_stats()
            self.assertGreater(stats['evictions'], 0)
            self.assertLessEqual(stats['peak_open_files'], 8)
            reader.close()
            for array, snapshot in zip((p, s, b), snapshots):
                self.assertTrue(array.flags.owndata)
                np.testing.assert_array_equal(array, snapshot)
            self.assertEqual(reader.mapping_stats()['open_files'], 0)
            reader.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                reader.read_rows([0])

    def test_split_tensors_and_all_supported_dtypes_after_eviction(self):
        for bits in (4, 8):
            for dt in ('F16', 'BF16', 'F32'):
                with self.subTest(bits=bits, dtype=dt), tempfile.TemporaryDirectory() as d:
                    ckpt, expected, rows = many_shards(d, shards=10, files=30,
                                                      split_tensors=True, bits=bits, dtype=dt)
                    reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 10)
                    try:
                        ids = np.array([19, 0, 6, 1, 6, 17, 4, 19, 2, 13, 9])
                        np.testing.assert_array_equal(decode(reader, ids), expected[ids])
                        self.assertGreater(reader.mapping_stats()['evictions'], 0)
                    finally:
                        reader.close()

    def test_no_persistent_descriptors_after_close_or_destruction(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, expected, rows = many_shards(d, shards=12, files=12)
            gc.collect()
            baseline = fd_count()
            for _ in range(5):
                reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 12)
                np.testing.assert_array_equal(decode(reader, np.arange(rows)), expected)
                reader.close()
                self.assertLessEqual(fd_count(), baseline)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 12)
            reader.read_rows([0, 10, 20])
            del reader
            gc.collect()
            self.assertLessEqual(fd_count(), baseline)

    def test_concurrent_reads_preserve_values(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, expected, rows = many_shards(d, shards=12, files=12)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 12)
            def task(i):
                ids = np.array([(i * 7) % rows, (i * 13) % rows, 0, rows-1])
                np.testing.assert_array_equal(decode(reader, ids), expected[ids])
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(task, range(24)))
                self.assertLessEqual(reader.mapping_stats()['peak_open_files'], 8)
            finally:
                reader.close()

    def test_backing_files_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, expected, rows = many_shards(d, shards=12, files=12)
            def hashes():
                return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(d).iterdir()}
            before = hashes()
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 12)
            arrays = reader.read_rows(np.arange(rows))
            for a in arrays:
                a[:] = 0
            reader.close()
            self.assertEqual(before, hashes())

    def test_malformed_header_fails_before_any_payload_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, rows = many_shards(d, shards=3, files=3)
            (Path(d) / 'part-2.safetensors').write_bytes(b'invalid')
            baseline = fd_count()
            with self.assertRaises(ValueError):
                IndexedMMapEmbedding(ckpt, 1, rows, 32, 3)
            self.assertLessEqual(fd_count(), baseline)

    def test_mmap_objects_are_readonly_and_closed(self):
        import mmap
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, rows = many_shards(d, shards=2)
            real_mmap = mmap.mmap
            opened = []
            def record(*args, **kwargs):
                mm = real_mmap(*args, **kwargs)
                opened.append(mm)
                return mm
            with patch('jangh2_bridge.ple.mmap.mmap', side_effect=record):
                reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 2)
                reader.read_rows([0, 3])
                self.assertEqual(len(opened), 1)
                with self.assertRaises(TypeError):
                    opened[0][0] = 0
                reader.close()
            self.assertTrue(all(mm.closed for mm in opened))

    def test_failed_mmap_open_closes_temporary_fd_and_previous_maps_on_close(self):
        import mmap
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, rows = many_shards(d, shards=2, files=2)
            baseline = fd_count()
            real_mmap = mmap.mmap
            count = 0
            opened = []
            def fail_second(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError(24, 'injected mapping failure')
                mm = real_mmap(*args, **kwargs)
                opened.append(mm)
                return mm
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 2)
            try:
                with patch('jangh2_bridge.ple.mmap.mmap', side_effect=fail_second):
                    with self.assertRaisesRegex(OSError, 'injected'):
                        reader.read_rows([0, 3])
                self.assertEqual(reader.mapping_stats()['open_files'], 1)
            finally:
                reader.close()
            self.assertTrue(all(mm.closed for mm in opened))
            self.assertLessEqual(fd_count(), baseline)

    def test_truncated_file_rejected_before_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, rows = many_shards(d, shards=2)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 2)
            file = Path(d) / 'part-0.safetensors'
            file.write_bytes(file.read_bytes()[:-1])
            try:
                with self.assertRaisesRegex(ValueError, 'size changed'):
                    reader.read_rows([0])
                self.assertEqual(reader.mapping_stats()['open_files'], 0)
            finally:
                reader.close()

    def test_empty_lookup_opens_no_files(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, rows = many_shards(d, shards=2)
            reader = IndexedMMapEmbedding(ckpt, 1, rows, 32, 2)
            try:
                self.assertEqual(reader.read_rows(np.array([], np.int64))[0].shape, (0, 8))
                self.assertEqual(reader.mapping_stats()['file_opens'], 0)
            finally:
                reader.close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--stress':
        stress_child(int(sys.argv[2]))
    else:
        unittest.main()
