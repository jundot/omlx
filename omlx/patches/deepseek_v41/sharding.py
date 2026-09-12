# SPDX-License-Identifier: Apache-2.0
"""Size-bounded shards that keep each quantized projection together."""

from pathlib import Path

import mlx.core as mx

MAX_SHARD_BYTES = 5_000_000_000


class ShardWriter:
    def __init__(self, destination, max_shard_bytes=MAX_SHARD_BYTES):
        if max_shard_bytes <= 0:
            raise ValueError("Shard size must be positive")
        self.destination = Path(destination)
        self.max_shard_bytes = max_shard_bytes
        self._values = {}
        self._bytes = 0
        self._shards = []
        self._seen = set()

    def add(self, values):
        if self._seen.intersection(values):
            raise ValueError("Duplicate tensor in shard writer")
        size = sum(value.nbytes for value in values.values())
        if self._values and self._bytes + size > self.max_shard_bytes:
            self._flush()
        self._seen.update(values)
        self._values.update(values)
        self._bytes += size
        # A projection larger than the target stays intact in its own shard.
        if self._bytes >= self.max_shard_bytes:
            self._flush()

    def _flush(self):
        if not self._values:
            return
        name = f".model-shard-{len(self._shards) + 1:05d}.safetensors"
        mx.save_safetensors(
            str(self.destination / name),
            self._values,
            metadata={"format": "mlx"},
        )
        self._shards.append((name, tuple(self._values)))
        self._values = {}
        self._bytes = 0

    def add_file(self, filename, keys):
        """Adopt a streamed, indivisible tensor group without reading its data."""
        self._flush()
        if self._seen.intersection(keys):
            raise ValueError("Duplicate tensor in prewritten shard")
        self._seen.update(keys)
        self._shards.append((filename, tuple(keys)))

    def finish(self):
        self._flush()
        mapping = {}
        count = len(self._shards)
        for i, (temporary, keys) in enumerate(self._shards, 1):
            name = f"model-{i:05d}-of-{count:05d}.safetensors"
            target = self.destination / name
            if target.exists():
                raise FileExistsError(target)
            (self.destination / temporary).rename(target)
            mapping.update(dict.fromkeys(keys, name))
        return mapping
