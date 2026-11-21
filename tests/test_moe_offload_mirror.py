# SPDX-License-Identifier: Apache-2.0
"""Tests for expert reads spread over checkpoint copies (moe_offload_mirror)."""

import itertools
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from omlx.model_settings import validate_moe_expert_offload
from omlx.patches.moe_offload_mirror import (
    MirrorRouter,
    ShardCopies,
    require_identical_mirrors,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm.models.switch_layers import SwitchGLU

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class _Clock:
    def __init__(self):
        self.time = 1

    def __call__(self):
        return self.time


class TestRouter:
    """Deterministic routing: no disk timing and no sleeps."""

    def test_measured_copies_share_reads_by_expected_finish(self):
        clock = _Clock()
        router = MirrorRouter(2, now=clock)
        assert router.claim(100) == 0  # the first probe uses the model directory
        clock.time += 100
        router.release(0, 100, completed=True)
        assert router.claim(100) == 1  # an unmeasured mirror gets a probe
        clock.time += 200
        router.release(1, 100, completed=True)
        assert router.claim(1000) == 0  # idle reads prefer the faster copy
        assert router.claim(100) == 1  # a backlog spills to the slower copy
        clock.time += 200
        router.release(1, 100, completed=True)
        clock.time += 800
        router.release(0, 1000, completed=True)
        assert router.served_bytes() == [1100, 200]

        # Keep the fast copy busy while the slower copy's reading expires.
        assert router.claim(10) == 0
        clock.time += 1_000_000_001
        assert router.claim(10) == 1  # a stale idle copy is probed again
        clock.time += 100
        router.release(1, 10, completed=False)
        router.release(0, 10, completed=True)
        assert router.served_bytes() == [1110, 200]  # the failed probe adds none

    def test_failed_read_releases_its_reservation(self):
        clock = _Clock()
        router = MirrorRouter(1, now=clock)
        assert router.claim(100) == 0
        clock.time += 10
        router.release(0, 100, completed=False)
        assert router.served_bytes() == [0]
        assert router.claim(20) == 0
        clock.time += 10
        router.release(0, 20, completed=True)
        assert router.served_bytes() == [20]

    @pytest.mark.parametrize("copies", [2, 3])
    def test_burst_before_any_completion_spreads_over_the_copies(self, copies):
        """Sequential claim/release pairs never exercise a cold start, or the
        reset after an idle period, with several reads already pending."""
        clock = _Clock()
        router = MirrorRouter(copies, now=clock)
        for _phase in ("cold", "post-idle"):
            choices = [router.claim(100) for _ in range(12)]
            assert choices == [i % copies for i in range(12)]
            clock.time += 100
            for replica in choices:
                router.release(replica, 100, completed=True)
            clock.time += 1_000_000_001
        assert router.served_bytes() == [2400 // copies] * copies

    def test_unmeasured_copies_are_balanced_by_pending_bytes(self):
        clock = _Clock()
        router = MirrorRouter(2, now=clock)
        large = router.claim(1000)
        small = router.claim(100)
        assert (large, small) == (0, 1)
        pending = router.claim(100)
        assert pending == 1  # both unknown: the copy with fewer pending bytes
        clock.time += 100
        router.release(small, 100, completed=True)
        measured = router.claim(100)
        assert measured == 1  # a pending probe does not beat a measured copy
        router.release(large, 1000, completed=False)
        retry = router.claim(50)
        assert retry == 0  # the failed probe freed the copy for another try
        clock.time += 100
        router.release(retry, 50, completed=True)
        router.release(pending, 100, completed=True)
        router.release(measured, 100, completed=True)
        assert router.served_bytes() == [50, 300]

    def test_concurrent_claims_keep_the_byte_accounting(self):
        router = MirrorRouter(3)

        def one_read(i):
            replica = router.claim(16)
            router.release(replica, 16, completed=i % 4 != 0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one_read, range(1000)))
        served = router.served_bytes()
        assert len(served) == 3 and sum(served) == 12000


def _write_shard(directory, payload=b"\x01" * 64, header=b'{"t":{}}', name="a"):
    directory.mkdir(exist_ok=True)
    shard = directory / f"{name}.safetensors"
    shard.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return shard


class TestMirrorValidation:
    def test_a_copy_with_other_tensor_bytes_passes(self, tmp_path):
        """Only sizes and headers are compared; see the function's docstring."""
        _write_shard(tmp_path / "model")
        _write_shard(tmp_path / "mirror", payload=b"\x02" * 64)
        require_identical_mirrors(tmp_path / "model", [str(tmp_path / "mirror")])

    @pytest.mark.parametrize(
        "damage, message",
        [
            (lambda d: (d / "a.safetensors").unlink(), "cannot serve a.safetensors"),
            (lambda d: _write_shard(d, payload=b"\x01" * 63), "holds a different"),
            (lambda d: _write_shard(d, header=b'{"u":{}}'), "holds a different"),
            (lambda d: (d / "a.safetensors").write_bytes(b"x"), "not a complete"),
            (
                lambda d: (d / "a.safetensors").write_bytes(b"\xff" * 9),
                "not a complete",
            ),
        ],
    )
    def test_a_mirror_that_differs_is_refused(self, tmp_path, damage, message):
        _write_shard(tmp_path / "model")
        _write_shard(tmp_path / "mirror")
        damage(tmp_path / "mirror")
        with pytest.raises(ValueError, match=message):
            require_identical_mirrors(tmp_path / "model", [tmp_path / "mirror"])

    def test_every_shard_of_the_model_is_required(self, tmp_path):
        _write_shard(tmp_path / "model")
        _write_shard(tmp_path / "model", name="b")
        _write_shard(tmp_path / "mirror")
        with pytest.raises(ValueError, match="cannot serve b.safetensors"):
            require_identical_mirrors(tmp_path / "model", [tmp_path / "mirror"])

    def test_the_same_directory_twice_is_refused(self, tmp_path):
        model, mirror = tmp_path / "model", tmp_path / "mirror"
        _write_shard(model)
        _write_shard(mirror)
        (tmp_path / "link").symlink_to(mirror)
        for mirrors in ([model], [mirror, tmp_path / "link"]):
            with pytest.raises(ValueError, match="listed twice or is the model"):
                require_identical_mirrors(model, mirrors)
        with pytest.raises(ValueError, match="is not a directory"):
            require_identical_mirrors(model, [tmp_path / "missing"])


class TestSettings:
    def test_mirrors_must_be_a_list_of_paths(self):
        for bad in ("/disk2/model", [""], [1], ["/disk2/model", None]):
            with pytest.raises(ValueError, match="list of directory paths"):
                validate_moe_expert_offload({"moe_expert_offload_mirrors": bad})
        validate_moe_expert_offload({"moe_expert_offload_mirrors": ["/disk2/model"]})

    def test_deepseek_v41_has_its_own_expert_storage(self):
        settings = {
            "moe_expert_offload_enabled": True,
            "moe_expert_offload_mirrors": ["/disk2/model"],
        }
        validate_moe_expert_offload(settings, model_type="qwen4_exp")
        with pytest.raises(ValueError, match="not supported for deepseek_v41"):
            validate_moe_expert_offload(settings, model_type="deepseek_v41")


def _copies_of(tmp_path, payloads):
    """Open one shard per payload; return the plan's copies and a ticking router."""
    router = MirrorRouter(len(payloads), now=itertools.count(1, 100).__next__)
    fds = [
        os.open(_write_shard(tmp_path / f"copy{i}", payload=p), os.O_RDONLY)
        for i, p in enumerate(payloads)
    ]
    copies = ShardCopies(router, fds[0])
    copies.fds.extend(fds[1:])
    return copies


@needs_mlx
class TestRoutedReads:
    def test_the_bytes_path_reads_the_copy_the_router_picks(self, tmp_path):
        from omlx.patches.moe_expert_offload import CheckpointExpertStore, _ReadPlan

        copies = _copies_of(tmp_path, [b"\x01" * 64, b"\x02" * 64])
        plan = _ReadPlan(copies.fds[0], 16, 8, np.uint8, None, (8,), copies)
        assert CheckpointExpertStore.read(plan) == b"\x01" * 8
        assert CheckpointExpertStore.read(plan) == b"\x02" * 8  # the mirror's probe
        assert copies.router.served_bytes() == [8, 8]

        with pytest.raises(OSError, match="short read"):
            CheckpointExpertStore.read(plan._replace(offset=1000))
        assert sum(copies.router.served_bytes()) == 16  # a failed read adds none

    def test_a_read_the_staging_pool_declines_is_not_measured(self, tmp_path):
        from omlx.custom_kernels.moe_staging import StagedRead, Staging
        from omlx.patches.moe_expert_offload import CheckpointExpertStore, _ReadPlan

        class Pool:
            def __init__(self, lease):
                self.lease, self.fds = lease, []

            def read(self, fd, offset, nbytes):
                self.fds.append(fd)
                return self.lease

        copies = _copies_of(tmp_path, [b"\x01" * 64, b"\x02" * 64])
        plan = _ReadPlan(copies.fds[0], 16, 8, np.uint8, None, (8,), copies)
        staging = Staging([8], CheckpointExpertStore.read, CheckpointExpertStore.to_mx)

        staging.pool = Pool(lease=object())
        assert isinstance(staging.read(plan), StagedRead)
        assert staging.pool.fds == [copies.fds[0]]
        assert copies.router.served_bytes() == [8, 0]

        staging.pool = Pool(lease=None)  # every buffer is leased out
        assert staging.read(plan) == b"\x02" * 8  # the bytes path, routed again
        assert staging.pool.fds == [copies.fds[1]]
        assert copies.router.served_bytes() == [8, 8]

    def test_the_store_opens_each_shard_in_every_copy(self, tmp_path):
        from omlx.patches.moe_expert_offload import CheckpointExpertStore, _pread_exact

        for i in range(3):
            (tmp_path / f"copy{i}").mkdir()
            mx.save_safetensors(
                str(tmp_path / f"copy{i}" / "model.safetensors"),
                {"experts.weight": mx.full((4, 16), i, dtype=mx.uint32)},
            )
        dirs = [tmp_path / f"copy{i}" for i in range(3)]
        plain = CheckpointExpertStore(dirs[0])
        assert plain.router is None
        assert plain.plan_expert("experts.weight", 2).copies is None

        store = CheckpointExpertStore(dirs[0], [str(d) for d in dirs[1:]])
        plan = store.plan_expert("experts.weight", 2)
        assert plan.copies.fds[0] == plan.fd
        for i, fd in enumerate(plan.copies.fds):
            slab = np.frombuffer(_pread_exact(fd, plan.offset, plan.nbytes), np.uint32)
            assert slab.tolist() == [i] * 16
        for _ in range(6):
            store.read(plan)
        assert sum(store.router.served_bytes()) == 6 * plan.nbytes

        (dirs[2] / "model.safetensors").write_bytes(b"truncated")
        with pytest.raises(ValueError, match="cannot serve model.safetensors"):
            CheckpointExpertStore(dirs[0], dirs[1:])


E, D, INTER, GROUP = 32, 64, 32, 32


def _offloaded(tmp_path, mirrors):
    """A two-layer quantized MoE whose checkpoint is copied to ``mirrors``."""
    from omlx.patches.moe_expert_offload import apply_moe_expert_offload

    class _Experts(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.switch_glu = glu

    class _Layer(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.experts = _Experts(glu)

    class _MiniMoE(nn.Module):
        def __init__(self, glus):
            super().__init__()
            self.layers = [_Layer(g) for g in glus]

    glus, tensors = [], {}
    for i in range(2):
        mx.random.seed(i)
        glu = SwitchGLU(D, INTER, E)
        nn.quantize(glu, group_size=GROUP, bits=4)
        glus.append(glu)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for field in ("weight", "scales", "biases"):
                if getattr(glu, proj).get(field) is not None:
                    name = f"layers.{i}.experts.switch_glu.{proj}.{field}"
                    tensors[name] = getattr(glu, proj)[field]
    primary = tmp_path / "model"
    primary.mkdir(parents=True)
    mx.save_safetensors(str(primary / "model.safetensors"), tensors)
    for mirror in mirrors:
        shutil.copytree(primary, mirror)
    model = _MiniMoE(glus)
    assert apply_moe_expert_offload(model, primary, 0.25, mirrors=mirrors) == 2
    return model


@needs_mlx
class TestOffloadWithMirrors:
    def _run(self, model):
        mx.random.seed(7)
        outs = []
        for step in range(24):
            x = mx.random.normal((1, 1, D))
            idx = mx.array([[[step % E, (step * 7 + 3) % E]]], dtype=mx.int32)
            for layer in model.layers:
                x = layer.experts.switch_glu(x, idx)[:, :, 0]
            outs.append(np.array(x))
        return outs

    def test_mirrors_change_where_bytes_come_from_not_the_output(self, tmp_path):
        from omlx.patches.moe_expert_offload import moe_offload_stats

        plain = _offloaded(tmp_path / "plain", [])
        reference = self._run(plain)
        assert "copy_bytes" not in moe_offload_stats(plain)

        mirrors = [tmp_path / "disk2", tmp_path / "disk3"]
        mirrored = _offloaded(tmp_path / "mirrored", mirrors)
        outs = self._run(mirrored)
        assert all(np.array_equal(a, b) for a, b in zip(outs, reference))

        stats = moe_offload_stats(mirrored)
        assert len(stats["copy_bytes"]) == 3
        assert sum(stats["copy_bytes"]) == stats["fetched_bytes"] > 0
        assert all(served > 0 for served in stats["copy_bytes"])

    def test_a_mirror_of_another_checkpoint_stops_the_load(self, tmp_path):
        from omlx.patches.moe_expert_offload import apply_moe_expert_offload

        model = _offloaded(tmp_path / "ok", [])
        other = tmp_path / "other"
        _write_shard(other, name="model")
        with pytest.raises(ValueError, match="holds a different model.safetensors"):
            apply_moe_expert_offload(model, tmp_path / "ok" / "model", mirrors=[other])
