# SPDX-License-Identifier: Apache-2.0
"""On a Mac, large frames reach MLX through a Metal buffer over the mailbox itself, with no CPU copy."""

from __future__ import annotations

import ctypes
import mmap
import multiprocessing
import sys

import mlx.core as mx
import numpy as np
import pytest
from rdma_loopback import LoopbackLink, PythonWordOps

from omlx.cluster.rdma import metal_view
from omlx.cluster.rdma.mailbox import ClientMailbox, ServiceMailbox
from omlx.cluster.rdma.metal_view import MetalWindow
from omlx.cluster.rdma.stage_plan import StageLink
from omlx.cluster.rdma.stage_transport import StageReceiver, StageSender
from omlx.cluster.rdma.words import load_word_ops

_OPS, _ = load_word_ops()
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or _OPS is None
    or not hasattr(_OPS.library, "mcdma_rpc_metal_wrap"),
    reason="needs a macOS libmcdma-rpc with Metal buffers",
)


@pytest.fixture
def region():
    memory = mmap.mmap(-1, 4 * 16384)
    yield memory, ctypes.addressof(ctypes.c_char.from_buffer(memory))


def test_a_window_view_is_the_mailbox_memory_itself(region):
    memory, base = region
    window = MetalWindow.open(mx, _OPS.library, base, len(memory))
    try:
        raw = np.frombuffer(memory, dtype=np.uint8)
        raw[4224:4228] = [9, 8, 7, 6]
        view = window.view(mx, 4224, 64)
        assert int(np.asarray(view).ctypes.data) == base + 4224
        assert np.asarray(view)[:4].tolist() == [9, 8, 7, 6]
        copied = window.copy(mx, 4224, 64)
        raw[4224] = 1
        # The copy owns its bytes, so the mailbox may take the next frame.
        assert int(np.asarray(copied)[0]) == 9
        with pytest.raises(ValueError):
            window.view(mx, len(memory) - 8, 16)
    finally:
        window.close()


def test_a_helper_without_metal_calls_gives_no_window(region):
    _, base = region
    assert MetalWindow.open(mx, object(), base, 16384) is None


def _sender_main(name, socket_path, mailbox_path):
    service = ServiceMailbox.attach(
        name, socket_path, PythonWordOps(), mailbox_path=mailbox_path
    )
    try:
        sender = StageSender(
            mx, service, StageLink(1, 0, name, socket_path), timeout_s=20
        )
        mx.random.seed(11)
        sender.send((mx.random.normal((4, 128, 2048)) * 2).astype(mx.bfloat16))
    finally:
        service.close()


def test_a_large_stage_message_arrives_exact_through_the_window(monkeypatch):
    monkeypatch.setattr(metal_view, "MIN_BYTES", 1 << 20)
    copies = []
    original = MetalWindow.copy
    monkeypatch.setattr(
        MetalWindow,
        "copy",
        lambda self, *args: copies.append(args[2]) or original(self, *args),
    )
    link = LoopbackLink(request_bytes=1 << 20, reply_bytes=4 << 20)
    process = multiprocessing.get_context("spawn").Process(
        target=_sender_main,
        args=(link.name, link.socket_path, link.mailbox_path),
        daemon=True,
    )
    process.start()
    link.wait_service()
    client = ClientMailbox.attach(link.name, _OPS)
    receiver = StageReceiver(
        mx, client, StageLink(1, 0, link.name, link.socket_path), timeout_s=60
    )
    try:
        assert receiver._window is not None
        got = receiver.recv_like(mx.zeros((4, 128, 2048), dtype=mx.bfloat16))
        mx.random.seed(11)
        expected = (mx.random.normal((4, 128, 2048)) * 2).astype(mx.bfloat16)
        assert mx.array_equal(got, expected).item()
        # A 2 MiB message fits one 4 MiB reply half, so one frame took the window.
        assert copies == [4 * 128 * 2048 * 2]
    finally:
        process.join(timeout=30)
        client.close()
        link.close()
    assert process.exitcode == 0
