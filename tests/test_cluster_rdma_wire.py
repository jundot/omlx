# SPDX-License-Identifier: Apache-2.0
"""Mailbox words, sizes and names, and the stage frame header, match mcdma-rpcd protocol 1."""

from __future__ import annotations

import pytest

from omlx.cluster.rdma import frames, layout


def test_words_pack_sequence_high_and_length_low():
    word = layout.pack_word(7, 4096)
    assert word == (7 << 32) | 4096
    assert (layout.word_seq(word), layout.word_length(word)) == (7, 4096)


def test_sequences_wrap_past_zero_which_marks_an_empty_word():
    assert layout.next_seq(1) == 2
    assert layout.next_seq(0xFFFFFFFF) == 1


@pytest.mark.parametrize("name", ["link1", "spark_a-2", "a" * 20])
def test_valid_link_names_fit_both_mailbox_schemes(name):
    assert layout.valid_link_name(name) == name
    assert len("/" + layout.client_shm_name(name)) <= 31


@pytest.mark.parametrize("name", ["", "a" * 21, "has space", "../etc", None, 3])
def test_invalid_link_names_are_refused(name):
    with pytest.raises(ValueError):
        layout.valid_link_name(name)


def test_listen_side_paths_follow_the_daemon_convention():
    assert layout.service_mailbox_path("link1") == "/dev/shm/mcdma-rpc.link1"
    assert layout.service_socket_path("link1") == "/tmp/mcdma-rpcd.link1.sock"


def test_sizes_default_to_four_mebibyte_halves_for_older_daemons():
    buffer = memoryview(bytearray(8 << 20))
    sizes = layout.read_sizes(buffer)
    assert (sizes.request, sizes.reply) == (4 << 20, 4 << 20)
    assert sizes.max_reply == (4 << 20) - layout.CTRL


def test_sizes_larger_than_the_mapping_are_refused():
    buffer = bytearray(1 << 20)
    buffer[layout.SIZES : layout.SIZES + 8] = (4 << 20).to_bytes(8, "little")
    with pytest.raises(ValueError, match="smaller than the halves"):
        layout.read_sizes(memoryview(buffer))


def test_frame_headers_round_trip():
    frame = frames.Frame(
        frames.FRAME, 5, 1, 3, 4096, 10000, 4096, "bfloat16", (1, 2, 2500)
    )
    packed = frames.pack(frame)
    assert len(packed) == frames.HEADER_BYTES
    assert frames.unpack(packed + b"\0" * 4096) == frame
    assert frames.unpack(frames.pack(frames.request(9, 0))) == frames.request(9, 0)


def test_frames_shorter_than_their_payload_are_refused():
    frame = frames.Frame(frames.FRAME, 1, 0, 1, 0, 64, 64, "float32", (16,))
    with pytest.raises(frames.FrameError, match="shorter than its header says"):
        frames.unpack(frames.pack(frame) + b"\0" * 10)


def test_foreign_payloads_are_not_frames():
    with pytest.raises(frames.FrameError, match="does not start with"):
        frames.unpack(b"\0" * frames.HEADER_BYTES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frames": 0},
        {"frame": 2, "frames": 2},
        {"offset": 60, "nbytes": 8, "total": 64},
        {"shape": (1,) * 9},
        {"dtype": "x" * 13},
    ],
)
def test_inconsistent_frames_are_refused(kwargs):
    base = {
        "kind": frames.FRAME,
        "message": 1,
        "frame": 0,
        "frames": 1,
        "offset": 0,
        "total": 64,
        "nbytes": 64,
        "dtype": "float32",
        "shape": (16,),
    }
    with pytest.raises(frames.FrameError):
        frames.Frame(**{**base, **kwargs})


def test_messages_split_into_frames_that_fit_the_reply_half():
    assert frames.plan(0, 100) == ((0, 0),)
    assert frames.plan(200, 100) == ((0, 100), (100, 100))
    assert frames.plan(250, 100) == ((0, 100), (100, 100), (200, 50))
    with pytest.raises(frames.FrameError):
        frames.plan(10, 0)
