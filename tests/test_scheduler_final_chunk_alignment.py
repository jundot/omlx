from omlx.scheduler import _align_final_prefill_chunk


def test_unaligned_final_chunk_is_split_to_64_aligned_prefix():
    # 350K prompt remainder: L=1817, L%64=25 -> 1792 aligned prefix, 25-token tail deferred
    assert _align_final_prefill_chunk(1817, 1817) == 1792
    assert _align_final_prefill_chunk(793, 793) == 768


def test_aligned_final_chunk_unchanged():
    assert _align_final_prefill_chunk(1024, 1024) == 1024
    assert _align_final_prefill_chunk(2048, 2048) == 2048


def test_non_final_chunk_unchanged():
    # throttled mid-prefill chunk: chunk_len < remaining
    assert _align_final_prefill_chunk(1024, 1817) == 1024


def test_small_tail_unchanged_and_never_zero():
    assert _align_final_prefill_chunk(25, 25) == 25
    assert _align_final_prefill_chunk(64, 64) == 64
    assert _align_final_prefill_chunk(65, 65) == 64  # smallest split, never 0
