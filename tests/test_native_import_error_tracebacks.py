def _assert_detached(exc):
    if exc is None:
        return
    assert exc.__traceback__ is None
    assert exc.__cause__ is None
    assert exc.__context__ is None


def test_optional_native_kernel_import_errors_do_not_retain_tracebacks():
    from omlx.custom_kernels.attention import fast as attention_fast
    from omlx.custom_kernels.lowbit import fast as lowbit_fast
    from omlx.custom_kernels.quant_matmul import fast as quant_fast
    from omlx.custom_kernels.sparse_index import fast as index_fast
    from omlx.patches.glm_moe_dsa.kernels import fast as glm_dispatch

    for fast in (attention_fast, index_fast, quant_fast, lowbit_fast):
        _assert_detached(fast.import_error())
    _assert_detached(glm_dispatch.native_import_error())
