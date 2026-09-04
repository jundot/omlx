def _assert_detached(exc):
    if exc is None:
        return
    assert exc.__traceback__ is None
    assert exc.__cause__ is None
    assert exc.__context__ is None


def test_optional_native_kernel_import_errors_do_not_retain_tracebacks():
    from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
    from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
    from omlx.patches.glm_moe_dsa.kernels import fast as glm_dispatch

    _assert_detached(glm_fast.import_error())
    _assert_detached(minimax_fast.import_error())
    _assert_detached(glm_dispatch.native_import_error())
