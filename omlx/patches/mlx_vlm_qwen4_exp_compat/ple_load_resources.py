"""Ownership of newly opened runtime PLE resources during one model load."""
from contextlib import contextmanager
from contextvars import ContextVar


_LOAD_RESOURCES = ContextVar("qwen4_ple_load_resources", default=None)


def register_ple_resource(resource, *, priority=0):
    """Track only resources created inside the current load, never old models."""
    resources = _LOAD_RESOURCES.get()
    if resources is not None:
        resources[id(resource)] = (priority, resource)


@contextmanager
def ple_load_resources():
    """Close new resources on failure; a successful load keeps model ownership.

    Each nested load has its own ownership scope. Embeddings have higher cleanup
    priority than raw readers so their workers drain before mmap teardown.
    """
    resources = {}
    token = _LOAD_RESOURCES.set(resources)
    try:
        yield
    except BaseException as primary:
        for _, resource in sorted(resources.values(), key=lambda item: -item[0]):
            try:
                resource.close()
            except BaseException as cleanup:
                # Preserve the load exception and continue cleaning other owners.
                primary.add_note(f"PLE cleanup failed: {type(cleanup).__name__}")
        raise
    finally:
        _LOAD_RESOURCES.reset(token)
        resources.clear()
