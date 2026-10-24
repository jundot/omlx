"""Close scheduler-owned generation resources on their MLX stream thread."""
from __future__ import annotations

import functools
import threading


def install_scheduler_shutdown(scheduler_type, executor, owner_thread):
    """Close the current BatchGenerator before main-thread model teardown.

    The PR scheduler leaves its generator alive after shutdown. Its eventual
    destructor restores MLX's wired-memory limit by synchronizing the generation
    stream, which this bridge creates on the MLX executor. That synchronization
    must happen there, while the worker and its stream are still installed.
    BatchGenerator.close() clears its wired-limit marker, so the later destructor
    becomes an ordinary no-op. Other shutdown errors are allowed to propagate.
    """
    original = scheduler_type.shutdown

    @functools.wraps(original)
    def shutdown(self, *args, **kwargs):
        generator = getattr(self, 'batch_generator', None)
        if generator is not None:
            if threading.get_ident() == owner_thread:
                generator.close()
            else:
                executor.submit(generator.close).result()
        return original(self, *args, **kwargs)

    scheduler_type.shutdown = shutdown

    def undo():
        if scheduler_type.shutdown is shutdown:
            scheduler_type.shutdown = original

    return undo
