"""Worker-stream cleanup, including shutdown invoked from another thread."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from jangh2_bridge.lifecycle import install_scheduler_shutdown


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.owner = self.executor.submit(threading.get_ident).result()
        self.events = []
        events, owner = self.events, self.owner

        class Generator:
            def __init__(self):
                self._old_wired_limit = 123

            def close(self):
                if self._old_wired_limit is not None:
                    if threading.get_ident() != owner:
                        raise RuntimeError('There is no generation Stream in current thread')
                    events.append(('synchronize and restore', threading.get_ident()))
                    self._old_wired_limit = None

        class Scheduler:
            def __init__(self):
                self.batch_generator = Generator()

            def shutdown(self):
                events.append(('scheduler shutdown', threading.get_ident()))
                return 'closed'

        self.Scheduler = Scheduler
        self.original = Scheduler.shutdown
        self.undo = install_scheduler_shutdown(Scheduler, self.executor, self.owner)

    def tearDown(self):
        self.undo()
        self.executor.shutdown(wait=True)

    def test_main_thread_shutdown_closes_on_worker_before_teardown(self):
        scheduler = self.Scheduler()
        self.assertEqual(scheduler.shutdown(), 'closed')
        self.assertEqual(self.events, [
            ('synchronize and restore', self.owner),
            ('scheduler shutdown', threading.get_ident()),
        ])
        # Same close path used by BatchGenerator.__del__, now safe on main.
        scheduler.batch_generator.close()
        self.assertIsNone(scheduler.batch_generator._old_wired_limit)

    def test_worker_shutdown_closes_directly_without_submitting_to_itself(self):
        scheduler = self.Scheduler()
        self.assertEqual(self.executor.submit(scheduler.shutdown).result(timeout=2), 'closed')
        self.assertEqual(self.events, [
            ('synchronize and restore', self.owner),
            ('scheduler shutdown', self.owner),
        ])

    def test_no_generator_requires_no_gpu_cleanup(self):
        scheduler = self.Scheduler()
        scheduler.batch_generator = None
        scheduler.shutdown()
        self.assertEqual(self.events, [('scheduler shutdown', threading.get_ident())])

    def test_close_failure_is_not_silenced(self):
        scheduler = self.Scheduler()

        def fail():
            raise RuntimeError('GPU synchronization failed')

        scheduler.batch_generator.close = fail
        with self.assertRaisesRegex(RuntimeError, 'GPU synchronization failed'):
            scheduler.shutdown()
        self.assertEqual(self.events, [])

    def test_undo_restores_original_and_respects_later_hook(self):
        self.undo()
        self.assertIs(self.Scheduler.shutdown, self.original)
        undo = install_scheduler_shutdown(self.Scheduler, self.executor, self.owner)
        replacement = lambda self: None
        self.Scheduler.shutdown = replacement
        undo()
        self.assertIs(self.Scheduler.shutdown, replacement)
