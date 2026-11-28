# SPDX-License-Identifier: Apache-2.0
"""Voice import boundary with a process-local simulated missing audio package."""

import subprocess
import sys
import unittest


class VoiceImportBoundaryTest(unittest.TestCase):
    def run_isolated(self, check):
        result = subprocess.run(
            [sys.executable, "-B", __file__, "--child", check],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonvoice_helpers_and_registry(self):
        self.run_isolated("test_nonvoice_helpers_and_registry")

    def test_nonvoice_engine(self):
        self.run_isolated("test_nonvoice_engine")

    def test_real_voice_request_reports_missing_audio(self):
        self.run_isolated("test_real_voice_request_reports_missing_audio")


def run_child(check):
    import asyncio
    from types import SimpleNamespace

    class MissingAudio:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mlx_audio" or fullname.startswith("mlx_audio."):
                raise ModuleNotFoundError("simulated missing mlx_audio", name="mlx_audio")

    sys.meta_path.insert(0, MissingAudio())
    from omlx.engine import vlm
    from omlx.engine.realtime_voice import RealtimeVoiceError
    from omlx.model_discovery import REALTIME_VOICE_MODEL_TYPES

    class VoiceChecks(unittest.TestCase):
        def test_nonvoice_helpers_and_registry(self):
            self.assertEqual(set(vlm._VOICE_BACKENDS), REALTIME_VOICE_MODEL_TYPES)
            for config in ({}, {"model_type": "sensenova_loop"}):
                self.assertEqual(vlm.realtime_voice_names(config), [])
                self.assertEqual(vlm.realtime_voice_instructions(config), "")
                self.assertEqual(
                    vlm.realtime_voice_capabilities(config),
                    {"input_image": False, "tools": False},
                )
            self.assertNotIn("mlx_audio", sys.modules)
            self.assertNotIn("omlx.engine.lfm3_realtime", sys.modules)
            self.assertNotIn("omlx.engine.qwen3_omni_realtime", sys.modules)

        def test_nonvoice_engine(self):
            engine = object.__new__(vlm.VLMBatchedEngine)
            engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type="text"))
            engine._engine = None
            engine._model_name = "text"
            self.assertFalse(engine.supports_realtime_voice)
            self.assertIsNone(engine.speech)
            with self.assertRaises(RealtimeVoiceError) as error:
                asyncio.run(engine.create_realtime_session())
            self.assertEqual(error.exception.code, "unsupported_model")

        def test_real_voice_request_reports_missing_audio(self):
            for model_type in ("lfm3_audio", "lfm3_omni", "qwen3_omni_moe"):
                with self.subTest(model_type=model_type):
                    with self.assertRaises(ModuleNotFoundError) as error:
                        vlm.realtime_voice_names({"model_type": model_type})
                    self.assertEqual(error.exception.name, "mlx_audio")
            with self.assertRaises(ModuleNotFoundError) as error:
                getattr(vlm, "LFM3VoiceBackend")
            self.assertEqual(error.exception.name, "mlx_audio")

    suite = unittest.TestSuite([VoiceChecks(check)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        raise SystemExit(run_child(sys.argv[2]))
    unittest.main()
