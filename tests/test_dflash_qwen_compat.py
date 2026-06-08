from types import SimpleNamespace


def test_dflash_qwen_compat_prefers_language_model(monkeypatch):
    from omlx.patches.dflash_qwen_compat import install_dflash_qwen_compat_patch

    class QwenGdnTargetOps:
        def text_wrapper(self, target_model):
            if hasattr(target_model, "model"):
                return target_model
            if hasattr(target_model, "language_model"):
                return target_model.language_model
            raise AttributeError

    fake_target_qwen_gdn = SimpleNamespace(QwenGdnTargetOps=QwenGdnTargetOps)

    def fake_import(name, *args, **kwargs):
        if name == "dflash_mlx.engine":
            return SimpleNamespace(target_qwen_gdn=fake_target_qwen_gdn)
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert install_dflash_qwen_compat_patch() is True

    ops = QwenGdnTargetOps()
    text = SimpleNamespace(model=object(), args=SimpleNamespace(tie_word_embeddings=False))
    outer = SimpleNamespace(model=object(), language_model=text)

    assert ops.text_wrapper(outer) is text
