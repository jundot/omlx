from ..qwen3_vl import VisionModel as Qwen3VLVisionModel


class VisionModel(Qwen3VLVisionModel):
    def __init__(self, config):
        # The pinned qwen3_vl tower predates the qwen4_exp type aliases; its
        # graph is otherwise the same published ViT used by Flash Next.
        original_type = config.model_type
        config.model_type = "qwen3_5_moe_vision"
        try:
            super().__init__(config)
        finally:
            config.model_type = original_type
        self.model_type = original_type
