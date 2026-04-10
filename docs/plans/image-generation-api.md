# Image Generation API Implementation Plan

## Overview

Add image generation capabilities (Text-to-Image, Image-to-Image) to oMLX by integrating [mflux](https://github.com/filipstrand/mflux), a native MLX image generation library.

## Goals

- Support Text-to-Image (T2I) generation
- Support Image-to-Image (I2I) transformation
- OpenAI Images API compatibility (`/v1/images/generations`)
- Leverage existing EnginePool architecture for model management

## Supported Models (via mflux)

| Model | Type | Notes |
|-------|------|-------|
| Z-Image / Z-Image-Turbo | T2I | Fast, high quality |
| FLUX.1-dev / FLUX.1-schnell | T2I | Popular diffusion models |
| FLUX.2 | T2I | Latest generation |
| FIBO | T2I | Efficient model |
| SeedVR2 | T2I | High quality |
| Qwen Image | T2I | Qwen's image model |

## Architecture

### New Components

```
omlx/
├── engine/
│   └── image.py              # ImageEngine (BaseNonStreamingEngine)
├── api/
│   └── image_models.py       # Pydantic models for Images API
│   └── image_routes.py       # FastAPI routes (optional, can be in server.py)
└── model_discovery.py        # Add IMAGE_MODEL_TYPES detection
```

### Class Design

```python
# omlx/engine/image.py

class ImageEngine(BaseNonStreamingEngine):
    """Engine for image generation using mflux."""

    def __init__(self, model_name: str, **kwargs):
        self._model_name = model_name
        self._model = None  # mflux model instance
        self._kwargs = kwargs

    async def generate_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        image: Optional[str] = None,  # For I2I (base64 or path)
        strength: float = 0.8,        # For I2I
    ) -> PIL.Image.Image:
        """Generate image from text prompt (T2I) or transform existing image (I2I)."""
        pass

    async def start(self) -> None:
        """Load model lazily (mflux import happens here)."""
        pass

    async def stop(self) -> None:
        """Unload model and free memory."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        pass
```

## API Design

### OpenAI-Compatible Endpoint

```
POST /v1/images/generations
```

**Request Body:**

```json
{
  "model": "flux.1-schnell",
  "prompt": "A white Siamese cat walking through a quiet neighborhood",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard",  // standard | hd (maps to num_inference_steps)
  "response_format": "b64_json",  // b64_json | url
  "style": "vivid",  // vivid | natural (maps to guidance_scale)
  "negative_prompt": "blurry, low quality",  // omlx extension
  "seed": 42,  // omlx extension
  "image": "data:image/png;base64,...",  // omlx extension for I2I
  "strength": 0.8  // omlx extension for I2I
}
```

**Response:**

```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "base64-encoded-image-data",
      "revised_prompt": "A white Siamese cat..."  // If prompt was modified
    }
  ]
}
```

### Pydantic Models

```python
# omlx/api/image_models.py

class ImageRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    model: str
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    quality: Literal["standard", "hd"] = "standard"
    response_format: Literal["url", "b64_json"] = "b64_json"
    style: Optional[Literal["vivid", "natural"]] = None
    user: Optional[str] = None

    # omlx extensions
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None  # Override quality mapping
    guidance_scale: Optional[float] = None
    image: Optional[str] = None  # Base64 image for I2I
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ImageData(BaseModel):
    """Single generated image result."""

    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    """OpenAI-compatible image generation response."""

    created: int
    data: List[ImageData]
```

## Model Discovery

Add image model detection to `model_discovery.py`:

```python
# Model types for image generation (diffusion models)
IMAGE_MODEL_TYPES = {
    "flux",
    "flux1",
    "flux.1",
    "flux2",
    "flux.2",
    "z_image",
    "z-image",
    "fibo",
    "seedvr",
    "seedvr2",
    "qwen_image",
}

# Architectures for image models
IMAGE_ARCHITECTURES = {
    "FluxPipeline",
    "ZImagePipeline",
    "FIBOPipeline",
    "SeedVR2Pipeline",
}

ModelType = Literal[
    "llm", "vlm", "embedding", "reranker",
    "audio_stt", "audio_tts", "audio_sts",
    "image_t2i"  # NEW
]
EngineType = Literal[
    "batched", "vlm", "embedding", "reranker",
    "audio_stt", "audio_tts", "audio_sts",
    "image"  # NEW
]
```

## Parameter Mappings

### Size → Width/Height

| Size | Width | Height |
|------|-------|--------|
| 256x256 | 256 | 256 |
| 512x512 | 512 | 512 |
| 1024x1024 | 1024 | 1024 |
| 1792x1024 | 1792 | 1024 |
| 1024x1792 | 1024 | 1792 |

### Quality → Steps

| Quality | num_inference_steps |
|---------|---------------------|
| standard | 20 |
| hd | 40 |

### Style → Guidance Scale

| Style | guidance_scale |
|-------|----------------|
| vivid | 5.0 |
| natural | 3.5 |
| None (default) | 3.5 |

## Implementation Steps

### Phase 1: Core Engine (2-3 days)

1. **Add mflux dependency** (optional extra)
   ```toml
   # pyproject.toml
   [project.optional-dependencies]
   image = [
       "mflux>=0.5.0",
   ]
   ```

2. **Implement ImageEngine** (`omlx/engine/image.py`)
   - Lazy import mflux in `start()`
   - Map model names to mflux classes
   - Handle T2I and I2I generation
   - Memory management (clear cache after generation)

3. **Update model_discovery.py**
   - Add IMAGE_MODEL_TYPES and IMAGE_ARCHITECTURES
   - Add "image_t2i" to ModelType/EngineType
   - Implement `detect_image_model_type()`

### Phase 2: API Layer (1-2 days)

4. **Add Pydantic models** (`omlx/api/image_models.py`)
   - ImageRequest, ImageData, ImageResponse

5. **Add API route** (in `server.py` or new `api/image_routes.py`)
   - `POST /v1/images/generations`
   - Support both b64_json and url response formats

6. **Update /v1/models endpoint**
   - Include image models with `model_type: "image_t2i"`

### Phase 3: Integration (1-2 days)

7. **Update EnginePool**
   - Handle "image" engine type
   - Add image models to LRU management

8. **Add admin panel support** (optional)
   - Display image models
   - Show generation stats

9. **Add tests**
   - Unit tests for ImageEngine
   - Integration tests for /v1/images/generations

## Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
image = [
    "mflux>=0.5.0",
    "Pillow>=9.0.0",  # Already in core deps
]
```

## Testing Strategy

```python
# tests/test_image_engine.py

def test_image_engine_load():
    """Test image engine loads mflux model correctly."""
    pass

def test_t2i_generation():
    """Test text-to-image generation."""
    pass

def test_i2i_generation():
    """Test image-to-image transformation."""
    pass

def test_openai_api_compatibility():
    """Test /v1/images/generations endpoint."""
    pass

# Mark as slow since they load real models
pytest.mark.slow
```

## Open Questions

1. **URL response format**: Should we serve generated images via a temporary endpoint?
   - Option A: Only support `b64_json` (simpler)
   - Option B: Add `/v1/images/{id}` endpoint for temporary URLs

2. **Batch generation**: OpenAI supports `n=1` only for DALL-E-3
   - omlx could support `n>1` for all models (serial generation)

3. **Model aliasing**: How to handle model name mapping?
   - `flux.1-schnell` → mflux `Flux1Schnell`
   - `z-image-turbo` → mflux `ZImageTurbo`

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| mflux API changes | Pin version, lazy import with try/except |
| High memory usage | Integrate with ProcessMemoryEnforcer |
| Slow generation | Run in MLX executor, add progress callbacks |
| I2I complexity | Start with T2I only, add I2I in Phase 2 |

## Success Criteria

- [ ] Can load and generate images with FLUX models
- [ ] OpenAI Images API compatibility
- [ ] Model discovery detects image models automatically
- [ ] Memory management via EnginePool LRU
- [ ] Tests passing for core functionality