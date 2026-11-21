"""Torch-free macOS/Apple Silicon media decoding and WeMM token-layout helpers."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, unquote_to_bytes, urlparse

import numpy as np
import soundfile as sf
from PIL import Image, ImageOps
from scipy.signal import resample_poly

IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"
AUDIO_MARKER = "<|audio_start|><|audio_pad|><|audio_end|>"

IMAGE_FACTOR = 32
FRAME_FACTOR = 2


def resolve_media(source: Any) -> Any:
    """Resolve data URI, HTTP(S) URL, or file:// URI into bytes or local path."""
    if isinstance(source, str) and source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator:
            raise ValueError("Malformed media data URL")
        data = unquote_to_bytes(payload)
        return base64.b64decode(data, validate=True) if header.endswith(";base64") else data
    if isinstance(source, str) and source.startswith(("https://", "http://")):
        import urllib.request

        with urllib.request.urlopen(source, timeout=120) as response:
            return response.read()
    if isinstance(source, str) and source.startswith("file://"):
        return unquote(urlparse(source).path)
    return source


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = 4096,
    max_pixels: int = 25165824,
) -> Tuple[int, int]:
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError(f"Extreme aspect ratio: {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def load_image(source: Any) -> Image.Image:
    source = resolve_media(source)
    if isinstance(source, (str, Path, bytes)):
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        arr = source
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image source type: {type(source)}")


def pcm16_roundtrip(samples: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    out, _ = sf.read(buffer, dtype="float32")
    return np.asarray(out, dtype=np.float32)


def _ffmpeg_decode_audio(source: str | bytes, target_sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0" if isinstance(source, bytes) else str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=source if isinstance(source, bytes) else None,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"No audio stream or ffmpeg decode failed: {stderr}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_audio(source: Any, target_sr: int = 16000) -> np.ndarray:
    """Decode file path, bytes, (waveform, sr), or {array, sampling_rate} into mono 16kHz float32."""
    if isinstance(source, dict):
        if source.get("array") is not None:
            source = (source["array"], source.get("sampling_rate", target_sr))
        else:
            source = source["path"]
    source = resolve_media(source)
    if isinstance(source, np.ndarray):
        samples = np.asarray(source, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
    elif isinstance(source, tuple) and len(source) == 2:
        samples, rate = source
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
        rate = int(rate)
        if rate <= 0:
            raise ValueError("Audio sampling rate must be positive")
        if rate != target_sr:
            divisor = math.gcd(rate, target_sr)
            samples = resample_poly(samples, target_sr // divisor, rate // divisor).astype(np.float32)
    else:
        samples = None
        try:
            bio = io.BytesIO(source) if isinstance(source, bytes) else str(source)
            data, rate = sf.read(bio, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            if rate != target_sr:
                divisor = math.gcd(rate, target_sr)
                data = resample_poly(data, target_sr // divisor, rate // divisor).astype(np.float32)
            samples = np.asarray(data, dtype=np.float32)
        except Exception:
            samples = _ffmpeg_decode_audio(source, target_sr=target_sr)

    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    return samples


def audio_features(
    samples: np.ndarray,
    extractor: Any,
    *,
    pcm16: bool = True,
) -> np.ndarray:
    """Slice audio into 30s (480,000 samples) chunks and compute (N, 128, 3000) log-mel features."""
    if pcm16:
        samples = pcm16_roundtrip(samples)
    else:
        samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio samples must be finite, nonempty mono float32")
    if len(samples) < 16000:
        samples = np.pad(samples, (0, 16000 - len(samples)))

    chunks = []
    for start in range(0, len(samples), 480000):
        seg = samples[start : start + 480000]
        out = extractor(seg, sampling_rate=16000, return_tensors="np")
        chunks.append(np.asarray(out["input_features"], dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _probe_video_ffprobe(path: str) -> Tuple[int, float, int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height,duration:format=duration",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    def _parse_fps(val: Optional[str]) -> float:
        if not val or val == "0/0":
            return 0.0
        if "/" in val:
            num, den = val.split("/", 1)
            return float(num) / float(den) if float(den) > 0 else 0.0
        return float(val)

    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")) or 24.0
    dur_str = stream.get("duration") or info.get("format", {}).get("duration")
    duration = float(dur_str) if dur_str and dur_str != "N/A" else 0.0
    nb_frames_str = stream.get("nb_frames")
    if nb_frames_str and str(nb_frames_str).isdigit() and int(nb_frames_str) > 0:
        total_frames = int(nb_frames_str)
    elif duration > 0:
        total_frames = max(1, int(round(duration * fps)))
    else:
        total_frames = 0
    if duration <= 0 and total_frames > 0:
        duration = total_frames / fps
    return total_frames, fps, width, height, duration


def compute_video_frame_count(
    total_frames: int,
    origin_fps: float,
    *,
    fps: float = 2.0,
    num_frames: Optional[int] = None,
    min_frames: int = 4,
    max_frames: int = 128,
) -> int:
    if num_frames is not None:
        count = int(num_frames)
    else:
        count = int(round(total_frames / origin_fps * float(fps)))
        count = max(min_frames, min(max_frames, count))
    count = min(count, total_frames)
    if count < FRAME_FACTOR:
        count = min(total_frames, FRAME_FACTOR)
    if count > FRAME_FACTOR and count % FRAME_FACTOR != 0:
        count = count - (count % FRAME_FACTOR)
    return max(1, count)


def _decode_video_cv2(path: str, indices: List[int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")
    frames = []
    target_set = set(indices)
    max_idx = max(indices)
    if max_idx < 3000:
        cache: Dict[int, np.ndarray] = {}
        idx = 0
        while idx <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in target_set:
                cache[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
        cap.release()
        if not cache:
            raise RuntimeError(f"OpenCV decoded zero frames from {path}")
        for req in indices:
            if req in cache:
                frames.append(cache[req])
            else:
                nearest = min(cache.keys(), key=lambda k: abs(k - req))
                frames.append(cache[nearest])
        return np.stack(frames, axis=0)

    for req in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, req)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1])
    cap.release()
    if not frames:
        raise RuntimeError(f"OpenCV decoded zero frames from {path}")
    return np.stack(frames, axis=0)


def _decode_video_ffmpeg(path: str, indices: List[int], width: int, height: int) -> np.ndarray:
    expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"select='{expr}'",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bytes = width * height * 3
    decoded_count = len(raw) // frame_bytes
    arr = raw[: decoded_count * frame_bytes].reshape(decoded_count, height, width, 3)
    if decoded_count == len(indices):
        return arr
    uniq = sorted(set(indices))
    mapping = {u: arr[min(i, decoded_count - 1)] for i, u in enumerate(uniq)}
    return np.stack([mapping[i] for i in indices], axis=0)


def _decode_zip_video(
    raw_bytes: bytes,
    *,
    fps: float = 2.0,
    num_frames: Optional[int] = None,
    max_frames: int = 128,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
        names = sorted(zf.namelist())
        img_files = [
            f
            for f in names
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            and not f.startswith("__MACOSX/")
        ]
        audio_files = [
            f
            for f in names
            if f.lower().endswith((".wav", ".mp3", ".flac"))
            and not f.startswith("__MACOSX/")
        ]
        if not img_files:
            raise RuntimeError("ZIP archive contains no image frames")
        total = len(img_files)
        limit = min(total, max_frames) if num_frames is None else min(total, int(num_frames))
        if limit == 1:
            indices = [0]
        else:
            indices = [i * (total - 1) // (limit - 1) for i in range(limit)]
        frames = []
        first_size = None
        for idx in indices:
            img = Image.open(io.BytesIO(zf.read(img_files[idx]))).convert("RGB")
            if first_size is None:
                first_size = img.size
            elif img.size != first_size:
                img = img.resize(first_size, Image.Resampling.BICUBIC)
            frames.append(np.asarray(img, dtype=np.uint8))
        arr = np.stack(frames, axis=0)
        meta: Dict[str, Any] = {
            "total_num_frames": total,
            "fps": float(fps),
            "frames_indices": indices,
            "duration": total / float(fps),
            "width": int(arr.shape[2]),
            "height": int(arr.shape[1]),
            "video_backend": "zip",
        }
        if audio_files:
            meta["audio_bytes"] = zf.read(audio_files[0])
        return arr, meta


def decode_video(
    source: Any,
    *,
    fps: float = 2.0,
    num_frames: Optional[int] = None,
    min_frames: int = 4,
    max_frames: int = 128,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode local/bytes/ZIP video into uint8 RGB array (T, H, W, 3) + WeMM metadata."""
    source = resolve_media(source)
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        n = len(arr)
        return arr, {
            "total_num_frames": n,
            "fps": float(fps),
            "frames_indices": list(range(n)),
            "duration": n / float(fps),
            "width": int(arr.shape[2]),
            "height": int(arr.shape[1]),
            "video_backend": "array",
        }

    header = source[:4] if isinstance(source, bytes) else Path(source).read_bytes()[:4]
    if header == b"PK\x03\x04":
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        return _decode_zip_video(raw, fps=fps, num_frames=num_frames, max_frames=max_frames)

    tmp_path = None
    try:
        if isinstance(source, bytes):
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as f:
                f.write(source)
            video_path = tmp_path
        else:
            video_path = str(source)

        total_frames, origin_fps, width, height, duration = _probe_video_ffprobe(video_path)
        if total_frames <= 0:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames <= 0:
                raise RuntimeError(f"Could not determine frame count for {video_path}")

        count = compute_video_frame_count(
            total_frames,
            origin_fps,
            fps=fps,
            num_frames=num_frames,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        indices = np.linspace(0, total_frames - 1, count, dtype=int).tolist()
        try:
            frames = _decode_video_cv2(video_path, indices)
        except Exception:
            frames = _decode_video_ffmpeg(video_path, indices, width, height)

        metadata = {
            "total_num_frames": total_frames,
            "fps": float(origin_fps),
            "frames_indices": indices,
            "duration": float(duration if duration > 0 else total_frames / origin_fps),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "video_backend": "opencv_ffmpeg",
        }
        return frames, metadata
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def tokens_per_audio_segment(audio_config: Dict[str, Any]) -> int:
    window = float(audio_config.get("second_per_window", 0.1))
    stride = float(audio_config.get("second_stride", window))
    qfc = audio_config.get("qformer_config", {})
    query_length = int(qfc.get("query_length", 1))
    kernel = round(1500 * window / 30.0)
    step = max(1, round(1500 * stride / 30.0))
    return ((1500 - kernel) // step + 1) * query_length


def audio_timestamp_groups(
    count: int,
    window: float = 0.1,
    duration: Optional[float] = None,
    timestamp_interval_sec: float = 1.0,
) -> Tuple[List[int], List[float], int]:
    count = int(count)
    if count <= 0:
        return [], [], 0
    padded_duration = count * float(window)
    if duration is None:
        duration, effective = padded_duration, count
    else:
        duration = min(float(duration), padded_duration)
        effective = max(1, min(count, math.ceil(duration / float(window) - 1e-9)))
    sample_fps = 1.0 / float(timestamp_interval_sec)
    groups = max(1, math.ceil(duration * sample_fps - 1e-9))
    base, remainder = divmod(effective, groups)
    sizes = [base] * groups
    step = float(groups) / float(remainder) if remainder else 0.0
    for i in range(1, remainder + 1):
        sizes[min(groups - 1, math.floor(i * step) - 1)] += 1
    sizes[-1] += count - effective
    times = [index / sample_fps for index in range(groups)]
    return sizes, times, effective


def build_audio_token_ids(
    tokenizer: Any,
    count: int,
    window: float = 0.1,
    duration: Optional[float] = None,
    timestamp_interval_sec: float = 1.0,
) -> List[int]:
    sizes, times, _ = audio_timestamp_groups(count, window, duration, timestamp_interval_sec)
    audio_start = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    audio_pad = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    result = [audio_start]
    for ts, size in zip(times, sizes):
        result.extend(tokenizer.encode(f"<{ts:.1f} seconds>", add_special_tokens=False))
        result.extend([audio_pad] * size)
    result.append(audio_end)
    return result


def build_video_token_ids(
    tokenizer: Any,
    grid_thw: Tuple[int, int, int],
    metadata: Dict[str, Any],
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    audio_count: int = 0,
    audio_duration: Optional[float] = None,
    audio_window: float = 0.1,
) -> Tuple[List[int], List[int], List[float]]:
    time_steps, grid_h, grid_w = (int(x) for x in grid_thw)
    indices = list(metadata["frames_indices"])
    indices += [indices[-1]] * max(0, time_steps * temporal_patch_size - len(indices))

    if audio_count > 0:
        sampled_fps = len(metadata["frames_indices"]) / float(audio_duration)
        step_ratio = np.float32(temporal_patch_size / sampled_fps / audio_window)
        boundary = np.floor(np.arange(1, time_steps, dtype=np.float32) * step_ratio).astype(np.int64)
        boundary = np.clip(boundary, 0, audio_count).tolist()
        bounds = [0, *boundary, audio_count]
        groups = [b - a for a, b in zip(bounds, bounds[1:])]
    else:
        groups = [0] * time_steps

    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    video_pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    audio_pad = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    tokens_per_step = (grid_h * grid_w) // (merge_size**2)

    result: List[int] = []
    timestamps: List[float] = []
    origin_fps = float(metadata["fps"])
    for idx in range(time_steps):
        left = indices[idx * temporal_patch_size]
        right = indices[(idx + 1) * temporal_patch_size - 1]
        if audio_count > 0:
            seconds = (left + right) / (2.0 * origin_fps)
        else:
            seconds = (left / origin_fps + right / origin_fps) / 2.0
        timestamps.append(seconds)
        result.extend(tokenizer.encode(f"<{seconds:.1f} seconds>", add_special_tokens=False))
        result.append(vision_start)
        result.extend([video_pad] * tokens_per_step)
        if groups[idx] > 0:
            result.extend([audio_pad] * groups[idx])
        result.append(vision_end)
    return result, groups, timestamps


def preprocess_frames_to_patches(
    frames: np.ndarray,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 4096,
    max_pixels: int = 25165824,
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    is_video: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Torch-free Qwen3-VL / WeMM patchifier for images (1 frame) or videos (T frames)."""
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    num_frames, height, width, _ = frames.shape
    factor = patch_size * merge_size
    per_frame_max = max(min_pixels, max_pixels // max(1, num_frames)) if is_video else max_pixels
    resized_h, resized_w = smart_resize(
        height, width, factor=factor, min_pixels=min_pixels, max_pixels=per_frame_max
    )

    resized_list = []
    for i in range(num_frames):
        img = Image.fromarray(frames[i])
        if img.size != (resized_w, resized_h):
            img = img.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        resized_list.append(arr)
    arr = np.stack(resized_list, axis=0)

    mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(image_std, dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))

    rem = arr.shape[0] % temporal_patch_size
    if rem != 0:
        pad_count = temporal_patch_size - rem
        arr = np.concatenate([arr, np.repeat(arr[-1:], pad_count, axis=0)], axis=0)

    grid_t = arr.shape[0] // temporal_patch_size
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    channels = arr.shape[1]

    patches = arr.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return flatten_patches.astype(np.float32), grid_thw


def detect_video_shots(
    video_path: str | Path,
    *,
    min_shot_sec: float = 0.8,
    bhat_threshold: float = 0.28,
    mad_threshold: float = 0.035,
    strong_bhat_threshold: float = 0.38,
) -> Tuple[List[Tuple[float, float]], float]:
    """Detect hard shot cuts using letterbox-cropped 2x2 spatial HSV histograms + luminance MAD.

    Returns (shots, duration_sec) where shots is a list of (start_sec, end_sec) tuples.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for shot detection: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)

    def _sig(frame: np.ndarray):
        small = cv2.resize(frame, (160, 90))
        # Exclude top/bottom letterbox bars when computing dark-scene cut scores
        crop = small[12:78, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ch, cw, _ = hsv.shape
        hists = []
        for r in (0, ch // 2):
            for c in (0, cw // 2):
                cell = hsv[r : r + ch // 2, c : c + cw // 2]
                hi = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hi, hi)
                hists.append(hi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return hists, gray

    prev = None
    diffs: List[Tuple[float, float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sig = _sig(frame)
        if prev is not None:
            bhat = float(
                np.mean([cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) for a, b in zip(prev[0], sig[0])])
            )
            mad = float(np.mean(np.abs(sig[1] - prev[1])))
            diffs.append((idx / fps, bhat, mad))
        prev = sig
        idx += 1
    cap.release()

    duration = idx / fps if fps > 0 else 0.0
    if not diffs:
        return [(0.0, duration)], duration

    bhats = np.array([d[1] for d in diffs], dtype=np.float32)
    mads = np.array([d[2] for d in diffs], dtype=np.float32)
    cuts = [0.0]
    for i, (t, b, m) in enumerate(diffs):
        lo = max(0, i - 8)
        hi = min(len(diffs), i + 9)
        med_b = float(np.median(bhats[lo:hi]))
        med_m = float(np.median(mads[lo:hi]))
        is_cut = (b > max(bhat_threshold, med_b * 2.2) and m > max(mad_threshold, med_m * 2.0)) or (
            b > strong_bhat_threshold
        )
        if is_cut and (t - cuts[-1] >= min_shot_sec) and (duration - t >= 0.4):
            cuts.append(t)
    cuts.append(duration)
    shots = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    return shots, duration
