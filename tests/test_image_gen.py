#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Image generation test script

Generate yellow-themed images using specified models.
"""

import base64
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from io import BytesIO


# API configuration
API_BASE = "http://localhost:8888/v1"
MODEL_ID = "FLUX.2-klein-4B-mflux-4bit"


def generate_image(
    prompt: str,
    model: str = MODEL_ID,
    size: str = "1024x1024",
    n: int = 1,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
) -> dict:
    """Call the image generation API.

    Args:
        prompt: Image description prompt
        model: Model ID
        size: Image size (e.g., "1024x1024")
        n: Number of images to generate
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed (for reproducible generation)
        negative_prompt: Negative prompt

    Returns:
        API response data
    """
    url = f"{API_BASE}/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json",
    }

    # Add optional parameters
    if num_inference_steps is not None:
        payload["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt

    print(f"Sending request to: {url}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Size: {size}")
    if seed is not None:
        print(f"Seed: {seed}")

    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)


def save_b64_image(b64_data: str, output_path: Path) -> None:
    """Save base64-encoded image to file.

    Args:
        b64_data: Base64-encoded image data
        output_path: Output file path
    """
    image_data = base64.b64decode(b64_data)
    img = Image.open(BytesIO(image_data))

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    img.save(output_path)
    print(f"Image saved to: {output_path}")


def main():
    """Main function."""
    # Yellow-themed prompts
    prompts = [
        "A vibrant yellow sunflower field at sunset, warm golden lighting",
        # "A cute yellow chick sitting on a green leaf, soft studio lighting",
        # "A yellow vintage bicycle parked against a brick wall, cinematic lighting",
    ]

    # Can also specify prompt directly via command line
    if len(sys.argv) > 1:
        prompts = [sys.argv[1]]

    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Generating image {i+1}/{len(prompts)}")
        print(f"{'='*60}")

        # Use fixed seed for reproducible results
        response = generate_image(
            prompt=prompt,
            size="1024x1024",
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42 + i,
        )

        # Save generated images
        for j, image_data in enumerate(response["data"]):
            b64_json = image_data.get("b64_json")
            if b64_json:
                # Generate short filename from prompt
                safe_prompt = "".join(
                    c if c.isalnum() or c in (" ", "_") else "_"
                    for c in prompt[:30]
                ).strip()
                safe_prompt = safe_prompt.replace(" ", "_")

                output_path = output_dir / f"yellow_{i+1}_{j+1}_{safe_prompt}.png"
                save_b64_image(b64_json, output_path)
            else:
                print("Warning: No base64 image data in response")

        # Save full response JSON (for debugging)
        json_path = output_dir / f"yellow_{i+1}_response.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"Response saved to: {json_path}")

    print(f"\n{'='*60}")
    print("Image generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
