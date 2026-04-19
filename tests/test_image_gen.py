#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
图像生成测试脚本

使用指定模型生成黄色主题的图像。
"""

import base64
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from io import BytesIO


# API配置
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
    """调用图像生成API。

    Args:
        prompt: 图像描述提示词
        model: 模型ID
        size: 图像尺寸（如 "1024x1024"）
        n: 生成图像数量
        num_inference_steps: 推理步数
        guidance_scale: 引导系数
        seed: 随机种子（用于可重复生成）
        negative_prompt: 负面提示词

    Returns:
        API响应数据
    """
    url = f"{API_BASE}/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json",
    }

    # 添加可选参数
    if num_inference_steps is not None:
        payload["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt

    print(f"发送请求到: {url}")
    print(f"模型: {model}")
    print(f"提示词: {prompt}")
    print(f"尺寸: {size}")
    if seed is not None:
        print(f"种子: {seed}")

    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP错误: {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"请求失败: {e}")
        sys.exit(1)


def save_b64_image(b64_data: str, output_path: Path) -> None:
    """将base64编码的图像保存到文件。

    Args:
        b64_data: base64编码的图像数据
        output_path: 输出文件路径
    """
    image_data = base64.b64decode(b64_data)
    img = Image.open(BytesIO(image_data))

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图像
    img.save(output_path)
    print(f"图像已保存到: {output_path}")


def main():
    """主函数。"""
    # 黄色主题的提示词
    prompts = [
        "A vibrant yellow sunflower field at sunset, warm golden lighting",
        # "A cute yellow chick sitting on a green leaf, soft studio lighting",
        # "A yellow vintage bicycle parked against a brick wall, cinematic lighting",
    ]

    # 也可以直接在命令行指定提示词
    if len(sys.argv) > 1:
        prompts = [sys.argv[1]]

    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"生成图像 {i+1}/{len(prompts)}")
        print(f"{'='*60}")

        # 使用固定种子以获得可重复的结果
        response = generate_image(
            prompt=prompt,
            size="1024x1024",
            num_inference_steps=4,
            guidance_scale=3.5,
            seed=42 + i,
        )

        # 保存生成的图像
        for j, image_data in enumerate(response["data"]):
            b64_json = image_data.get("b64_json")
            if b64_json:
                # 从提示词生成简短的文件名
                safe_prompt = "".join(
                    c if c.isalnum() or c in (" ", "_") else "_"
                    for c in prompt[:30]
                ).strip()
                safe_prompt = safe_prompt.replace(" ", "_")

                output_path = output_dir / f"yellow_{i+1}_{j+1}_{safe_prompt}.png"
                save_b64_image(b64_json, output_path)
            else:
                print("警告: 响应中没有base64图像数据")

        # 保存完整的响应JSON（用于调试）
        json_path = output_dir / f"yellow_{i+1}_response.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"响应已保存到: {json_path}")

    print(f"\n{'='*60}")
    print("图像生成完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
