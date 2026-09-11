# SPDX-License-Identifier: Apache-2.0
"""Exercise the oMLX engine with local ParoQuant and DFlash2 checkpoints."""

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

from omlx.engine.dflash import DFlashEngine
from omlx.model_settings import ModelSettings


async def main(args):
    settings = ModelSettings(
        dflash_enabled=True,
        dflash_block_size=5,
        dflash_verify_mode="off",
        dflash_in_memory_cache=True,
    )
    engine = DFlashEngine(
        args.target, args.draft, model_settings=settings, fallback_engine_type="vlm"
    )
    records = []

    async def generate(label, prompt, **kwargs):
        start = time.perf_counter()
        first = None
        final = None
        async for output in engine.stream_generate(
            prompt, max_tokens=32, temperature=0.0, **kwargs
        ):
            if first is None and output.new_text:
                first = time.perf_counter() - start
            final = output
        row = {
            "label": label,
            "seconds": time.perf_counter() - start,
            "ttft": first,
            "output": dataclasses.asdict(final),
        }
        records.append(row)
        Path(args.output).write_text(json.dumps(records, indent=2, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)
        return final

    try:
        start = time.perf_counter()
        await engine.start()
        print("ENGINE_LOADED", time.perf_counter() - start, flush=True)
        if args.vision_smoke_only:
            import base64
            import io

            from PIL import Image

            await generate("before_image", "Continue the count: one, two, three, four,")
            image = io.BytesIO()
            Image.new("RGB", (224, 224), (255, 0, 0)).save(image, format="PNG")
            image_url = (
                "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
            )
            output = await engine.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {
                                "type": "text",
                                "text": "What color fills this image? Answer with only the color name.",
                            },
                        ],
                    }
                ],
                max_tokens=64,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
            row = {
                "label": "image_fallback",
                "text": output.text,
                "in_vlm_fallback": engine._in_fallback_mode,
                "vision_tower_present": hasattr(
                    engine._fallback_engine._vlm_model, "vision_tower"
                ),
            }
            records.append(row)
            Path(args.output).write_text(
                json.dumps(records, indent=2, default=str) + "\n"
            )
            assert "red" in output.text.lower(), output.text
            assert row["in_vlm_fallback"] and row["vision_tower_present"]
            return
        tok = engine.tokenizer
        for length in args.contexts:
            filler = tok.encode(
                "The library holds books on history, science, mathematics, and art.\n"
            )
            tail = tok.encode("\nContinue the count: one, two, three, four,")
            prompt = (filler * ((length // len(filler)) + 1))[
                : length - len(tail)
            ] + tail
            cold = await generate(f"cold_{length}", prompt)
            warm = await generate(f"warm_{length}", prompt)
            assert cold.text == warm.text, "Prefix-cache reuse changed greedy output"
            assert warm.cached_tokens > 0, "Repeated prompt did not reuse its cache"
        # Seed reproducibility checks exercise speculative rejection sampling.
        prompt = tok.encode("The capital of France is")
        outputs = []
        for _ in range(2):
            final = None
            async for output in engine.stream_generate(
                prompt,
                max_tokens=32,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                min_p=0.05,
                seed=123,
            ):
                final = output
            outputs.append(final.text)
        print("SEEDED_EQUAL", outputs[0] == outputs[1], flush=True)
        assert outputs[0] == outputs[1]
        # Close a stream early, then serve another request on the same engine.
        stream = engine.stream_generate(prompt, max_tokens=512, temperature=0.0)
        async for output in stream:
            if output.new_text:
                break
        await stream.aclose()
        after_cancel = await generate("after_cancel", prompt)
    finally:
        await engine.stop()
    # A second start uses a fresh target and must re-arm the restored class hooks.
    try:
        await engine.start()
        after_reload = await generate("after_reload", "The capital of France is")
        assert after_reload.text == after_cancel.text, "Reload changed greedy output"
    finally:
        await engine.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--vision-smoke-only", action="store_true")
    asyncio.run(main(parser.parse_args()))
