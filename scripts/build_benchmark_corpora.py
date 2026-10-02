#!/usr/bin/env python3
"""Build the bundled throughput-benchmark context corpora.

This script is a maintainer tool, not part of the runtime. It creates stable
UTF-8 snapshots from oMLX source and pinned public literary sources.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "omlx" / "admin" / "bench_corpora"
KNOTE_COMMIT = "add4f9dd99db7e322018d9993c86aadd8e8f4335"
KNOTE_TREE_URL = (
    "https://api.github.com/repos/AKS-DHLAB/KNoTE/git/trees/"
    f"{KNOTE_COMMIT}?recursive=1"
)
MOBY_DICK_URL = (
    "https://raw.githubusercontent.com/jundot/omlx/"
    "34e9e985b753509e0172d31b99331419436e575f/"
    "omlx/admin/bench_corpus.txt"
)
AOZORA_WORKS = (
    (
        "Kokoro",
        "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip",
    ),
    (
        "I Am a Cat",
        "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
    ),
)
REPRESENTATIVE_TOKEN_COUNTS = {
    "Qwen3.6": {
        "code_python.txt": 340883,
        "code_mixed.txt": 410409,
        "novel_ko.txt": 441225,
        "novel_en.txt": 311730,
        "novel_ja.txt": 297031,
    },
    "DeepSeek-V4": {
        "code_python.txt": 338047,
        "code_mixed.txt": 402092,
        "novel_ko.txt": 504071,
        "novel_en.txt": 309112,
        "novel_ja.txt": 351506,
    },
}


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "omlx-corpus-builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _source_stream(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"\n\n# ===== FILE: {relative} =====\n\n{text}")
    return "".join(parts)


def _python_corpus() -> str:
    paths = sorted((REPO_ROOT / "omlx").rglob("*.py"))
    stream = _source_stream(paths)
    # Around 350K tokens in Qwen3.6 while avoiding unnecessary wheel growth.
    return stream[:1_500_000]


def _mixed_corpus() -> str:
    buckets = [
        sorted((REPO_ROOT / "omlx").rglob("*.py")),
        sorted((REPO_ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift")),
        sorted((REPO_ROOT / "omlx" / "admin" / "static" / "js").rglob("*.js"))
        + sorted((REPO_ROOT / "omlx" / "admin" / "templates").rglob("*.html")),
        sorted((REPO_ROOT / "omlx" / "custom_kernels").rglob("*.cpp"))
        + sorted((REPO_ROOT / "omlx" / "custom_kernels").rglob("*.h"))
        + sorted((REPO_ROOT / "omlx" / "custom_kernels").rglob("*.metal")),
    ]
    streams = [_source_stream(paths) for paths in buckets]
    chunk_size = 4096
    offsets = [0] * len(streams)
    parts: list[str] = []
    total = 0
    target = 1_600_000
    while total < target:
        progressed = False
        for index, stream in enumerate(streams):
            start = offsets[index]
            if start >= len(stream):
                continue
            chunk = stream[start : start + chunk_size]
            offsets[index] += len(chunk)
            parts.append(chunk)
            total += len(chunk)
            progressed = True
            if total >= target:
                break
        if not progressed:
            break
    return "".join(parts)


def _knote_corpus() -> tuple[str, list[str]]:
    tree = json.loads(_read_url(KNOTE_TREE_URL))
    paths = sorted(
        item["path"]
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").startswith("dataset/")
        and item.get("path", "").endswith(".xml")
    )
    if len(paths) != 33:
        raise RuntimeError(f"Expected 33 KNoTE works, found {len(paths)}")

    works: list[str] = []
    for path in paths:
        quoted = urllib.parse.quote(path)
        url = (
            "https://raw.githubusercontent.com/AKS-DHLAB/KNoTE/"
            f"{KNOTE_COMMIT}/{quoted}"
        )
        root = ET.fromstring(_read_url(url))
        body = root.find(".//{*}body")
        if body is None:
            raise RuntimeError(f"KNoTE work has no TEI body: {path}")
        paragraphs = []
        for paragraph in body.findall(".//{*}p"):
            text = "".join(paragraph.itertext()).strip()
            if text:
                paragraphs.append(text)
        works.append(f"\n\n===== {Path(path).stem} =====\n\n" + "\n\n".join(paragraphs))
    return "".join(works).strip() + "\n", paths


def _clean_aozora_text(raw: bytes) -> str:
    text = raw.decode("shift_jis")
    sections = re.split(r"-{20,}", text)
    if len(sections) >= 4:
        text = "".join(sections[2:-1])
    text = re.sub(r"《[^》]*》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = text.replace("｜", "")
    return text.strip()


def _aozora_corpus() -> str:
    works: list[str] = []
    for title, url in AOZORA_WORKS:
        with zipfile.ZipFile(io.BytesIO(_read_url(url))) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if len(names) != 1:
                raise RuntimeError(f"Unexpected Aozora archive layout for {title}")
            text = _clean_aozora_text(archive.read(names[0]))
        works.append(f"\n\n===== {title} =====\n\n{text}")
    return "".join(works).strip() + "\n"


def _write_text(name: str, text: str) -> dict[str, object]:
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"
    path = OUTPUT_DIR / name
    path.write_text(text, encoding="utf-8")
    data = path.read_bytes()
    return {
        "file": name,
        "bytes": len(data),
        "characters": len(text),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    korean, knote_paths = _knote_corpus()
    entries = [
        _write_text("code_python.txt", _python_corpus()),
        _write_text("code_mixed.txt", _mixed_corpus()),
        _write_text("novel_ko.txt", korean),
        _write_text("novel_en.txt", _read_url(MOBY_DICK_URL).decode("utf-8")),
        _write_text("novel_ja.txt", _aozora_corpus()),
    ]

    manifest = {
        "format": 1,
        "corpora": entries,
        "representative_token_counts": REPRESENTATIVE_TOKEN_COUNTS,
        "sources": {
            "code_python": "oMLX production Python source snapshot",
            "code_mixed": "oMLX Python, Swift, JavaScript/Jinja, C++/Metal source snapshot",
            "novel_ko": {
                "repository": "https://github.com/AKS-DHLAB/KNoTE",
                "commit": KNOTE_COMMIT,
                "works": knote_paths,
            },
            "novel_en": MOBY_DICK_URL,
            "novel_ja": [url for _, url in AOZORA_WORKS],
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
