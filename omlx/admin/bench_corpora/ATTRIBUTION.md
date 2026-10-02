# Benchmark corpus attribution

The files in this directory are bundled only as deterministic throughput
benchmark inputs. Runtime code never downloads corpus content.

- `code_python.txt` is a snapshot of production Python source from oMLX,
  licensed under Apache-2.0.
- `code_mixed.txt` is a snapshot of Python, Swift, JavaScript/Jinja, C++, and
  Metal source from oMLX, licensed under Apache-2.0.
- `novel_ko.txt` contains the body text of the 33 works in the
  [KNoTE dataset](https://github.com/AKS-DHLAB/KNoTE), pinned to commit
  `add4f9dd99db7e322018d9993c86aadd8e8f4335`. KNoTE is licensed under
  CC BY 4.0 and is maintained by the Digital Humanities Lab at the Academy of
  Korean Studies.
- `novel_en.txt` is Project Gutenberg eBook #2701, *Moby-Dick; or, The Whale*
  by Herman Melville, with Project Gutenberg boilerplate excluded at runtime.
- `novel_ja.txt` contains cleaned text from Natsume Soseki's *Kokoro* and
  *I Am a Cat*, distributed by [Aozora Bunko](https://www.aozora.gr.jp/).
  Both works are public domain.

`manifest.json` records source details, byte sizes, SHA-256 hashes, and token
counts measured with representative Qwen3.6 and DeepSeek V4 tokenizers.
