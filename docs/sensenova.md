# Experimental SenseNova Loop text route

Discovery routes `sensenova_loop` to the text engine. Before the standard MLX-LM loader runs, oMLX registers `sensenova_mlx.registered_model` from the separately supplied support package; there is no custom loader hook or outer bridge object. The registered model reuses the support adapter and singleton cache implementation. Strict loading checks keys and shapes, **not dtype**; there is no additional checkpoint validation. Generic generation does not enforce the exact `[248044, 248046]` EOS list: ensure the tokenizer EOS configuration is correct.

Vision tensors remain in the artifact, but generic text-engine handling drops media; image, video, audio and file inputs are not supported. The existing generic tool parser handles declarations and history, but broad compatibility is unestablished. The registered model exposes its type for serial scheduler admission. Prefix/SSD caching follows global settings, and oMLX's reconstructibility guard refuses the singleton cache classes; restoration is not validated. Keep caching disabled globally (`--no-cache`) for first use. External state restoration, ignore_eos, specprefill, speculative/draft, external-engine and distributed paths remain untested; leave them disabled. Default two loop passes and FP32 routing remain unchanged. Set `MLX_ENABLE_TF32=0` **before** starting the process and use short bounded requests; long context is not validated.

## Installation

Use Apple Silicon and Python 3.12: oMLX itself supports Python 3.11+, but this support package requires Python 3.12+. Obtain an oMLX checkout with the SenseNova route, including `docs/sensenova-support-metadata.patch`. The support package must include `sensenova_mlx/registered_model.py`; the published revision below predates that addition, so use a separately supplied updated support package containing it. Obtain the `support/` directory (including its license files and `docs/`) from the [published model repository at revision e5d9e0054ab9f1d2d5b581db9d18b3ff65479851](https://huggingface.co/mlx-community/SenseNova-Loop-9B-mlx-affine8-g64/tree/e5d9e0054ab9f1d2d5b581db9d18b3ff65479851/support). Downloading just that directory does not require downloading weights. Separately obtain the complete model artifact if needed.

The published support metadata conflicts with oMLX's MLX-LM, Transformers and NumPy requirements. Apply the included metadata-only patch to a writable copy of the support directory, then install both packages together normally in the same environment. From the oMLX checkout root, with `SUPPORT_DIR` set to the downloaded directory:

```sh
export SUPPORT_DIR=/path/to/downloaded/support
# Verify before changing the copy (macOS):
shasum -a 256 "$SUPPORT_DIR/pyproject.toml"
# Expected: 4877c47798dbb3a7ee879f45836c7c7318fe28edf5113e0e9b49fcdab33624f
(cd "$SUPPORT_DIR" && patch -p1 --dry-run < "$OLDPWD/docs/sensenova-support-metadata.patch")
(cd "$SUPPORT_DIR" && patch -p1 < "$OLDPWD/docs/sensenova-support-metadata.patch")
shasum -a 256 "$SUPPORT_DIR/pyproject.toml"
# Expected: 4d88f10f762732cf935d24272905b666442feb4842dcebf1cf31c705cdc5bea0
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install . "$SUPPORT_DIR"
export MLX_ENABLE_TF32=0
```

The patch aligns MLX 0.32.4, MLX-LM `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, Transformers `>=5.22.2,<5.23`, NumPy `>=1.24,<2.4`, Pillow 13.0.0 and safetensors 0.8.0. Torch remains declared in the optional `vision` and `reference` extras, which were not tested with this metadata. oMLX still declares its normal dependencies including mlx-audio. To reverse the metadata change in the copy, run `patch -R -p1` with the same patch from the support directory. The public source itself is unchanged.

A selected 110-distribution local closure was installed and checked with Python 3.12, Transformers 5.22.2 and NumPy 2.3.5. Earlier real hello and tool observations used the previous custom loader, not this registered path; they are not evidence for the new path, fresh install, general quality or client compatibility. A new resolver can select different versions; the instructions above are not a lockfile. Cancellation, concurrent queues, prefix restoration and long context remain unverified. Voice execution with this route has not been established.

## Optional tokenizer fixture

With the support package installed, set `SENSENOVA_MODEL_PATH` to a local artifact containing its tokenizer, template and generation config, then run from an oMLX checkout using its test environment:

```sh
SENSENOVA_MODEL_PATH=/path/to/local/artifact python -m unittest discover -s tests -p test_sensenova_tools.py
```

The fixture is skipped without that variable or the optional package. It uses local tokenizer files and synthetic data, not weights or model generation.

oQ conversion of SenseNova is not supported: oQ treats this model type as text-only and would drop the vision weights; our converter retains all vision tensors, and the released artifact does not use oQ.
