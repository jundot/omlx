# omlx-cuda-helper

Status: experimental

A Prefill Accelerator helper for oMLX on Linux with NVIDIA GPUs.

The Mac that runs oMLX keeps the whole model, the KV cache and the API. This helper holds the first layers of a model on its GPUs, runs them for each long prompt and sends the result to the Mac, which runs the remaining layers. The helper has no UI and no settings of its own: oMLX finds it, pairs with it, copies the model files it needs, and starts and stops it.

See `docs/experimental/prefill-accelerator.md` for the feature as a whole.

## Run with Docker

`jundot/omlx-cuda-helper` has images for linux/amd64 (Linux, and Docker Desktop on Windows) and linux/arm64 (DGX Spark); Docker picks the one for the machine. They need an NVIDIA driver of version 580 or later (CUDA 13) and GPU access for containers: the NVIDIA Container Toolkit on Linux (DGX OS has it), the WSL 2 backend on Windows.

Linux and DGX Spark:

```bash
docker run -d --name omlx-helper --restart unless-stopped --gpus all --network host \
  -v omlx-helper:/data jundot/omlx-cuda-helper
docker logs -f omlx-helper
```

Host networking lets the oMLX Cluster tab find the helper. Arguments after the image name go to `serve` (see the options below), for example `--name gpu-box`.

Windows (Docker Desktop):

```powershell
docker run -d --name omlx-helper --restart unless-stopped --gpus all `
  -p 8850-8852:8850-8852 -v omlx-helper:/data jundot/omlx-cuda-helper `
  --name gpu-box --no-mdns --pair MAC_ADDRESS
```

Docker Desktop does not pass discovery through to the container, so the helper asks the Mac to pair (`MAC_ADDRESS` is the Mac's address, with `:PORT` when oMLX does not listen on 8000). Type the code from `docker logs omlx-helper` in the Cluster tab. The Mac then reaches the helper at the Windows machine's address on the published ports, which the Windows firewall must let in.

The volume holds the helper's identity, its pairings, the model files oMLX copies to it and the compiled kernels. To use model folders already on the machine, mount them at `/data/models`. Other commands run in the container, for example `docker exec omlx-helper omlx-cuda-helper --state-dir /data status`.

The amd64 image runs on GPUs from the Ampere generation on (RTX 30, 40 and 50, RTX A and Ada, A100, H100), the arm64 image on GB10. Only 2x RTX A6000 on Linux has been tested so far, not Windows or DGX Spark.

To build the image yourself, run `docker buildx build --platform linux/amd64 -t omlx-cuda-helper .` in this folder (about an hour on 6 cores; MLX is built from source). Build the arm64 image on an arm64 machine, such as a Mac with Apple silicon or the DGX Spark itself: under emulation it takes many hours.

## Requirements

- One or two NVIDIA GPUs. Tested on 2x RTX A6000 (48 GB each), Ubuntu 26.04, driver 595.
- Without Docker: Python 3.11 or later (tested with 3.12) and the patched MLX CUDA wheel built by `mlx-build/` (tested with the CUDA 12.9 toolchain; MLX refuses CUDA 13.1).
- The Mac and the helper on the same network. The link carries each chunk's hidden state and cache rows back to the Mac, so it limits the speedup on short prompts and on models that send a lot per token.
- Optional: a direct Ethernet cable between the Mac and the helper. When it is faster than the shared network, the Mac moves the helper's traffic to it. The cable needs no addresses: without them the Mac reaches the helper at its IPv6 link-local address, and static IPv4 addresses work too. Discovery and pairing still run on the shared network, so keep both machines on it.
- On the Mac: an oMLX version with helper support, with **Global Settings > Advanced > Enable Distributed Inference** turned on.

## Build the MLX wheel

The helper needs MLX with a few CUDA changes (expert gather kernels for affine weights, single rounding in 16-bit products and affine dequantization, a kernel disk cache). `mlx-build/` pins the MLX commit and carries the patches.

```bash
# The toolchain, in a conda environment (see mlx-build/environment.txt).
conda create -p ./env -c conda-forge --file mlx-build/environment.txt
conda activate ./env
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"

git clone https://github.com/ml-explore/mlx.git
mlx-build/apply_patches.sh mlx
mlx-build/build_wheel.sh mlx wheels
pip install wheels/mlx-*.whl
```

The build targets `sm_86` GPUs (RTX 30 series, RTX A6000). Set `MLX_CUDA_ARCHITECTURES` for another generation; only `sm_86` is tested.

A worker refuses to start on an MLX build without these changes.

## Install and run

```bash
pip install .
omlx-cuda-helper serve --name gpu-box --models-dir /data/omlx-models
```

| Option | Default | Meaning |
|---|---|---|
| `--state-dir` | `~/.omlx-cuda-helper` | Identity, pairing keys and runtime files |
| `--models-dir` | `<state-dir>/models` | Where model files are looked up and copied to |
| `--name` | host name | Name shown in the oMLX Cluster tab |
| `--gpus` | all | GPU indices to use, for example `0,1` |
| `--bind` | all addresses | Address to listen on |
| `--http-port` | 8850 | Discovery and pairing invites |
| `--control-port` | 8851 | Control channel |
| `--data-port` | 8852 | Prefill data while a model is in use |
| `--no-mdns` | off | Do not announce on the network |
| `--pair HOST[:PORT]` | none | Ask this oMLX to pair at start, for networks without mDNS |

A model folder you put in the models folder yourself is used in place. oMLX lists and deletes only the copies it made there.

`omlx-cuda-helper status` lists the paired Macs and `omlx-cuda-helper unpair NODE_ID` forgets one.

Compiled CUDA kernels are cached under `<state-dir>/cache`, one folder per MLX build, so the first start of a model after a new build takes longer.

## Pair and use

1. Start the helper. It appears under discovered devices in the Cluster tab of oMLX.
2. Press **Pair** on it. The helper prints a line like `PAIRING CODE 123456` in its log; type the code on the Mac. The code expires after 10 minutes.
3. Run the Prefill Accelerator checks for the helper in the Cluster tab (control channel, link, model families, GPU memory, free disk).
4. In a model's settings, under **Acceleration**, choose the helper as **Prefill Accelerator**. Missing model files are copied to the helper's models folder over the control channel; a stopped copy resumes.
5. When the model is loaded the helper loads its layers and reports **Ready**. It frees its GPU memory when the Mac unloads the model, turns the accelerator off, or goes away.

No SSH is involved: pairing gives both sides a key, and the Mac proves that key on the control channel before the helper accepts a command.

## Supported models

The same families as the Mac helper. A helper and a Mac work together when they agree on the family's contract version, not on the oMLX version.

| Family | `model_type` | Layers that fit on 2x 48 GB |
|---|---|---|
| Qwen3.8 27B (Qwen3.5 architecture) | `qwen3_5` | 58 of 64 |
| Qwen3.8 Flash-Next | `qwen4_exp` | 42 of 48 |
| MiniMax M3 / M3.1 | `minimax_m3_vl` | 22 of 60 |
| GLM-5.3 Flash | `glm5_next` | 21 of 45 |
| DeepSeek V4.1 | `deepseek_v41` | 12 of 40 (10 for oQ4e) |
| Inkling 2 Small | `inkling2` | 10 of 48 |

Layer counts are for the oQ4e checkpoints (oQ3e for DeepSeek V4.1, the published NVFP4 checkpoint for Inkling 2). Inkling 2 oQ conversions below 4 bits keep their experts in affine 2- or 3-bit codes, 4.2 to 5.2 GiB a layer, so more layers fit. How many of them a prompt uses is chosen per prompt from the measured speed of both machines and of the link.

## Two GPUs

Two GPUs run as a pipeline, not as one large GPU: each GPU has its own worker process and holds a contiguous range of layers, and the second worker continues from the first worker's output. With more than two GPUs, the two with the most free memory are used.

## Numerics

The helper computes in the checkpoint's 16-bit float type like the Mac does, but CUDA and Metal do not round every sum in the same order. A helper's output therefore matches the Mac's own output to rounding noise per layer, not bit for bit. Models with discrete steps (expert routing, sparse attention block selection) can amplify that noise into a different but equivalent continuation, the same way they do when the Mac itself takes a different kernel path. See the measurements in `docs/experimental/prefill-accelerator.md`.

## Security

- The pairing exchange uses plain HTTP on the local network, and the helper answers every pairing invite by printing a code. Run the helper only on a network you trust.
- A paired Mac can use the helper's GPUs and write model files under the models folder. It gets no shell and no access outside that folder.

## Limits

- At most two GPUs per model.
- INT8 Activation Prefill is not available on the helper.
- A model's layers must fit in GPU memory: the number of layers the helper can take, and with it the speedup, is set by the GPU memory.
- A long prompt's cache takes GPU memory too. When the next chunk would not fit, the helper drops its last layers one at a time and the Mac runs them from that chunk on. The helper loads them again while no prompt waits. On a GPU that shares the system RAM (DGX Spark) the helper leaves the OS 16 GiB.

## Third-party code

Some CUDA kernels are adapted from other projects. Their sources, licenses and provenance notes are under `omlx_cuda_helper/third_party/`, and the MLX patches taken from another project are marked in `mlx-build/series` with the license in `mlx-build/LICENSE.mlx-serve`.
