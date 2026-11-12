# Prefill Accelerator

Status: experimental

A paired Mac runs the first layers of each long prompt for a model loaded on this Mac, so prefill finishes sooner. The model stays on this Mac: decoding, the KV cache, the prefix cache and the API do not change. The other Mac only holds the layers it runs and gives them back when this Mac unloads the model.

It suits a large model that fits on one Mac with fast memory (good for decoding) while another Mac has faster GPU compute but not enough memory for the whole model, for example an M3 Ultra with an M5 Max over Thunderbolt.

## Supported models

| Family | `model_type` | Notes |
|---|---|---|
| DeepSeek V4.1 | `deepseek_v41` | Engram rows are read on this Mac and sent with each chunk |
| GLM-5.3 Flash | `glm5_next` | Works with a DFlash drafter |
| Qwen3.8 Flash-Next | `qwen4_exp` | PLE rows are read on this Mac and sent with each chunk |
| MiniMax M3 / M3.1 | `minimax_m3_vl` | Works with the DSpark drafter |
| Qwen3.8 27B (Qwen3.5 architecture) | `qwen3_5` | Checkpoints served by the VLM engine. With a context window above the native one, prompts stay on this Mac when the helper holds an attention layer |

Measured on an M3 Ultra (512 GB) with an M5 Max over Thunderbolt 5, for a 64K-token prompt:

| Model | This Mac alone | With an M5 Max 128 GB | With an M5 Max 64 GB |
|---|---|---|---|
| DeepSeek V4.1 Flash oQ3e | 1,060 tok/s | 3,000-3,160 tok/s | 1,470 tok/s |
| GLM-5.3 Flash oQ4e | 540 tok/s | 1,390 tok/s | 740 tok/s |
| Qwen3.8 Flash-Next oQ4e | 1,310 tok/s | 3,670 tok/s | 2,860 tok/s |
| MiniMax M3.1 Flash oQ4e | 300 tok/s | 570 tok/s | 390 tok/s |

The 64 GB column keeps the helper to the layers a 64 GB M5 Max can hold, on the same GPU. A helper holding fewer layers takes a smaller share of each chunk, so models with large layers gain less from a smaller Mac.

DeepSeek V4.1 and Qwen3.8 ran the helper's layers with INT8 Activation Prefill turned on. It changes numerics as it does on a single M5-series Mac. Greedy output matched the helper-off output on the tested prompts for DeepSeek V4.1 (INT8), GLM-5.3, and Qwen3.8 with 16-bit activations on the helper. MiniMax split and whole-model caches were identical on a partial checkpoint with 16-bit activations; MiniMax helper layers use INT8 Activation Prefill too when it is turned on.

## Requirements

On both Macs:

- The same oMLX version. A different version is refused.
- **Global Settings > Advanced > Enable Distributed Inference** turned on, then oMLX restarted. This shows the Cluster tab.
- Local Network access for oMLX allowed when macOS asks (System Settings > Privacy & Security > Local Network).

On the helper Mac:

- **Remote Login** turned on (System Settings > General > Sharing). Pairing sets up the SSH key the helper is started with; the helper's oMLX app does not need to be open.
- Room for the model files of the layers it runs. Files the helper already has in its model folders are used in place; otherwise they are copied to `~/.omlx/prefill-helper/models/<model>` after you confirm.

## Network

The helper sends each chunk's hidden state and cache rows back, up to about 180 MB/s on the tested models.

| Link | Result |
|---|---|
| Thunderbolt | Full speed. No network setup: when the Thunderbolt Bridge has no IPv4 address, its IPv6 link-local address is used. RDMA is not needed. |
| 5 GbE or faster | Full speed. The built-in 10 GbE of a Mac Studio qualifies; a MacBook Pro needs a fast USB or Thunderbolt Ethernet adapter. |
| 2.5 GbE | A large speedup, with less headroom. |
| 1 GbE, Wi-Fi | The link limits the speedup. |

The Cluster tab's Prefill Accelerator checks report which of these the Macs use.

## Setup

1. On both Macs, enable Distributed Inference and restart oMLX, then pair the two Macs in the Cluster tab.
2. After pairing, choose **Prefill Accelerator** and run its checks (SSH, link, version, GPU memory, free disk, power).
3. Open the model's settings, go to **Acceleration**, and choose the Mac under **Prefill Accelerator**.
4. If the helper Mac lacks the model files, they are copied to `~/.omlx/prefill-helper/models` there. The copy starts on its own when at least 50 GB stays free. When deleting older copies there would keep 50 GB free, the row names them and **Delete and copy** removes them first. Otherwise the row shows the space left after copying and **Copy anyway** fills the disk past 50 GB. A model that does not fit at all is refused.
5. When the model is loaded the helper starts, loads its layers and reports **Ready**. Long prompts then show "via <Mac>" in the prefill progress.

**Layers on the helper** is **Auto** by default: each prompt is split by the measured speed of both Macs, and the measurements are kept per model and Mac so a restarted helper splits its first prompt the same way. A fixed number is available for testing.

DeepSeek V4.1 **CED prefill** is on unless turned off in the model's settings. **INT8 Activation Prefill** is off unless turned on there. On a Mac without INT8 tensor units (M4 and earlier), INT8 Activation Prefill applies only to the helper's layers, and changing it restarts the helper without reloading the model.

## Linux helper with NVIDIA GPUs

`omlx-cuda-helper` (`apps/omlx-cuda-helper/`) turns a Linux machine with NVIDIA GPUs into a helper. This Mac uses it like a helper Mac: choose it under **Prefill Accelerator** in the model's settings. Its README covers the build and the options. What differs from a helper Mac:

- **No SSH and no oMLX on the helper.** The helper is a small daemon without a UI. Pair it in the Cluster tab with the 6-digit code it prints in its log; this Mac then drives it over a control channel that the pairing key protects. In the Cluster tab it has no Join button, because it can only be a prefill helper.
- **Compatibility is per model family**, not per oMLX version. The checks list the families the helper serves and name the side to update when they disagree.
- **Two GPUs run as a pipeline.** Each GPU holds a range of layers. GPU memory decides how many layers the helper takes, and with it the speedup.
- **Long prompts drop layers.** When the next chunk's cache would not fit in the helper's GPU memory, the helper drops its last layers one at a time and this Mac runs them from that chunk on. The helper loads them again while no prompt waits.
- **INT8 Activation Prefill** does not apply to the helper's layers.
- **The link matters more.** The checks warn below 5 GbE. **Auto** measures the link and what each layer sends, uses fewer layers on a slow link, and keeps a prompt on this Mac when the transfer alone would take longer than prefilling here.
- **A direct Ethernet cable** to the helper is used when it is faster than the network both machines share. It needs no addresses (the helper is reached at its IPv6 link-local address), but discovery and pairing run on the shared network, so the helper stays on it too.

Setup:

1. On the machine with the NVIDIA GPU, start the helper with Docker. The Cluster tab shows the commands for Linux, Windows and DGX Spark under **Add a PC with an NVIDIA GPU**; building from source is described in the helper's README.
2. In the Cluster tab, press **Pair** on the helper and type the code from its log (`docker logs omlx-helper`). On Windows the helper asks this Mac to pair instead, because Docker Desktop does not pass discovery through.
3. Run the Prefill Accelerator checks (control channel, link, model families, GPU memory, free disk).
4. Choose the helper in the model's settings. Missing model files are copied over the control channel to the helper's models folder.

Measured on an M3 Ultra (512 GB) with two RTX A6000 (48 GB each) over a direct 2.5 GbE cable, for a 32K-token prompt with **Auto**. 1 GbE is an estimate from Auto's speed model, calibrated on measured runs.

| Model | Layers that fit | Layers Auto used | This Mac alone | 2.5 GbE | 1 GbE (estimate, approx.) | Sent per token |
|---|---|---|---|---|---|---|
| Qwen3.8 27B oQ4e | 58 of 64 | 56 | 405 tok/s | 2,079 tok/s (+413%) | 1,025 tok/s (+153%) | 109 KB |
| Qwen3.8 Flash-Next oQ4e | 42 of 48 | 35 | 1,388 tok/s | 3,574 tok/s (+157%) | 1,950 tok/s (+40%) | 65 KB |
| DeepSeek V4.1 Flash oQ3e | 12 of 40 | 12 | 1,119 tok/s | 1,971 tok/s (+76%) | 1,820 tok/s (+63%) | 45 KB |
| DeepSeek V4.1 Flash oQ4e | 10 of 40 | 10 | 1,148 tok/s | 1,761 tok/s (+53%) | 1,740 tok/s (+52%) | 45 KB |
| GLM-5.3 Flash oQ4e | 21 of 45 | 21 | 561 tok/s | 947 tok/s (+69%) | 920 tok/s (+64%) | 60 KB |
| MiniMax M3.1 Flash oQ4e | 22 of 60 | 22 | 327 tok/s | 485 tok/s (+48%) | 475 tok/s (+45%) | 66 KB |

At 2.5 GbE the link was busy for about two thirds of each Qwen3.8 run and peaked at 308 MB/s. For DeepSeek V4.1, GLM-5.3 and MiniMax the layers that stay on this Mac set the speed, because only 10 to 22 of their layers fit in the two GPUs. On a slower link Auto uses fewer layers. It can keep DeepSeek V4.1 on this Mac there: most of the 45 KB each token sends is the boundary between layers (about 42 KB), which does not shrink when the helper takes fewer layers. Auto times a prompt on this Mac alone before it trusts a close call.

### Numerics

A helper Mac runs the same Metal kernels as this Mac. The Linux helper computes in the same 16-bit float type, but CUDA and Metal do not add up every sum in the same order, so each layer matches this Mac's result to rounding noise, not bit for bit. Models with discrete steps (expert routing, sparse attention block selection) can turn that noise into a different continuation.

Measured with greedy decoding on 8 prompts per model (a generated log with an open question, the same log with 20 values to look up, documentation and source code to summarize, each at 8K and 32K tokens), with the helper holding as many layers as fit:

| Model | Helper layers | First-token KL, median (max) | Top token changed | Same 64-token continuation | Lookups right of 40, off / on |
|---|---|---|---|---|---|
| Qwen3.8 27B oQ4e | 63 of 64 | 0.0009 (0.003) | 0 of 8 | 6 of 8 | 40 / 40 |
| GLM-5.3 Flash oQ4e | 21 of 45 | 0.006 (0.03) | 0 of 8 | 0 of 8 | 39 / 40 |
| Qwen3.8 Flash-Next oQ4e | 47 of 48 | 0.010 (0.02) | 0 of 8 | 7 of 8 | 40 / 40 |
| MiniMax M3.1 Flash oQ4e | 21 of 60 | 0.006 (0.14) | 2 of 8 | 3 of 8 | 40 / 40 |
| DeepSeek V4.1 Flash oQ3e | 12 of 40 | 0.035 (0.47) | 2 of 8 | 1 of 8 | 40 / 40 |
| DeepSeek V4.1 Flash oQ4e | 10 of 40 | 0.042 (0.14) | 2 of 8 | 2 of 8 | 40 / 40 |

KL is taken over the first token's 20 most likely tokens. With the helper off, two runs of a prompt are identical. GLM-5.3 answered with reasoning text, which diverges early. The answers to the lookup questions did not get worse in any run.

The same kind of difference appears on this Mac alone when a kernel path changes: with MiniMax M3.1, switching this Mac's prefill attention to its other implementation in the same 21 layers gave a first-token KL of up to 0.11, a different top token in 2 of 8 prompts and the same continuation in 2 of 8. DeepSeek V4.1 moves the most; its expert layers quantize their inputs, which amplifies small differences. Use a helper Mac when the output has to match this Mac's own.

## How it works

Each prompt is cut into a fixed chunk grid when it is admitted. For every chunk the helper runs layers `[0, K)` and sends what crosses the cut (the hidden state, plus the shared state some families need) and the chunk's cache rows for its layers. This Mac runs layers `[K, n)` and appends the helper's rows to its own cache, so after the last chunk this Mac holds the full cache as if it had prefilled alone. The helper computes a couple of chunks ahead, so both Macs work at the same time.

Large lookup tables (DeepSeek V4.1 Engram, Qwen3.8 PLE) stay on this Mac. It reads each chunk's rows and sends them ahead, so the helper holds no table and reads nothing from its SSD.

A benchmark waits up to 10 minutes for the helper to be ready, and an uploaded result names the helper's chip, memory, layers and link (never its name). Prompts shorter than 2,048 tokens and prompts with images prefill here. Other prompts queue for the helper in arrival order. If the helper disconnects or is stopped, the prompt continues here from the last finished chunk.

## On the helper Mac

The dashboard and the Mac app show a banner while the Mac is helping, with a **Stop** button. The host's Cluster tab lists the model copies it made here and can delete them. Models in the helper Mac's own model folders are used in place and never deleted, and neither is the copy the running helper uses. A model load on the helper Mac that no longer fits because of the helper is refused with a message that names it. The helper exits when the host's SSH session ends, so a crashed host never leaves layers loaded.

## Admin API

All endpoints need an admin session and Distributed Inference turned on.

| Endpoint | Description |
|---|---|
| `GET /admin/api/prefill-helper/candidates?model_id=` | Paired Macs and how many layers of the model each can hold |
| `GET /admin/api/prefill-helper/check?node_id=` | Prefill Accelerator checks for one paired Mac |
| `GET /admin/api/prefill-helper/models/{id}` | The model's helper state |
| `POST /admin/api/prefill-helper/models/{id}/{copy\|retry\|stop}` | Confirm a copy prompt, retry, or stop the helper |
| `GET /admin/api/prefill-helper/copies?node_id=`, `POST .../copies/delete?node_id=&name=` | Model copies on a paired Mac, and deleting one |
| `GET /admin/api/prefill-helper/lending`, `POST .../lending/stop` | On the helper Mac: what it runs, and Stop |

The model settings `prefill_helper_node_id` and `prefill_helper_layers` (null for Auto) choose the helper; they are per model and not part of profiles.
