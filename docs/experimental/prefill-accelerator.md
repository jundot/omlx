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

Measured on an M3 Ultra (512 GB) with an M5 Max (128 GB) over Thunderbolt 5, for a 64K-token prompt:

| Model | This Mac alone | With the helper |
|---|---|---|
| DeepSeek V4.1 Flash oQ3e | 1,060 tok/s | 3,000-3,160 tok/s |
| GLM-5.3 Flash oQ4e | 540 tok/s | 1,390 tok/s |
| Qwen3.8 Flash-Next oQ4e | 1,310 tok/s | 3,670 tok/s |

DeepSeek V4.1 and Qwen3.8 ran the helper's layers with INT8 Activation Prefill, the default with a helper. It changes numerics as it does on a single M5-series Mac. Greedy output matched the helper-off output on the tested prompts for DeepSeek V4.1 (INT8), GLM-5.3, and Qwen3.8 with 16-bit activations on the helper. MiniMax was checked on a partial checkpoint (split and whole-model caches identical); no full-model speed has been measured.

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
4. If the helper lacks the files, the row shows their size, the destination and the free space; press **Copy files**.
5. When the model is loaded the helper starts, loads its layers and reports **Ready**. Long prompts then show "via <Mac>" in the prefill progress.

**Layers on the helper** is **Auto** by default: each prompt is split by the measured speed of both Macs, and the measurements are kept per model and Mac so a restarted helper splits its first prompt the same way. A fixed number is available for testing.

On a Mac without INT8 tensor units (M4 and earlier), choosing a helper also turns on the model's **INT8 Activation Prefill**, which then applies only to the helper's layers. Turn it off there to keep those layers on 16-bit activations. Changing it reloads the model.

## How it works

Each prompt is cut into a fixed chunk grid when it is admitted. For every chunk the helper runs layers `[0, K)` and sends what crosses the cut (the hidden state, plus the shared state some families need) and the chunk's cache rows for its layers. This Mac runs layers `[K, n)` and appends the helper's rows to its own cache, so after the last chunk this Mac holds the full cache as if it had prefilled alone. The helper computes a couple of chunks ahead, so both Macs work at the same time.

Large lookup tables (DeepSeek V4.1 Engram, Qwen3.8 PLE) stay on this Mac. It reads each chunk's rows and sends them ahead, so the helper holds no table and reads nothing from its SSD.

Prompts shorter than 2,048 tokens and prompts with images prefill here. Other prompts queue for the helper in arrival order. If the helper disconnects or is stopped, the prompt continues here from the last finished chunk.

## On the helper Mac

The dashboard and the Mac app show a banner while the Mac is helping, with a **Stop** button. A model load on the helper Mac that no longer fits because of the helper is refused with a message that names it. The helper exits when the host's SSH session ends, so a crashed host never leaves layers loaded.

## Admin API

All endpoints need an admin session and Distributed Inference turned on.

| Endpoint | Description |
|---|---|
| `GET /admin/api/prefill-helper/candidates?model_id=` | Paired Macs and how many layers of the model each can hold |
| `GET /admin/api/prefill-helper/check?node_id=` | Prefill Accelerator checks for one paired Mac |
| `GET /admin/api/prefill-helper/models/{id}` | The model's helper state |
| `POST /admin/api/prefill-helper/models/{id}/{copy\|retry\|stop}` | Copy files, retry, or stop the helper |
| `GET /admin/api/prefill-helper/lending`, `POST .../lending/stop` | On the helper Mac: what it runs, and Stop |

The model settings `prefill_helper_node_id` and `prefill_helper_layers` (null for Auto) choose the helper; they are per model and not part of profiles.
