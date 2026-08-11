#!/usr/bin/env python3
"""Convert EschaLabs/Qwen3.6-35B-A3B-Escha-W2 (eschamoe/EXL3 trellis) into a
standard mlx-lm `qwen3_5_moe` checkpoint.

Decode spec (bit-exact vs the reference; see docs in omlx/custom_kernels/escha):
  escha_code  I16 [E, in/16, out/16, 16*K]   packed EXL3 trellis codes
  escha_rin/rout F16 [E, in|out]              channel scales OUTSIDE the Hadamard
  W[out,in] = (H128 . Wtilde . H128 * rin[:,None] * rout[None,:]).T
Non-experts: int8 symmetric per-channel (weight_int8 * weight_scale[:,None]),
F16 otherwise; MTP dropped (mlx qwen3_5_moe has no MTP).
Norm weights are stored in the shifted convention (true = stored + 1) and are
baked +1 here, exactly like mlx-lm qwen3_5.sanitize does when MTP is present.

Routed experts can be emitted either as MLX affine-quantized weights
(--expert-format affine, default) or shipped verbatim as trellis codes
(--expert-format trellis) for the decode-on-the-fly Metal kernel path
(omlx.custom_kernels.escha + omlx.patches.escha).

Only the EXL3 funnel unpack runs in numpy (bit-exact, cheap); every heavy op
(decode_3inst MCG hash, tile perm, Hadamard matmuls, scales, quantize) runs in
MLX on the Metal GPU.
"""
import argparse, json, os, struct, sys, time, warnings
import numpy as np
import mlx.core as mx
warnings.filterwarnings("ignore")

MCG_MULT = 0xCBAC1FED
LOP3_AND = 0x8FFF8FFF
LOP3_XOR = 0x3B603B60
HAD_BLOCK = 128
TILE = 16

_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4",
           "I16": "<i2", "I8": "<i1", "U8": "<u1", "U32": "<u4", "BF16": "<u2"}


class Shard:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.data_start = 8 + n
        self._fh = open(path, "rb")

    def get(self, key, index=None):
        meta = self.header[key]
        dt = np.dtype(_DTYPES[meta["dtype"]])
        shape = list(meta["shape"])
        begin, end = meta["data_offsets"]
        if index is not None:
            stride = int(np.prod(shape[1:])) * dt.itemsize
            begin += index * stride
            end = begin + stride
            shape = shape[1:]
        self._fh.seek(self.data_start + begin)
        raw = self._fh.read(end - begin)
        arr = np.frombuffer(raw, dtype=dt).reshape(shape)
        if meta["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def build_reader(src_dir):
    ip = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(ip):
        wmap = json.load(open(ip))["weight_map"]
    else:
        wmap = {}
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".safetensors"):
                with open(os.path.join(src_dir, fn), "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
                wmap.update({k: fn for k in hdr if k != "__metadata__"})
    shards = {}

    def get(key, index_=None):
        fn = wmap[key]
        if fn not in shards:
            shards[fn] = Shard(os.path.join(src_dir, fn))
        return shards[fn].get(key, index_)
    return get


def tensor_core_perm():
    perm = np.empty(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rows):
                perm[t * 8 + j * 4 + i] = r * 16 + c
    return perm


PERM = tensor_core_perm()


def unpack_trellis_np(packed, k):
    """(..., 16*K) int16 -> (..., 256) uint16 codes. exllamav3 pack.cu port."""
    lead = packed.shape[:-1]
    u32 = packed.reshape(-1, 16 * k).view(np.uint32)
    n_words = k * 256 // 32
    t = np.arange(128)
    b0 = t * 2 * k + k - 16 + 256 * k
    b2 = b0 + k + 16
    i0 = b0 // 32
    i1 = (b2 - 1) // 32
    s1 = (i1 + 1) * 32 - b2
    a = u32[:, i0 % n_words].astype(np.uint64)
    b = u32[:, i1 % n_words].astype(np.uint64)
    w1 = (((a << np.uint64(32)) | b) >> s1.astype(np.uint64)).astype(np.uint32)
    w0 = (w1 >> np.uint32(k)) & np.uint32(0xFFFF)
    w1 = w1 & np.uint32(0xFFFF)
    codes = np.empty((u32.shape[0], 256), dtype=np.uint16)
    codes[:, 0::2] = w0
    codes[:, 1::2] = w1
    return codes.reshape(*lead, 256)


def decode_3inst_mx(codes):
    """MCG codebook (decode_3inst<1>): (code*mul)&mask^xor, sum of the two f16
    halves -- computed with exact float math, no bitcasts needed."""
    x = codes.astype(mx.uint32) * mx.array(np.uint32(MCG_MULT))
    x = (x & mx.array(np.uint32(LOP3_AND))) ^ mx.array(np.uint32(LOP3_XOR))
    lo = x & mx.array(np.uint32(0xFFFF))
    hi = (x >> mx.array(np.uint32(16))) & mx.array(np.uint32(0xFFFF))
    vals = []
    for b in (lo, hi):
        bf = b.astype(mx.float32)
        s = mx.floor(bf / 32768.0)
        e = mx.floor((bf % 32768.0) / 1024.0)
        m = bf % 1024.0
        mant = m + mx.where(e > 0, 1024.0, 0.0)
        exp2 = mx.where(e == 0, -24.0, e - 25.0)   # f16 mantissa is fractional
        vals.append((1.0 - 2.0 * s) * mant * mx.power(2.0, exp2))
    return vals[0] + vals[1]


def decode_tiles_mx(code_np, k):
    """code_np [E,tk,tn,16K] int16 -> [E, in, out] float32 on GPU."""
    E, tk, tn = code_np.shape[:3]
    codes_mx = mx.array(unpack_trellis_np(code_np, k))
    vals = decode_3inst_mx(codes_mx.astype(mx.uint32))
    perm = mx.array(np.argsort(PERM).astype(np.int32))  # numpy scatter == gather by inverse perm
    idx = mx.broadcast_to(perm[None, None, None, :], (E, tk, tn, 256))
    tiles = mx.take_along_axis(vals, idx, axis=-1)
    return tiles.reshape(E, tk, tn, 16, 16).transpose(0, 1, 3, 2, 4).reshape(E, tk * 16, tn * 16)


def make_H():
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < HAD_BLOCK:
        h = np.concatenate([np.concatenate([h, h], 1), np.concatenate([h, -h], 1)], 0)
    return h / np.sqrt(HAD_BLOCK)


H128 = mx.array(make_H())


def had_axis_mx(w, axis):
    w = mx.moveaxis(w, axis, -1)
    lead = w.shape[:-1]
    n = w.shape[-1]
    y = w.reshape(*lead, n // HAD_BLOCK, HAD_BLOCK)
    flat = y.reshape(-1, n // HAD_BLOCK, HAD_BLOCK)
    out = mx.matmul(flat, H128)
    return mx.moveaxis(out.reshape(*lead, n), -1, axis)


def decode_expert_proj_mx(code_np, rin_np, rout_np, chunk=64):
    """-> W [E,out,in] float32 (GPU). bit-exact vs the escha reference."""
    E = code_np.shape[0]
    outs = []
    for e0 in range(0, E, chunk):
        c = code_np[e0:e0 + chunk]
        w = decode_tiles_mx(c, c.shape[-1] // 16)
        w = had_axis_mx(had_axis_mx(w, 1), 2)
        ri = mx.array(rin_np[e0:e0 + chunk].astype(np.float32))
        ro = mx.array(rout_np[e0:e0 + chunk].astype(np.float32))
        w = w * ri[:, :, None] * ro[:, None, :]
        outs.append(w.transpose(0, 2, 1))
    return mx.concatenate(outs, axis=0)


def quantize(w, group_size, bits):
    q = mx.quantize(w, group_size=group_size, bits=bits)
    return q[0], q[1], q[2]


def to_bf16(arr):
    return mx.array(np.ascontiguousarray(arr).astype(np.float32)).astype(mx.bfloat16)


def deq_int8(q8, scale):
    return q8.astype(np.float32) * scale[:, None].astype(np.float32)


def pack_q8(w8, scale, group_size=128):
    """Bit-exact affine-Q8 repack of the escha int8 w8a16 contract (lossless).

    q = w8 + 128 (unsigned, one XOR), MLX packs uint32 little-endian == memory
    order. scales/biases are per-output-channel constants (per row), so a
    larger group just stores the constant fewer times. f32 (NOT f16) scales
    make MLX's affine dequant bit-exact: dequant = f32(scale)*q - 128*f32(scale)
    = f32(scale*w8), and because the affine side equals the escha contract the
    w8a16 values are preserved exactly. Mirrors EschaLabs/escha-mlx quant.py.
    """
    w8 = np.ascontiguousarray(w8)
    n, k = w8.shape
    assert k % group_size == 0 and k % 4 == 0, (n, k, group_size)
    q = w8.reshape(n, k).view(np.uint8)     # int8 view; +128 == ^0x80 (2's comp)
    packed = (q ^ np.uint8(0x80)).reshape(n, k).view(np.uint32)  # [N, K/4]
    s32 = scale.astype(np.float32)[:, None]
    ng = k // group_size
    scales = np.repeat(s32, ng, axis=1)                     # f32 [N, ng]
    biases = np.repeat((np.float32(-128.0) * s32), ng, axis=1)
    return packed, scales, biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--gate-bits", type=int, default=8)
    ap.add_argument("--expert-format", choices=("affine", "trellis"), default="affine",
                    help="affine: re-quantize experts with MLX affine (default); "
                         "trellis: ship escha_code/rin/rout for the Metal decode path")
    ap.add_argument("--shard-bytes", type=int, default=2_800_000_000)
    ap.add_argument("--max-layers", type=int, default=40)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--shared-only", action="store_true")
    ap.add_argument("--no-quant", action="store_true",
                    help="emit dense bf16 for the non-expert parts (debug/tests)")
    args = ap.parse_args()
    part_name = None
    if args.layer is not None:
        part_name = f"part-L{args.layer:02d}.safetensors"
    elif args.shared_only:
        part_name = "part-shared.safetensors"

    if args.expert_format == "trellis" and args.no_quant:
        sys.exit("--no-quant and --expert-format trellis are mutually exclusive")

    global out_tensors
    out_tensors = {}
    get = build_reader(args.src)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    def emit(name, arr):
        out_tensors[name] = arr

    def flush(fname):
        if out_tensors:
            mx.save_safetensors(os.path.join(args.out, fname), out_tensors)
            print(f"wrote {fname} ({os.path.getsize(os.path.join(args.out, fname)) / 1e9:.2f} GB)",
                  flush=True)
            out_tensors.clear()

    def emit_q(module, dense_f32, bits=None, gs=None):
        bits = bits or args.bits
        gs = gs or args.group_size
        if args.no_quant:
            emit(f"{module}.weight", to_bf16(dense_f32))
            return
        w, s, b = quantize(to_bf16(dense_f32), gs, bits)
        emit(f"{module}.weight", w)
        emit(f"{module}.scales", s)
        emit(f"{module}.biases", b)

    def emit_copy(module, arr, bare=False):
        emit(module if bare else f"{module}.weight", to_bf16(arr))

    def emit_q8(module, w8, scale, group_size=128):
        """Bit-exact affine-Q8: keeps the escha int8 values exactly (lossless)."""
        packed, scales, biases = pack_q8(w8, scale, group_size)
        emit(f"{module}.weight", mx.array(packed))
        emit(f"{module}.scales", mx.array(scales))
        emit(f"{module}.biases", mx.array(biases))

    def emit_q8_chunked(module, w8, scale, rows=32768, group_size=128):
        if args.no_quant:
            emit(f"{module}.weight", to_bf16(deq_int8(w8, scale)))
            return
        ws, ss, bs = [], [], []
        for i in range(0, w8.shape[0], rows):
            packed, scales, biases = pack_q8(w8[i:i + rows], scale[i:i + rows], group_size)
            ws.append(mx.array(packed)); ss.append(mx.array(scales)); bs.append(mx.array(biases))
        emit(f"{module}.weight", mx.concatenate(ws))
        emit(f"{module}.scales", mx.concatenate(ss))
        emit(f"{module}.biases", mx.concatenate(bs))

    TARGET = "language_model.model"

    if args.shared_only:
        layers = []
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = range(args.max_layers)

    for li, l in enumerate(layers):
        t = time.time()
        pre = f"model.language_model.layers.{l}"
        for proj, (out_, in_) in (("gate_up_proj", (1024, 2048)), ("down_proj", (2048, 512))):
            code = get(f"{pre}.mlp.experts.{proj}.escha_code")
            rin = get(f"{pre}.mlp.experts.{proj}.escha_rin")
            rout = get(f"{pre}.mlp.experts.{proj}.escha_rout")
            if args.expert_format == "trellis":
                # Ship the codes + scales verbatim; the omlx Metal kernel decodes
                # them on the fly (omlx.custom_kernels.escha / omlx.patches.escha).
                base = f"{TARGET}.layers.{l}.mlp.switch_mlp"
                if proj == "gate_up_proj":
                    emit(f"{base}.gate_up_proj.escha_code", mx.array(code))
                    emit(f"{base}.gate_up_proj.escha_rin", mx.array(rin))
                    emit(f"{base}.gate_up_proj.escha_rout", mx.array(rout))
                else:
                    emit(f"{base}.down_proj.escha_code", mx.array(code))
                    emit(f"{base}.down_proj.escha_rin", mx.array(rin))
                    emit(f"{base}.down_proj.escha_rout", mx.array(rout))
                continue
            W = decode_expert_proj_mx(code, rin, rout)
            if proj == "gate_up_proj":
                emit_q(f"{TARGET}.layers.{l}.mlp.switch_mlp.gate_proj", W[:, :512], bits=2)
                emit_q(f"{TARGET}.layers.{l}.mlp.switch_mlp.up_proj", W[:, 512:], bits=2)
            else:
                emit_q(f"{TARGET}.layers.{l}.mlp.switch_mlp.down_proj", W, bits=3)
            del W
        # Router gates stay fp16 (bit-exact to the source; tiny). The shared
        # expert is int8 + per-row scale -> bit-exact affine-Q8 (group 128).
        emit_copy(f"{TARGET}.layers.{l}.mlp.gate", get(f"{pre}.mlp.gate.weight"))
        for nm in ("gate_proj", "up_proj", "down_proj"):
            emit_q8(f"{TARGET}.layers.{l}.mlp.shared_expert.{nm}",
                    get(f"{pre}.mlp.shared_expert.{nm}.weight_int8"),
                    get(f"{pre}.mlp.shared_expert.{nm}.weight_scale"))
        emit_copy(f"{TARGET}.layers.{l}.mlp.shared_expert_gate",
                  get(f"{pre}.mlp.shared_expert_gate.weight"))
        # Qwen3.6 stores RMSNorm weights shifted (true = stored + 1). Bake the
        # shift in, mirroring mlx-lm qwen3_5.sanitize's norm handling.
        for nm in ("input_layernorm", "post_attention_layernorm"):
            emit_copy(f"{TARGET}.layers.{l}.{nm}", get(f"{pre}.{nm}.weight") + 1.0)
        if l % 4 != 3:
            la = f"{TARGET}.layers.{l}.linear_attn"
            sp = f"{pre}.linear_attn"
            emit_copy(f"{la}.norm", get(f"{sp}.norm.weight"))
            emit_copy(f"{la}.A_log", get(f"{sp}.A_log"), bare=True)
            emit_copy(f"{la}.dt_bias", get(f"{sp}.dt_bias"), bare=True)
            emit_copy(f"{la}.conv1d", get(f"{sp}.conv1d.weight").transpose(0, 2, 1))
            for nm in ("in_proj_a", "in_proj_b"):
                emit_copy(f"{la}.{nm}", get(f"{sp}.{nm}.weight"))
            for nm in ("in_proj_qkv", "in_proj_z", "out_proj"):
                emit_q8(f"{la}.{nm}", get(f"{sp}.{nm}.weight_int8"),
                        get(f"{sp}.{nm}.weight_scale"))
        else:
            sa = f"{TARGET}.layers.{l}.self_attn"
            sp = f"{pre}.self_attn"
            for nm in ("q_norm", "k_norm"):
                emit_copy(f"{sa}.{nm}", get(f"{sp}.{nm}.weight") + 1.0)
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                emit_q8(f"{sa}.{nm}", get(f"{sp}.{nm}.weight_int8"),
                        get(f"{sp}.{nm}.weight_scale"))
        print(f"layer {l:02d} done in {time.time() - t:.1f}s", flush=True)
        if args.layer is None and not args.shared_only:
            flush(f"part-L{li:02d}.safetensors")
    if args.layer is not None:
        flush(part_name)
        return

    # ---- shared / head ----
    emit_copy(f"{TARGET}.norm", get("model.language_model.norm.weight") + 1.0)

    emb = get("model.language_model.embed_tokens.weight_int8")
    esc = get("model.language_model.embed_tokens.weight_scale")
    emit_q8_chunked(f"{TARGET}.embed_tokens", emb, esc)
    lm = get("lm_head.weight_int8")
    lsc = get("lm_head.weight_scale")
    emit_q8_chunked("language_model.lm_head", lm, lsc)
    if args.shared_only:
        flush(part_name)
        return

    flush("part-shared.safetensors")
    print(f"weights done in {time.time() - t0:.0f}s; run finalize_escha_mlx.py to assemble")


if __name__ == "__main__":
    main()
