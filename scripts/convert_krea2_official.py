"""Convert a ComfyUI-native INT8 ConvRot Krea 2 Turbo checkpoint to the MLX build.

Input: the "int8_convrot" repackage (e.g. a Civitai "moodyAmateurMix…" file produced by the
StarNodes converter) — I8 weights with a per-output-row F32 "weight_scale", all under a
"model.diffusion_model." prefix. Two things make it unloadable by krea2-alis-mlx directly:

  1. The per-token block linears are stored ROTATED: W_rot = W @ H^T with a block-diagonal
     "regular" Hadamard (kron powers of the 4x4 kernel [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],
     [-1,1,1,1]], normalized 1/sqrt(n)), groupwise along the input dim (convrot_groupsize=256,
     from the file's own _quantization_metadata). At runtime ComfyUI folds the inverse into the
     activation path; a plain MLX loader would feed rotated weights to an unrotated pipeline —
     every layer computes garbage and generation returns pure noise.
  2. The quantization layout (int8 + row scale) isn't MLX's packed group-64 format.

This script dequantizes, UN-ROTATES the convrot layers with their Hadamard (verified against
the official baseline: cosine > 0.99), and re-quantizes to the MLX 8-bit build layout
(precision "8bit") that krea2-alis-mlx loads. Non-convrot layers are dequantized to bf16.
Key names are the file's with the prefix stripped, which matches SingleStreamDiT 1:1.

Usage:  python scripts/convert_krea2_official.py ~/Downloads/moody….safetensors
Writes  <stem>.mlx8bit.safetensors next to the input. Peak RAM stays ~1 GB (tensors are
memmapped and processed one at a time; the output is streamed, not materialized).
"""

from __future__ import annotations

import json
import math
import re
import struct
import sys

import numpy as np

PREFIX = "model.diffusion_model."
GROUP, BITS = 64, 8
BULK = re.compile(r"^blocks\.\d+\.(attn|mlp)\.")   # what krea2's quantize_bulk quantizes
NP_DTYPE = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "U16": np.uint16,
            "I32": np.int32, "U32": np.uint32, "I64": np.int64,
            "F16": np.float16, "F32": np.float32, "F64": np.float64}
WIDTH = {"U32": 4, "I8": 1, "I64": 8, "BF16": 2}


def read_header(path: str):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n   # header dict, byte offset where tensor data starts


def bf16_bytes(a: np.ndarray) -> bytes:
    """float32 → bfloat16 bytes, round-to-nearest-even (matches MLX's cast)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).ravel()
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).tobytes()


def _slice(mm, header, key: str) -> np.ndarray:
    t = header[key]
    s, e = t["data_offsets"]
    return np.frombuffer(mm[s:e].tobytes(), dtype=NP_DTYPE[t["dtype"]])


def hadamard(n: int) -> np.ndarray:
    """comfy-kitchen's "regular" Hadamard: kron powers of the 4x4 kernel, normalized 1/sqrt(n).
    Symmetric, so W = _rotate_weight(W_rot) undoes the quantizer's rotation."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    h = h4
    while h.shape[0] < n:
        h = np.kron(h, h4)
    return h / np.sqrt(h.shape[0])


def unrotate(w: np.ndarray, group_size: int) -> np.ndarray:
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"in_features {in_dim} not divisible by convrot_groupsize {group_size}")
    h = hadamard(group_size)
    return (w.reshape(out_dim, in_dim // group_size, group_size) @ h.T).reshape(out_dim, in_dim)


def convert(src: str, dst: str | None = None, quiet: bool = False) -> str:
    import mlx.core as mx

    if dst is None:
        dst = re.sub(r"\.safetensors$", "", src) + ".mlx8bit.safetensors"
    header, data_base = read_header(src)
    qmeta = json.loads(header["__metadata__"]["_quantization_metadata"])["layers"] \
        if "_quantization_metadata" in header.get("__metadata__", {}) else {}
    header.pop("__metadata__", None)

    tensors = {}   # file keys with the prefix stripped, minus the per-row scales
    scales = {}    # "X.weight" -> "X.weight_scale" tensor meta
    for k, t in header.items():
        if not k.startswith(PREFIX):
            continue
        name = k[len(PREFIX):]
        if name.endswith(".weight_scale"):
            scales[name[:-len(".weight_scale")] + ".weight"] = t
        else:
            tensors[name] = t
    if not scales:
        raise SystemExit("No per-row weight_scale quantization found — this file isn't the "
                         "ComfyUI I8 format; add it directly instead.")

    def tensor(name):
        t = tensors[name]
        s, e = t["data_offsets"]
        return np.frombuffer(mm[data_base + s:data_base + e].tobytes(),
                             dtype=NP_DTYPE[t["dtype"]]).reshape(t["shape"])

    def scale_of(weight_name):
        t = scales[weight_name]
        s, e = t["data_offsets"]
        return np.frombuffer(mm[data_base + s:data_base + e].tobytes(),
                             dtype=np.float32).reshape(t["shape"][0], 1)

    # plan every output tensor (dtype/shape) so the header can be written before the data
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    plan = {}   # out_name -> (dtype, shape, kind); kind: q=requantize, d=dequant→bf16, c=cast→bf16
    for name, t in tensors.items():
        shape = t["shape"]
        if t["dtype"] == "I8" and name.endswith(".weight") and len(shape) == 2 and BULK.match(name):
            mod = name[: -len(".weight")]   # MLX quantized params: "X.weight" + "X.scales" + "X.biases"
            plan[name] = ("U32", [shape[0], math.ceil(shape[1] * BITS / 32)], "q")
            plan[mod + ".scales"] = ("BF16", [shape[0], shape[1] // GROUP], "s")
            plan[mod + ".biases"] = ("BF16", [shape[0], shape[1] // GROUP], "s")
        else:
            plan[name] = ("BF16", shape, "d" if t["dtype"] == "I8" else "c")

    off = 0
    st_header = {}
    for name, (dt, shape, _) in plan.items():
        nbytes = math.prod(shape) * WIDTH[dt]
        st_header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    head = json.dumps({"__metadata__": {"format": "pt", "source": "i8-row-scale repackage"},
                       **st_header}, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)

    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for i, (name, (dt, shape, kind)) in enumerate(plan.items()):
            if kind == "s":     # scales/biases ride along with their quantized weight (the "q" row)
                continue
            if kind == "q":     # dequantize I8×row-scale, undo ConvRot, re-quantize as MLX group-64 8-bit
                w = mx.array(tensor(name).astype(np.float32) * scale_of(name))
                # metadata keys are MODULE names ("…attn.wq"), ours carry ".weight"
                entry = qmeta.get(PREFIX + name.removesuffix(".weight")) or {}
                g = entry.get("convrot_groupsize")
                if entry.get("convrot") and g:
                    w = mx.array(unrotate(np.asarray(w.astype(mx.float32)), g))
                wq, gscales, gbiases = mx.quantize(w, group_size=GROUP, bits=BITS)
                f.write(np.asarray(wq, dtype=np.uint32).tobytes())
                f.write(bf16_bytes(np.asarray(gscales)))
                f.write(bf16_bytes(np.asarray(gbiases)))
            elif kind == "d":
                f.write(bf16_bytes(tensor(name).astype(np.float32) * scale_of(name)))
            else:
                f.write(bf16_bytes(tensor(name)))
            if not quiet and (i + 1) % 32 == 0:
                print(f"\r  {st_header[name]['data_offsets'][1] / 1e9:.1f} / {off / 1e9:.1f} GB",
                      end="", flush=True)
    if not quiet:
        print(f"\r  wrote {dst} ({off / 1e9:.1f} GB)".ljust(60))
    return dst


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    convert(sys.argv[1])
