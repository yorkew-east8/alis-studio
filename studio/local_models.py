"""User-added local model checkpoints (currently Krea 2 Turbo fine-tunes).

A local model is a .safetensors transformer referenced IN PLACE — no copy, these run 10–15 GB —
registered in "~/Library/Application Support/Alis Studio/local_models.json". Its configuration is
a sidecar JSON with the same basename next to the file (e.g. ``moody…Shot.json``) carrying optional
{"precision", "label", "min_ram"}; anything the sidecar omits is inferred: precision from the
quantization tensors in the safetensors header (must be a recipe krea2-alis-mlx can load:
"8bit", "mixed-4-8" or "bf16"), the label from the file name.
"""

from __future__ import annotations

import json
import os
import struct
import threading

LOCAL_PREFIX = "local:"
PRECISIONS = ("8bit", "mixed-4-8", "bf16")

_LOCK = threading.Lock()   # guards the registry file (UI adds/deletes vs. /api/models reads)


def _registry_path() -> str:
    d = os.path.join(os.path.expanduser("~/Library/Application Support/Alis Studio"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "local_models.json")


def _load() -> list:
    try:
        with open(_registry_path()) as f:
            entries = json.load(f)
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError):
        return []


def _save(entries: list) -> None:
    with open(_registry_path(), "w") as f:
        json.dump(entries, f, indent=1)


def list_models() -> list:
    """All registered entries, each with a live "missing" flag (the file lives outside our control,
    so it can disappear between registrations)."""
    with _LOCK:
        entries = _load()
    out = []
    for e in entries:
        e = dict(e)
        e["missing"] = not os.path.isfile(e.get("path", ""))
        out.append(e)
    return out


def get(name: str):
    """One entry by bare name (no LOCAL_PREFIX), or None. Also returns entries whose file has
    vanished — callers decide whether that's fatal."""
    if not name:
        return None
    for e in list_models():
        if e["name"] == name:
            return e
    return None


def remove(name: str) -> None:
    """Unregister an entry. Never touches the model file itself — it's the user's."""
    with _LOCK:
        entries = [e for e in _load() if e.get("name") != name]
        _save(entries)


def _sanitize_stem(stem: str) -> str:
    return "".join(c for c in stem if c.isalnum() or c in "-_. ").lstrip(". ")


def _read_header(path: str) -> dict:
    """Parse a safetensors header ({tensor_name: {dtype, shape, ...}}) without reading the data."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"{os.path.basename(path)} is too small to be a .safetensors file.")
        (n,) = struct.unpack("<Q", raw)
        if not (2 <= n <= 100 * 1024 * 1024):
            raise ValueError(f"{os.path.basename(path)} doesn't look like a .safetensors file "
                             "(implausible header length).")
        head = f.read(1)
        if head != b"{":
            raise ValueError(f"{os.path.basename(path)} isn't a .safetensors file.")
        try:
            header = json.loads(b"{" + f.read(n - 1))
        except ValueError as e:
            raise ValueError(f"Couldn't parse the .safetensors header of {os.path.basename(path)}: {e}") from None
    header.pop("__metadata__", None)
    return header


def _infer_precision(header: dict) -> str:
    """Which krea2-alis-mlx recipe this file was quantized with, from the packed-tensor geometry.

    An MLX-quantized Linear stores a uint32-packed "weight" plus "scales" of shape (out, in/64)
    (group_size 64). Bits per value = 32 · packed / (out·in): a 4-bit pack gives 8 values per
    uint32, 8-bit gives 4. Only the mixed-4/8 recipe ships 4-bit bulk, so any 4-bit tensor means
    "mixed-4-8"; all-8-bit means "8bit"; no quantized tensors at all means "bf16"."""
    bits_seen = set()
    for name, t in header.items():
        if not name.endswith(".scales"):
            continue
        w = header.get(name[: -len("scales")] + "weight")
        if not w or w.get("dtype") != "U32":
            continue
        shape = t.get("shape") or []
        wshape = w.get("shape") or []
        if len(shape) < 2 or len(wshape) < 2:
            continue
        out, groups = shape[0], shape[-1]
        total = out * groups * 64                 # out × in, in = groups · group_size(64)
        packed = wshape[0] * wshape[-1]           # uint32 words: 32/bits values each
        bits = 32.0 * packed / total if total else 0
        if abs(bits - round(bits)) < 0.1 and round(bits) in (4, 8):
            bits_seen.add(round(bits))
    if 4 in bits_seen:
        return "mixed-4-8"
    if 8 in bits_seen:
        return "8bit"
    return "bf16"


def _is_row_scale_repackage(header: dict) -> bool:
    """The ComfyUI/community 8-bit repackage: int8 weights with a per-row F32 weight_scale (and no
    MLX-packed group tensors). Structurally sound but NOT loadable by krea2-alis-mlx — point the
    user at scripts/convert_krea2_official.py."""
    has_mlx_pack = any(k.endswith(".scales") for k in header)
    if has_mlx_pack:
        return False
    i8 = {k for k, t in header.items() if t.get("dtype") == "I8"}
    return bool(i8) and any(k[: -len(".weight")] + ".weight_scale" in header for k in i8)


def _sidecar(path: str) -> dict:
    """Optional same-named JSON next to the model file: {"precision","label","min_ram"}."""
    p = os.path.splitext(path)[0] + ".json"
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as f:
            cfg = json.load(f)
    except ValueError as e:
        raise ValueError(f"Couldn't parse the sidecar config {os.path.basename(p)}: {e}") from None
    if not isinstance(cfg, dict):
        raise ValueError(f"Sidecar config {os.path.basename(p)} must be a JSON object.")
    return cfg


def add(path: str) -> dict:
    """Register a local .safetensors transformer. Returns the new entry; raises ValueError with a
    user-facing message when the file, its header, or the sidecar config is unusable."""
    src = os.path.expanduser((path or "").strip())
    if not src or not os.path.isfile(src) or not src.endswith(".safetensors"):
        raise ValueError("Not a .safetensors file: " + str(path or "(empty)"))
    stem = _sanitize_stem(os.path.splitext(os.path.basename(src))[0])
    if not stem:
        raise ValueError(f"Couldn't derive a model name from {os.path.basename(src)!r} — rename the file.")

    side = _sidecar(src)
    precision = side.get("precision")
    if precision is not None and precision not in PRECISIONS:
        raise ValueError(f"Sidecar config precision must be one of {', '.join(PRECISIONS)} "
                         f"(got {precision!r}).")
    label = side.get("label")
    if label is not None and not str(label).strip():
        raise ValueError("Sidecar config 'label' must be a non-empty string.")
    min_ram = side.get("min_ram")
    if min_ram is not None and (not isinstance(min_ram, (int, float)) or min_ram <= 0):
        raise ValueError("Sidecar config 'min_ram' must be a positive number (GiB).")

    header = _read_header(src)          # validates the file before we register it
    if _is_row_scale_repackage(header):
        raise ValueError(
            f"{os.path.basename(src)} is a ComfyUI-style 8-bit repackage (int8 weights + per-row "
            "weight_scale) — krea2-alis-mlx can't load it directly. Convert it once, then add the "
            f"converted file:\n  python scripts/convert_krea2_official.py {src}")
    if precision is None:
        precision = _infer_precision(header)

    with _LOCK:
        entries = _load()
        if any(e.get("name") == stem for e in entries):
            raise ValueError(f"'{stem}' is already registered — delete it first to re-add.")
        if any(e.get("path") == src for e in entries):
            raise ValueError(f"That file is already registered as "
                             f"'{next(e['name'] for e in entries if e.get('path') == src)}'.")
        entry = {"backend": "krea2-turbo", "name": stem, "path": src, "precision": precision,
                 "label": str(label or stem),
                 "min_ram": int(min_ram) if min_ram else 0,
                 "size_gb": round(os.path.getsize(src) / 1e9, 1)}
        entries.append(entry)
        _save(entries)
    return entry
