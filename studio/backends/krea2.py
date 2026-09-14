"""Krea 2 Turbo backend — wraps the `krea2-alis-mlx` package (the pure-MLX port).

Weights are pulled from Hugging Face on first use (cached under ~/.cache/krea2_alis_mlx);
drop a transformer_*.safetensors in the working directory to use a local copy instead.
Users can also register arbitrary local checkpoints (fine-tunes) through the model manager —
see studio/local_models.py; those load straight from their own path with the precision their
file was quantized with.
"""

from __future__ import annotations

import json
import os
import struct

from .. import local_models
from .base import Backend
from .mflux_common import _img2img_args, _img2img_params, _lora_args, _lora_params

_LOCAL = local_models.LOCAL_PREFIX


# --- LoKr (ai-toolkit kron) LoRAs --------------------------------------------
# The krea2-alis-mlx LoRA loader only understands rank-decomposed adapters
# (lora_A/B, lora_down/up) and raises on anything else. Civitai is full of
# ai-toolkit LoKr files — weights stored as a Kronecker product lokr_w1 ⊗ lokr_w2
# (typical layout here: w1 (4,4), w2 = the layer's (out/4, in/4) block) — so the
# backend applies those itself, mirroring the package's wrap/unwrap design. The
# delta is applied factorized (never materializing out×in): with x viewed as
# (in_m, in_n), kron(w1, w2)·x costs out·in/4 FLOPs — a quarter of a dense
# delta — and it is exact. Truncating a trained LoKr to a low-rank A/B pair is
# not an option (measured on a 24k-step file: rank 256 keeps ~55% of the delta
# energy — the spectra are flat; these are full finetunes in LoKr clothing).
# Scaling: for direct w1/w2 files both ai-toolkit (runtime_scale forced to 1
# when both factors are full matrices) and ComfyUI's calculate_weight (alpha
# ignored when no _a/_b rebuild happened) apply kron(w1, w2) at scale 1.0 — the
# file's per-layer .alpha buffer (often junk, e.g. 1e10) is deliberately not read.

_LOKR_CLS = None


def _lokr_cls():
    """The LoKr wrapper class, built on first use — keeps mlx imports out of module import."""
    global _LOKR_CLS
    if _LOKR_CLS is None:
        from mlx import nn

        class _LoKrLinear(nn.Module):
            """Wraps a (quantized) linear: y = base(x) + Σ scale_i·kron(w1_i, w2_i)·x.

            Multiple LoKr files on one layer are just more branches (stacking, like the
            package's LoRALinear holds multiple A/B pairs)."""

            def __init__(self, base, branches):
                super().__init__()
                self.base = base
                self.branches = list(branches)  # each (w1 (out_l, in_m), w2 (out_k, in_n), scale)

            def __call__(self, x):
                y = self.base(x)
                for w1, w2, s in self.branches:
                    # kron row/col order is (p, o) and (q, s) major — reshape into factor
                    # blocks, contract w2 on the input side, w1 across the out blocks.
                    h = x.reshape(*x.shape[:-1], w1.shape[1], w2.shape[1])  # (…, in_m, in_n)
                    h = (h @ w2.T).swapaxes(-1, -2)      # (…, out_k, in_m)
                    h = h @ w1.T                         # (…, out_k, out_l)
                    delta = h.swapaxes(-1, -2).reshape(*x.shape[:-1], -1)  # p-major = kron rows
                    y = y + (s * delta).astype(y.dtype)
                return y

        _LOKR_CLS = _LoKrLinear
    return _LOKR_CLS


def _is_lokr(path: str) -> bool:
    """True if the safetensors file carries lokr_w1 tensors (header scan only — no weights load)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= n <= 100 * 1024 * 1024:
                return False
            header = json.loads(f.read(n))
        return any(k.endswith(".lokr_w1") for k in header if k != "__metadata__")
    except (OSError, ValueError):
        return False


def _apply_lokrs(transformer, specs) -> list:
    """Wrap transformer linears with LoKr branches for [(path, scale), …].

    Two-phase like krea2.lora.apply_loras: every target is resolved and shape-checked
    before the first swap, so a bad file leaves the tree untouched. Returns the wrapped
    paths (feed back to _unload_lokrs)."""
    import mlx.core as mx
    from mlx import nn
    from krea2.lora import _get_submodule, _linear_dims, _set_submodule, _to_mlx_path

    cls = _lokr_cls()
    grouped: dict = {}   # target path → [(w1, w2, scale), …]
    for path, scale in specs:
        weights = mx.load(path)
        w1s, w2s = {}, {}
        for k in weights:
            if k.endswith((".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2")):
                raise ValueError(
                    f"{os.path.basename(path)} stores its LoKr factors low-rank (_a/_b tensors) — "
                    "this app applies direct w1/w2 LoKr files only.")
            if k.endswith(".lokr_w1"):
                w1s[_to_mlx_path(k[: -len(".lokr_w1")])] = weights[k]
            elif k.endswith(".lokr_w2"):
                w2s[_to_mlx_path(k[: -len(".lokr_w2")])] = weights[k]
        missing = set(w1s) ^ set(w2s)
        if missing:
            raise ValueError(f"{os.path.basename(path)} has unpaired lokr_w1/w2 tensors for: "
                             f"{sorted(missing)[:5]}")
        for tgt, w2 in w2s.items():
            grouped.setdefault(tgt, []).append((w1s[tgt], w2, float(scale)))

    # phase 1: resolve and validate every target — nothing mutated yet
    resolved: dict = {}
    for tgt, branches in grouped.items():
        try:
            cur = _get_submodule(transformer, tgt)
        except (AttributeError, IndexError, KeyError, ValueError) as e:
            raise ValueError(f"LoKr LoRA targets '{tgt}' which doesn't exist in the model tree") from e
        while isinstance(cur, cls):   # never nest our own wrappers
            cur = cur.base
        # the package may have wrapped a plain LoRA here — branch off it; dims come from its base
        inner = cur.base if type(cur).__name__ == "LoRALinear" else cur
        if not isinstance(inner, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(f"LoKr LoRA target '{tgt}' is {type(cur).__name__}, not a linear layer")
        out_f, in_f = _linear_dims(inner)   # catch a wrong-base LoKr here, not as a matmul error mid-run
        for w1, w2, _ in branches:
            if w1.shape[0] * w2.shape[0] != out_f or w1.shape[1] * w2.shape[1] != in_f:
                raise ValueError(
                    f"LoKr factors {list(w1.shape)} ⊗ {list(w2.shape)} don't tile the "
                    f"({out_f}, {in_f}) layer at '{tgt}' — was this LoKr built for Krea 2? "
                    "One made for a different model won't fit.")
        resolved[tgt] = cur

    # phase 2: swap the wrappers in — no raise conditions below
    for tgt, branches in grouped.items():
        bf = [(w1.astype(mx.bfloat16), w2.astype(mx.bfloat16), s) for w1, w2, s in branches]
        _set_submodule(transformer, tgt, cls(resolved[tgt], bf))
    mx.eval(transformer.parameters())
    return list(grouped)


def _unload_lokrs(transformer, paths) -> int:
    """Restore the given wrapped paths to their pre-LoKr modules. Returns how many were unwrapped."""
    import mlx.core as mx
    from krea2.lora import _get_submodule, _set_submodule

    cls = _lokr_cls()
    removed = 0
    for tgt in paths or []:
        try:
            cur = _get_submodule(transformer, tgt)
        except (AttributeError, IndexError, KeyError, ValueError):
            continue
        while isinstance(cur, cls):
            cur = cur.base
            removed += 1
        _set_submodule(transformer, tgt, cur)
    if removed:
        mx.eval(transformer.parameters())
    return removed


class Krea2Backend(Backend):
    min_ram_gib = 24   # 12.9B DiT; mixed-4/8 ~9.8 GB + activations → wants ≥24 GB
    id = "krea2-turbo"
    label = "Krea 2 Turbo"
    prompt_note = "Understands Korean and other languages natively (Qwen3 text encoder)."
    variants = [
        {"id": "8bit", "label": "8-bit · best quality"},
        {"id": "mixed-4-8", "label": "mixed-4/8 · smaller"},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280, 1536, 2048], "default_size": 1024,   # Krea 2 Turbo is a native 1K–2K model
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 2048, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 50, "default": 8},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output",
         "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 0, "max": 10, "step": 0.5, "default": 0, "fixed": True,
         "hint": "Krea 2 Turbo is distilled — keep guidance at 0."},
        {"key": "sampler", "label": "Sampler", "type": "select", "group": "Sampling",
         "options": [{"value": "euler", "label": "Euler (flow-match)"}], "default": "euler", "fixed": True},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Distilled Turbo runs without guidance, so a negative prompt has no effect."},
        *_img2img_params(),   # Input image + Strength — krea2-alis-mlx >= 0.2 does rectified-flow img2img
        *_lora_params(),      # LoRA library — krea2-alis-mlx >= 0.3 applies runtime low-rank adapters
    ]
    catalog = [
        {"variant": "8bit", "label": "8-bit · best quality", "size_gb": 14.2, "note": "near-lossless"},
        {"variant": "mixed-4-8", "label": "mixed-4/8 · smaller", "size_gb": 9.8, "note": "smallest near-lossless"},
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import krea2.pipeline  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._pipe = None
        self._variant = None
        self._lokr_sig = ()      # currently-applied LoKr (path, scale) set, to skip redundant re-wraps
        self._lokr_paths = []    # wrapped target paths, for a clean unload
        self._lokr_pipe = None   # the pipe the wrappers live in (a reload orphans them)

    def will_load(self, variant: str) -> bool:
        return self._pipe is None or self._variant != variant   # mirrors the reload check in _get

    # --- user-registered local checkpoints (studio/local_models.py) ---
    def _local_entry(self, variant: str):
        """The registry entry for a 'local:<name>' variant id, or None for built-in variants."""
        return local_models.get(variant[len(_LOCAL):]) if variant.startswith(_LOCAL) else None

    def extra_variants(self) -> list:
        return [{"id": _LOCAL + e["name"], "label": e["label"],
                 "min_ram": e.get("min_ram") or 0} for e in local_models.list_models()]

    def catalog_entries(self) -> list:
        out = list(self.catalog)
        for e in local_models.list_models():
            out.append({"variant": _LOCAL + e["name"], "label": e["label"],
                        "size_gb": e.get("size_gb"), "local": True,
                        "note": ("file missing" if e["missing"] else "local · " + e["precision"]),
                        "installed": not e["missing"]})
        return out

    def _get(self, variant: str):
        import gc
        import mlx.core as mx
        from krea2.pipeline import Krea2Pipeline, resolve_weights

        if self._pipe is None or self._variant != variant:
            entry = self._local_entry(variant)
            if entry is not None:
                if entry["missing"]:
                    raise ValueError(f"The local model file is gone: {entry['path']} — restore it, "
                                     "or delete the model here and add it again.")
                prec, path = entry["precision"], entry["path"]
            else:
                prec, path = resolve_weights(os.getcwd(), precision=variant, download=True)
            # free the previous build before loading another — two 12.9B transformers won't fit
            self._pipe, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            self._pipe = Krea2Pipeline(path, precision=prec, base_dir=os.environ.get("KREA2_BASE_DIR"))
            # built-ins cache per precision; local checkpoints per full variant id (two fine-tunes
            # with the same precision are still different weights)
            self._variant = variant if entry is not None else prec
        return self._pipe

    def generate(self, *, prompt, variant, params, step_callback):
        from krea2.pipeline import Krea2Pipeline
        image_path, strength = _img2img_args(params)   # (None, None) = plain txt2img
        lora_paths, lora_scales = _lora_args(params)   # (None, None) = no LoRA (server pre-resolves names→paths)
        kwargs = {}
        if image_path:
            import inspect
            # capability check BEFORE _get — don't load 14 GB of weights just to fail on an old package
            if "init_image" not in inspect.signature(Krea2Pipeline.generate).parameters:  # pre-0.2
                raise ValueError("This build of krea2-alis-mlx predates img2img — update it with "
                                 "`pip install -U git+https://github.com/avlp12/krea2_alis_mlx` "
                                 "(or reinstall the app), or remove the input image.")
            kwargs = {"init_image": image_path, "strength": strength}
        if lora_paths and not hasattr(Krea2Pipeline, "set_loras"):   # pre-0.3 — check before the big load
            raise ValueError("This build of krea2-alis-mlx predates LoRA support — update it with "
                             "`pip install -U git+https://github.com/avlp12/krea2_alis_mlx` "
                             "(or reinstall the app), or clear the selected LoRA(s).")
        pipe = self._get(variant)
        if self._lokr_pipe is not pipe:   # a reloaded build came up base — our wrappers died with it
            self._lokr_sig, self._lokr_paths, self._lokr_pipe = (), [], None
        specs = list(zip(lora_paths, lora_scales)) if lora_paths else []
        plain = [(p, s) for p, s in specs if not _is_lokr(p)]
        lokrs = [(p, s) for p, s in specs if _is_lokr(p)]
        # Our LoKr wrappers come off before the package re-applies its plain set: set_loras's
        # unload peels LoRALinear by path and would no-op past a _LoKrLinear sitting on top,
        # silently missing a plain-set update underneath. Restore + re-wrap is reference
        # swapping — microseconds — so correctness beats a signature-skip here.
        _unload_lokrs(pipe.transformer, self._lokr_paths)
        self._lokr_paths, self._lokr_sig = [], ()
        # Krea 2 plain LoRA is a runtime low-rank branch (not fused at load), so the set is
        # applied per generation without reloading — set_loras is a cheap no-op when unchanged,
        # and clearing it (empty specs) reverts to base when nothing is selected. A wrong-base
        # LoRA raises here, before any denoise work. Guarded so an old package (no set_loras)
        # still runs plain. LoKr files never reach it — the backend wraps those itself.
        if hasattr(pipe, "set_loras"):
            try:
                pipe.set_loras(plain)
            except ValueError as e:
                # Only claim "wrong model" for the base-mismatch signatures — a malformed file
                # (unpaired A/B, unknown key, collision) raises too, and that hint would mislead.
                mismatch = any(s in str(e) for s in
                               ("doesn't exist in the model tree", "not a linear layer", "shape mismatch"))
                hint = " A LoRA must be built for Krea 2 Turbo — one made for a different model won't fit." if mismatch else ""
                raise ValueError(f"Couldn't apply the selected LoRA to Krea 2 Turbo: {e}{hint}") from None
        if lokrs:
            try:
                self._lokr_paths = _apply_lokrs(pipe.transformer, lokrs)
            except ValueError as e:
                raise ValueError(f"Couldn't apply the LoKr LoRA to Krea 2 Turbo: {e}") from None
            self._lokr_sig = tuple(lokrs)
            self._lokr_pipe = pipe
        return pipe.generate(
            prompt,
            width=int(params.get("width", params.get("size", 1024))),
            height=int(params.get("height", params.get("size", 1024))),
            steps=int(params.get("steps", 8)),
            seed=int(params.get("seed", 0)),
            num_images=int(params.get("num_images", 1)),
            step_callback=step_callback,
            **kwargs,
        )

    # --- model management ---
    @staticmethod
    def _transformer_path(variant: str):
        from krea2.pipeline import _CACHE, BUILDS
        repo, fname = BUILDS[variant]
        return repo, fname, os.path.join(_CACHE, repo.replace("/", "__"), fname)

    def is_installed(self, variant: str) -> bool:
        if variant.startswith(_LOCAL):
            entry = self._local_entry(variant)
            return bool(entry and not entry["missing"])
        _, fname, cache = self._transformer_path(variant)
        return os.path.exists(cache) or os.path.exists(os.path.join(os.getcwd(), fname))

    def _specs(self, variant: str):
        """(url, dest) for the variant's transformer + the shared base (encoder/VAE/tokenizer)."""
        from huggingface_hub import HfApi
        from krea2.pipeline import _CACHE, BASE_REPO

        def url(repo, f):
            return f"https://huggingface.co/{repo}/resolve/main/{f}"

        repo, fname, cache = self._transformer_path(variant)
        specs = [(url(repo, fname), cache)]
        base_root = os.path.join(_CACHE, BASE_REPO.replace("/", "__"))
        exts = (".safetensors", ".json", ".jinja", ".txt", ".model")
        for s in HfApi().model_info(BASE_REPO).siblings:
            f = s.rfilename
            if (f.startswith(("vae/", "text_encoder/", "tokenizer/")) or f == "model_index.json") and f.endswith(exts):
                specs.append((url(BASE_REPO, f), os.path.join(base_root, f)))
        return specs

    def download(self, variant: str, progress) -> None:
        if variant.startswith(_LOCAL):
            raise ValueError("Local models have nothing to download — if it shows as missing, its "
                             "file was moved or deleted; delete it here and add it again.")
        from studio.download import download_files
        download_files(self._specs(variant), progress)

    def delete(self, variant: str) -> None:
        if variant.startswith(_LOCAL):
            local_models.remove(variant[len(_LOCAL):])   # unregisters only — never deletes the file
            if self._variant == variant:                 # drop the loaded build if we just removed it
                self._pipe, self._variant = None, None
            return
        _, _, cache = self._transformer_path(variant)
        if os.path.exists(cache):
            os.remove(cache)
        if self._variant == variant:  # drop the loaded build if we just deleted it
            self._pipe, self._variant = None, None
