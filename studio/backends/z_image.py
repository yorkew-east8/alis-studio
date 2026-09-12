"""Z-Image-Turbo backend via mflux (Tongyi-MAI / Alibaba, Apache-2.0).

Z-Image-Turbo is a ~6B single-stream DiT with a Qwen3-4B text encoder (so it understands
Korean and other languages natively) and the FLUX VAE, distilled to ~9 steps with no CFG.
At 4-bit the whole pipeline is small enough (~6 GB) to run on a 16 GB Mac — unlike the 12.9B
Krea 2 Turbo default, which effectively needs a big-RAM machine.

mflux already ships a first-class MLX implementation (the same library this app uses for FLUX
and Qwen-Image), so this is a thin wiring backend like the others — no model port.

The default 4-bit build loads mflux's pre-quantized repo (``filipstrand/Z-Image-Turbo-mflux-4bit``,
~6 GB) so a clean 16 GB machine never has to hold the full-precision weights. The 8-bit and bf16
builds quantize on the fly from the official ``Tongyi-MAI/Z-Image-Turbo`` repo (~33 GB download).
"""

from __future__ import annotations

import os

from .base import Backend
from .mflux_common import (_apply_memory_policy, _construct_checking_lora, _hf_downloaded,
                           _img2img_args,
                           _img2img_params, _lora_args, _lora_params, _lora_sig, _wire_progress)

class ZImageTurboBackend(Backend):
    """Z-Image-Turbo (open, Apache-2.0) via mflux — downloads on first use, no HF gating.

    Subclassable: a Z-Image finetune backend (e.g. cyber_z.py) overrides id/label/info/variants and
    BUILDS — everything else (params, loading, img2img, memory policy) is shared."""

    # variant id -> (mflux model_path, quantize).
    # 4-bit: a ready pre-quantized repo (no on-the-fly quantization, light download, 16 GB-friendly).
    # 8-bit/bf16: the official full-precision repo, quantized at load (8) or kept as-is (None=bf16).
    # Subclasses override this dict; every variants[] id MUST be a BUILDS key (unknown ids raise).
    BUILDS = {
        "4bit": ("filipstrand/Z-Image-Turbo-mflux-4bit", None),
        "8bit": (None, 8),
        "bf16": (None, None),
    }
    id = "z-image-turbo"
    label = "Z-Image Turbo"
    supports_preview = True   # ZImage exposes _decode_latents(latents, config) → live preview works (inherited by cyber-z)
    min_ram_gib = 16   # 4-bit pipeline ~6 GB resident; 1024² peaks ~8.5 GB with VAE tiling → runs on 16 GB
    prompt_note = "The general-purpose base model (Apache-2.0 — the safest license here). Understands Korean natively (Qwen3 encoder); distilled — fast at ~9 steps."
    info = "Apache-2.0 (open) · 4-bit runs on a 16 GB Mac (best at 512–768px) · downloads on first use via mflux"
    # 4-bit is listed first on purpose: it is the default (variants[0]) and the only 16 GB-friendly build.
    variants = [
        {"id": "4bit", "label": "4-bit · ~6 GB · 16 GB-Mac friendly"},
        # 8-bit/bf16 quantize on the fly from the ~33 GB bf16 repo — the transient full-precision
        # weights want a roomy Mac, hence the explicit floors
        {"id": "8bit", "label": "8-bit · ~33 GB download · wants ≥ 24 GB RAM", "min_ram": 24},
        {"id": "bf16", "label": "bf16 · full precision, ~33 GB · wants ≥ 32 GB RAM", "min_ram": 32},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280], "default_size": 1024,   # native 1024; usable to ~1280
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1536, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 20, "default": 9},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 0, "max": 10, "step": 0.5, "default": 0, "fixed": True,
         "hint": "Z-Image Turbo is distilled — it runs without CFG (guidance 0)."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Turbo runs without guidance, so a negative prompt has no effect."},
        *_img2img_params(),
        *_lora_params(),
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.z_image.variants.z_image  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._variant = None
        self._loras = ()   # LoRA signature of the cached model — mflux fuses LoRA at construction

    def _get(self, variant, params=None):
        import gc
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage
        loras = _lora_sig(params or {})
        if self._model is None or self._variant != variant or self._loras != loras:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            builds = self.BUILDS
            if variant not in builds:   # never substitute silently — the user would get (and cache) the wrong build
                raise ValueError(f"Unknown build '{variant}' for {self.label} — expected one of: {', '.join(builds)}.")
            model_path, quantize = builds[variant]
            lora_paths, lora_scales = _lora_args(params or {})
            try:
                self._model = _construct_checking_lora(
                    lambda: ZImage(model_config=ModelConfig.z_image_turbo(),
                                   quantize=quantize, model_path=model_path,
                                   lora_paths=lora_paths, lora_scales=lora_scales),
                    lora_paths)
            except Exception as e:  # pre-quant builds live in community repos — guide, don't traceback
                m = str(e).lower()
                if model_path and any(k in m for k in ("not found", "404", "401", "403",
                                                       "repository", "gated", "restricted")):
                    raise ValueError(
                        f"The {variant} build ({model_path}) couldn't be downloaded — it may have "
                        "moved or you may be offline. Try another build of this model."
                    ) from None
                raise
            self._variant = variant
            self._loras = loras
            if model_path and "/" in model_path and not os.path.exists(model_path):
                try:  # mflux never fetches the repo's root config.json, but that's the file the Hub
                    from huggingface_hub import hf_hub_download  # counts — touch it so downloads register
                    hf_hub_download(model_path, "config.json")
                except Exception:
                    pass
        return self._model

    def will_load(self, variant):
        # a LoRA-set change also reloads, but that happens inside generate (params aren't available
        # here); it's a seconds-long fuse, not a download — no loading banner needed
        return self._model is None or self._variant != variant

    def is_downloaded(self, variant):
        from mflux.models.common.config import ModelConfig
        mp = self.BUILDS.get(variant, (None, None))[0]
        if mp and "/" in mp and not os.path.exists(mp):
            repos = [mp]   # pre-quant repos are self-contained (transformer + vae + text encoder)
        else:
            repos = [ModelConfig.z_image_turbo().model_name]   # 8-bit/bf16 quantize from the official repo
        return _hf_downloaded(repos)

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant, params)
        w, h = int(params.get("width", 1024)), int(params.get("height", 1024))
        _apply_memory_policy(model, w, h)
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 9)),
                height=h, width=w,
                guidance=0.0,   # turbo is distilled (supports_guidance=False); mflux forces 0 anyway
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out
