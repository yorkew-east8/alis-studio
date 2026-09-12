"""ERNIE-Image-Turbo backend via mflux (Baidu, Apache-2.0).

ERNIE-Image is Baidu's 8B single-stream diffusion transformer (April 2026), paired here with a
Mistral-style text encoder and a prompt/perception encoder (~15.7B params across the pipeline) and
the FLUX.2 VAE. It is the strongest model in this app for **text rendering and structured layouts**
(posters, comics, labels) — complex-instruction following where the letters have to be right. The
Turbo variant is distilled to produce clean images in ~8 steps (guidance 1.0, no CFG).

mflux ships a first-class MLX implementation (`mflux.models.ernie_image`), so this is a thin wiring
backend like Z-Image / Qwen-Image — no model port. img2img and LoRA come for free from mflux.

A roomy-Mac model. The weights are ~32 GB (bf16) and mflux loads them fully before quantizing, so
even the 8-bit build peaks near full precision at load — there is no pre-quantized repo that would
let it stream onto a 16 GB machine (unlike Z-Image's 4-bit repo). 4-bit is therefore *not* offered:
it would shrink the resident footprint but not the load peak, so it buys a small Mac nothing while
implying it can run — 8-bit is the honest floor. (Power users can still `mflux-save` a 4-bit ERNIE.)
"""

from __future__ import annotations

from .base import Backend
from .mflux_common import (_apply_memory_policy, _construct_checking_lora, _hf_downloaded, _img2img_args,
                           _img2img_params, _lora_args, _lora_params, _lora_sig, _wire_progress)

_QUANT = {"8bit": 8, "bf16": None}


class ErnieImageTurboBackend(Backend):
    id = "ernie-image-turbo"
    label = "ERNIE-Image Turbo"
    supports_preview = True   # mflux ErnieImage exposes _decode_latents(latents=…) → live preview works
    min_ram_gib = 48   # ~32 GB bf16 weights, loaded fully before quantizing → 8-bit load peaks ~40 GB
    prompt_note = ("Baidu's 8B model — the strongest here for text rendering and posters/labels/"
                   "structured layouts. Distilled: fast at ~8 steps. Best with English or Chinese prompts.")
    info = "Apache-2.0 (open) · ~32 GB download on first use via mflux · wants a roomy Mac (8-bit ~48 GB)"
    variants = [
        {"id": "8bit", "label": "8-bit · ~32 GB download · wants ≥ 48 GB RAM", "min_ram": 48},
        {"id": "bf16", "label": "bf16 · full precision · wants ≥ 64 GB RAM", "min_ram": 64},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1376], "default_size": 1024,   # native 1024; rectangular to ~1376×768
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1376, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 30, "default": 8,
         "hint": "Turbo is distilled — 8 is the model's recipe; 8–12 is plenty."},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 1, "max": 5, "step": 0.5, "default": 1.0,
         "hint": "Turbo's recipe is 1.0 (no CFG, fastest). Above 1.0 turns CFG on — slower, and the "
                 "negative prompt starts to matter."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced", "default": "",
         "hint": "Only used when Guidance is above 1.0 (CFG on)."},
        *_img2img_params(),
        *_lora_params(),
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.ernie_image.variants.txt2img.ernie_image  # noqa: F401
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
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
        loras = _lora_sig(params or {})
        if self._model is None or self._variant != variant or self._loras != loras:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            if variant not in _QUANT:   # never substitute silently — the user would get (and cache) the wrong build
                raise ValueError(f"Unknown build '{variant}' for {self.label} — expected one of: {', '.join(_QUANT)}.")
            lora_paths, lora_scales = _lora_args(params or {})
            self._model = _construct_checking_lora(
                lambda: ErnieImage(quantize=_QUANT.get(variant), model_config=ModelConfig.ernie_image_turbo(),
                                   lora_paths=lora_paths, lora_scales=lora_scales),
                lora_paths)
            self._variant = variant
            self._loras = loras
        return self._model

    def will_load(self, variant):
        return self._model is None or self._variant != variant

    def is_downloaded(self, variant):
        from mflux.models.common.config import ModelConfig
        from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition
        return _hf_downloaded([ModelConfig.ernie_image_turbo().model_name],
                              ErnieWeightDefinition.get_download_patterns())

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant, params)
        w, h = int(params.get("width", 1024)), int(params.get("height", 1024))
        _apply_memory_policy(model, w, h)
        neg = (params.get("negative") or "").strip() or None
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 8)),
                height=h, width=w,
                guidance=float(params.get("guidance", 1.0) or 1.0), negative_prompt=neg,
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out
