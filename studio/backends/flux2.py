"""FLUX.2-klein-4B backend via mflux (Black Forest Labs, Jan 2026, Apache-2.0, ungated).

The distilled 4B member of the FLUX.2 family: high-quality images in ~4 steps, ~15 GB download
(bf16) — quantized on the fly like the other mflux backends. Unlike FLUX.1 (gated, English-centric
T5/CLIP), klein is fully open and uses the FLUX.2 text stack. Supports img2img and LoRA.
"""

from __future__ import annotations

from .base import Backend
from .mflux_common import (_apply_memory_policy, _construct_checking_lora, _hf_downloaded, _img2img_args,
                           _img2img_params, _lora_args, _lora_params, _lora_sig, _wire_progress)

_QUANT = {"8bit": 8, "4bit": 4, "bf16": None}


class Flux2KleinBackend(Backend):
    id = "flux2-klein"
    label = "FLUX.2 klein 4B"
    min_ram_gib = 16   # 4B DiT; 4-bit pipeline is small — the 16 GB class alongside Z-Image
    prompt_note = "BFL's 2026 fast model — strong prompt following, ~4 steps. English prompts work best."
    info = "Apache-2.0 (open, ungated) · ~15 GB download on first use via mflux · distilled ~4-step"
    variants = [
        {"id": "4bit", "label": "4-bit · 16 GB-Mac friendly"},
        {"id": "8bit", "label": "8-bit · wants ≥ 24 GB RAM", "min_ram": 24},
        {"id": "bf16", "label": "bf16 · full precision · wants ≥ 32 GB RAM", "min_ram": 32},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280], "default_size": 1024,
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1536, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 12, "default": 4,
         "hint": "Distilled — 4 steps is the sweet spot."},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 1, "max": 4, "step": 0.5, "default": 1.0, "fixed": True,
         "hint": "klein is distilled — guidance stays at 1."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Distilled klein runs without CFG, so a negative prompt has no effect."},
        *_img2img_params(),
        *_lora_params(),
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.flux2.variants.txt2img.flux2_klein  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._variant = None
        self._loras = ()

    def will_load(self, variant):
        return self._model is None or self._variant != variant

    def is_downloaded(self, variant):
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
        return _hf_downloaded([ModelConfig.flux2_klein_4b().model_name],
                              Flux2KleinWeightDefinition.get_download_patterns())

    def _get(self, variant, params=None):
        import gc
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        loras = _lora_sig(params or {})
        if self._model is None or self._variant != variant or self._loras != loras:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            lora_paths, lora_scales = _lora_args(params or {})
            self._model = _construct_checking_lora(
                lambda: Flux2Klein(model_config=ModelConfig.flux2_klein_4b(),
                                   quantize=_QUANT.get(variant, 4),
                                   lora_paths=lora_paths, lora_scales=lora_scales),
                lora_paths)
            self._variant = variant
            self._loras = loras
        return self._model

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
                num_inference_steps=int(params.get("steps", 4)),
                height=h, width=w,
                guidance=1.0,   # distilled
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out
