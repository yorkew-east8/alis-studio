"""Qwen-Image-Edit backend via mflux (Tongyi / Alibaba, Apache-2.0).

Instruction image editing: attach an image and describe the change ("make the hat red") — unlike
plain img2img (strength-based) this follows an edit instruction. mflux normalizes the output to about
1 megapixel (≈1024²) keeping the input's aspect ratio, so the result is close to — but not exactly —
the input's size. A separate model (Qwen/Qwen-Image-Edit-2509, ~54 GB), loaded lazily on first use.
Reuses the shared image-upload param and the progress bridge.

Only 8-bit and bf16 are offered: mflux's 4-bit quantization of this model is broken upstream (it
decodes to grainy noise regardless of step count — reproduced with the stock mflux CLI), so it is
deliberately not exposed. 8-bit peaks ~39 GB, so this backend needs a roomy Mac (see min_ram_gib).
"""

from __future__ import annotations

from .base import Backend
from .mflux_common import (_apply_memory_policy, _construct_checking_lora, _hf_downloaded, _lora_args,
                           _lora_params, _lora_sig, _wire_progress)

_QUANT = {"8bit": 8, "bf16": None}


class QwenImageEditBackend(Backend):
    id = "qwen-image-edit"
    label = "Qwen-Image Edit"
    min_ram_gib = 64   # 8-bit floor. Measured peaks: 8-bit ~39 GB, bf16 ~58 GB → ~1.65× headroom floors.
    prompt_note = "Describe the change to make (e.g. \"make the hat red\"). Understands Korean and other languages (Qwen encoder)."
    info = "Apache-2.0 · instruction image editing · ~54 GB download on first use · 8-bit needs ~64 GB RAM (bf16 ~96 GB)"
    variants = [
        {"id": "8bit", "label": "8-bit · needs ~64 GB RAM", "min_ram": 64},
        {"id": "bf16", "label": "bf16 · full precision · needs ~96 GB RAM", "min_ram": 96},
    ]
    params = [
        {"key": "init_image", "label": "Image to edit", "type": "image", "group": "Edit",
         "hint": "Required — attach an image, then describe the change in the prompt."},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output", "min": 1, "max": 50, "default": 4,
         "hint": "Few-step edit model — 4 is the model's default; 4–8 is plenty, more rarely helps."},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 1, "max": 10, "step": 0.5, "default": 4.0,
         "hint": "How strongly to follow the edit instruction."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced", "default": ""},
        *_lora_params(),
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.qwen.variants.edit.qwen_image_edit  # noqa: F401
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
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
        return _hf_downloaded([ModelConfig.qwen_image_edit().model_name],
                              QwenWeightDefinition.get_download_patterns())

    def _get(self, variant, params=None):
        import gc
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        loras = _lora_sig(params or {})
        if self._model is None or self._variant != variant or self._loras != loras:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            lora_paths, lora_scales = _lora_args(params or {})
            self._model = _construct_checking_lora(
                lambda: QwenImageEdit(quantize=_QUANT.get(variant, 8), model_config=ModelConfig.qwen_image_edit(),
                                      lora_paths=lora_paths, lora_scales=lora_scales),
                lora_paths)
            self._variant = variant
            self._loras = loras
        return self._model

    def generate(self, *, prompt, variant, params, step_callback):
        img_path = params.get("image_path")   # set by the server from the uploaded init_image
        if not img_path:
            raise ValueError("Attach an image to edit — Qwen-Image Edit transforms an existing image.")
        model = self._get(variant, params)
        try:  # mflux decodes the edit at ~1 MP (≈1024²) regardless of input size, so base the VAE-tiling
            import math  # decision on that normalized size — not the raw input — to match the real decode
            from PIL import Image
            iw, ih = Image.open(img_path).size
            scale = math.sqrt((1024 * 1024) / max(1, iw * ih))
            _apply_memory_policy(model, round(iw * scale), round(ih * scale))
        except Exception:
            pass
        neg = (params.get("negative") or "").strip() or None
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                image_paths=[img_path], image_path=img_path,
                num_inference_steps=int(params.get("steps", 4)),
                guidance=float(params.get("guidance", 4.0) or 4.0),
                negative_prompt=neg,
            )
            out.append(img.image)
        return out
