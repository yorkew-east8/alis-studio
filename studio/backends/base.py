"""Backend interface — a pluggable image-generation model.

To add a model to Alis Studio, drop a `studio/backends/<name>.py` that subclasses Backend,
declares its settings (`params`) and downloadable builds (`catalog`), implements `generate(...)`,
and register it in `studio/registry.py`. The web UI discovers everything via `/api/models` and
`/api/catalog` and renders itself — no UI changes needed to add a model.

`params` is a list of control specs the UI renders into the settings panel and passes back to
`generate(..., params=...)`. Each spec: {key, label, type, group, default, ...type-specific}. Types:
  - "resolution": {sizes:[...], default_size, aspects:["1:1",...], min, max, multiple} → emits width,height
  - "int" / "float": {min, max, step, default}
  - "select": {options:[{value,label}], default}
  - "seed": {default}                      (number + randomize + lock)
  - "text": {default}
  - "image": {}                            drop-zone upload → data URI in params[key]; the server
                                           decodes it to params["image_path"] for img2img backends
Flags: "fixed": shown read-only; "enabled": False shown disabled; "hint": one-line note.
"""

from __future__ import annotations


class Backend:
    id = ""                      # stable id, e.g. "krea2-turbo"
    label = ""                   # display name, e.g. "Krea 2 Turbo"
    info = ""                    # one-line note shown for non-downloadable models (e.g. "auto-downloads on first use")
    prompt_note = ""             # one-line hint about prompt language (shown under the prompt box)
    variants: list[dict] = []    # selectable builds: [{"id": "8bit", "label": "8-bit · best quality"}]
    params: list[dict] = []      # settings schema (see module docstring)
    catalog: list[dict] = []     # downloadable builds (see studio/registry conventions); empty = managed elsewhere
    min_ram_gib = 0              # unified-memory floor to run this model acceptably; drives the "recommended for your Mac" hint
    supports_preview = False     # True if generate() can stream in-progress frames (the UI shows a "Live preview" toggle)

    @classmethod
    def is_available(cls) -> bool:
        """True if this backend's code dependencies are importable. Unavailable backends are
        skipped at registration so the app still runs with whatever is installed."""
        return True

    def extra_variants(self) -> list[dict]:
        """Runtime-added selectable builds (user-registered local checkpoints), merged after the
        class-level `variants` in meta(). Each dict: {"id", "label", "min_ram"?}."""
        return []

    def meta(self) -> dict:
        return {"id": self.id, "label": self.label, "info": self.info,
                "prompt_note": self.prompt_note, "variants": self.variants + self.extra_variants(),
                "params": self.params,
                "min_ram_gib": self.min_ram_gib, "supports_preview": self.supports_preview}

    def min_ram_for(self, variant: str) -> int:
        """RAM floor (GiB) for a specific build. Defaults to the backend-wide min_ram_gib, but a
        variant may override it (e.g. a bf16 build needs far more than the 8-bit one) by carrying a
        "min_ram" key in its `variants` entry. Drives the UI warning and the server-side gate."""
        for v in self.variants + self.extra_variants():
            if v.get("id") == variant and v.get("min_ram"):
                return int(v["min_ram"])
        return self.min_ram_gib

    def catalog_entries(self) -> list[dict]:
        """Builds shown in the model manager (/api/catalog). Static by default; backends with
        runtime-added builds extend it."""
        return self.catalog

    def generate(self, *, prompt: str, variant: str, params: dict, step_callback):
        """Return a list of PIL.Image. `params` carries the resolved settings (width, height,
        steps, seed, num_images, and any model-specific keys). Call step_callback(step, total)
        per denoising step so the UI can show live progress."""
        raise NotImplementedError

    def will_load(self, variant: str) -> bool:
        """True if the next generate() will construct/load this variant's model into memory (it isn't
        the one already cached on this backend) — a multi-second wait the UI should explain. The very
        first time a model is used it also downloads its weights; later loads come from disk cache.
        Safe to consult off the GPU thread: generation is single-flight (serialized under _LOCK + the
        one GPU thread), so the state can't change before _get() runs. Default False (no load step)."""
        return False

    # --- model management (override for downloadable backends) ---
    def is_installed(self, variant: str) -> bool:
        return True

    def download(self, variant: str, progress) -> None:
        """Download `variant`'s weights, calling progress(done_bytes, total_bytes) as it goes."""
        raise NotImplementedError

    def delete(self, variant: str) -> None:
        raise NotImplementedError
