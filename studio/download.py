"""HTTP-bridge file downloader with a progress callback.

Mirrors the krea2 package's resilient downloader (plain HTTP to the HF CDN, resumable, integrity-
checked) but reports progress to a callback instead of printing — so the model manager can show a
live bar. We bypass huggingface_hub's Xet client on purpose (it can hang behind some firewalls).
Sends the user's Hugging Face token when one is available — krea/Krea-2-Turbo is a gated repo now.
"""

from __future__ import annotations

import os


def _auth(url: str) -> dict:
    """Bearer token for huggingface.co URLs (krea/Krea-2-Turbo is now a gated repo).

    Reads HF_TOKEN and the token stored by `hf auth login` — the stored one is what works for
    the Mac app, which inherits no shell env when launched from Finder.
    """
    if not url.startswith("https://huggingface.co/"):
        return {}
    try:
        from krea2.pipeline import _auth_headers  # krea2-alis-mlx >= 0.3.1
        return _auth_headers()
    except Exception:
        try:
            from huggingface_hub import get_token
            token = (get_token() or "").strip()
        except Exception:
            token = (os.environ.get("HF_TOKEN") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}


def _gate_error(url: str, status: int) -> RuntimeError:
    repo = "/".join(url.split("/")[3:5])  # https://huggingface.co/<org>/<name>/resolve/...
    return RuntimeError(
        f"{repo} requires Hugging Face access ({status}). One-time setup: "
        f"1) visit https://huggingface.co/{repo} while logged in and accept the terms; "
        "2) run `hf auth login` (or `huggingface-cli login`) in Terminal. Then retry the download."
    )


def _head_size(url: str) -> int:
    import requests
    try:
        r = requests.head(url, headers=_auth(url), allow_redirects=True, timeout=30)
        # a 401/403 answer carries the error page's length, not the file's — treat as unknown,
        # or a leftover .part bigger than the error page gets deleted as "stale" (data loss)
        return int(r.headers.get("content-length") or 0) if r.ok else 0
    except Exception:
        return 0


def _download_one(url: str, dest: str, total: int, on_bytes) -> None:
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if total and pos == total:        # complete .part left by a crash before the rename
        os.replace(tmp, dest)
        on_bytes(total)
        return
    if total and pos > total:         # stale/corrupt leftover
        os.remove(tmp)
        pos = 0
    headers = {**_auth(url), **({"Range": f"bytes={pos}-"} if pos else {})}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as r:
        if r.status_code == 416 and pos:  # unknowable total + stale/complete .part -> restart clean
            os.remove(tmp)
            return _download_one(url, dest, total, on_bytes)
        if r.status_code in (401, 403):   # gated repo: raise setup instructions, not a bare 401
            raise _gate_error(url, r.status_code)
        r.raise_for_status()  # other HTTP errors surface as-is, not as a resume hint
        resume = bool(pos) and r.status_code == 206
        pos = pos if resume else 0
        total = total or (pos + int(r.headers.get("content-length") or 0))
        done = pos
        try:
            with open(tmp, "ab" if resume else "wb") as f:
                for chunk in r.iter_content(4 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    on_bytes(done)
        except requests.exceptions.RequestException as e:
            # a mid-stream drop lands here; .part is kept so the next run resumes
            raise OSError(f"download of {os.path.basename(dest)} interrupted; re-run to resume") from e
    # commit only a verified-complete, non-empty file. For length-less (chunked) transfers requests
    # raises above on a short read, so a clean loop means complete — but never commit an empty result.
    if (total and done != total) or done == 0:
        raise OSError(f"incomplete download of {os.path.basename(dest)} ({done}/{total or '?'} bytes)")
    os.replace(tmp, dest)


def download_files(specs, progress) -> None:
    """specs: list of (url, dest). progress(done_total_bytes, grand_total_bytes) is called as bytes land."""
    sizes = [_head_size(u) for u, _ in specs]
    grand = sum(sizes)
    base = 0
    for (url, dest), sz in zip(specs, sizes):
        exists = os.path.exists(dest) and os.path.getsize(dest) > 0
        if exists and ((sz and os.path.getsize(dest) == sz) or not sz):
            # size match, or the pre-flight couldn't answer (gate/offline) — trust the cached file
            # rather than brick a complete install
            base += sz
            progress(base, grand)
            continue
        start = base
        _download_one(url, dest, sz, lambda d: progress(start + d, grand))
        base += sz
        progress(base, grand)
