"""Model download/cache configuration helpers for VietNerm SDK."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DownloadConfig:
    """Download/runtime options for HuggingFace model loading."""

    cache_dir: Optional[str] = None
    local_files_only: bool = False
    force_download: bool = False
    revision: Optional[str] = None
    token: Optional[str] = None
    disable_ssl_verify: bool = False

    def apply_environment(self) -> None:
        if self.disable_ssl_verify:
            os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""

    def to_hf_kwargs(self) -> dict:
        kwargs = {
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.revision:
            kwargs["revision"] = self.revision
        if self.token:
            kwargs["token"] = self.token
        return kwargs


def clear_cache(cache_dir: Optional[str] = None, repo_id: Optional[str] = None) -> int:
    """Clear VietNerm/HuggingFace cache and return deleted entry count."""
    root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "huggingface"

    if repo_id:
        repo_key = f"models--{repo_id.replace('/', '--')}"
        target = root / "hub" / repo_key
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            return 1
        return 0

    if not root.exists():
        return 0

    deleted = 0
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            deleted += 1
        else:
            child.unlink(missing_ok=True)
            deleted += 1
    return deleted


def predownload_model(repo_id: str, config: Optional[DownloadConfig] = None) -> str:
    """Pre-download a model snapshot into local cache and return local path."""
    cfg = config or DownloadConfig()
    cfg.apply_environment()

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, **cfg.to_hf_kwargs())
