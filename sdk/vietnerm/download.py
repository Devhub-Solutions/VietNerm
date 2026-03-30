"""Model download/cache configuration helpers for VietNerm SDK."""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import httpx


@contextlib.contextmanager
def no_ssl_verification() -> Iterator[None]:
    """Temporarily force all ``httpx.Client`` instances to ``verify=False``."""
    client_cls = httpx.Client
    async_client_cls = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        kwargs.pop("verify", None)
        return client_cls(*args, verify=False, **kwargs)

    def _async_client_factory(*args, **kwargs):
        kwargs.pop("verify", None)
        return async_client_cls(*args, verify=False, **kwargs)

    httpx.Client = _client_factory  # type: ignore[assignment]
    httpx.AsyncClient = _async_client_factory  # type: ignore[assignment]
    try:
        yield
    finally:
        httpx.Client = client_cls  # type: ignore[assignment]
        httpx.AsyncClient = async_client_cls  # type: ignore[assignment]


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

    @contextlib.contextmanager
    def ssl_context(self) -> Iterator[None]:
        """Context manager that enables SSL bypass if configured."""
        if self.disable_ssl_verify:
            with no_ssl_verification():
                yield
        else:
            with contextlib.nullcontext():
                yield


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


def predownload_model(
    repo_id: str,
    config: Optional[DownloadConfig] = None,
    local_dir: Optional[str] = None,
    local_dir_use_symlinks: bool = False,
) -> str:
    """Pre-download a model snapshot into local cache and return local path."""
    cfg = config or DownloadConfig()
    cfg.apply_environment()

    from huggingface_hub import snapshot_download

    kwargs = cfg.to_hf_kwargs()
    if local_dir:
        kwargs["local_dir"] = local_dir
        kwargs["local_dir_use_symlinks"] = local_dir_use_symlinks

    with cfg.ssl_context():
        return snapshot_download(repo_id=repo_id, **kwargs)
