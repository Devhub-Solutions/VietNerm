"""Model download/cache configuration helpers for VietNerm SDK."""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

import httpx

T = TypeVar("T")


@dataclass
class DownloadConfig:
    """Download/runtime options for HuggingFace model loading."""

    cache_dir: Optional[str] = None
    local_files_only: bool = False
    force_download: bool = False
    revision: Optional[str] = None
    token: Optional[str] = None
    disable_ssl_verify: bool = False
    auto_disable_ssl_fallback: bool = True

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
def no_ssl_verification():
    """Context manager to temporarily disable SSL verification in httpx and requests.
    
    This is used as a fallback when HuggingFace Hub downloads fail due to 
    self-signed certificates or other SSL issues.
    """
    import httpx
    import requests
    import urllib3
    
    # Disable urllib3 warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Patch httpx
    original_httpx_client = httpx.Client
    def patched_httpx_client(*args, **kwargs):
        if "verify" not in kwargs:
            kwargs["verify"] = False
        return original_httpx_client(*args, **kwargs)
    
    # Patch requests
    original_requests_request = requests.Session.request
    def patched_requests_request(self, method, url, **kwargs):
        if "verify" not in kwargs:
            kwargs["verify"] = False
        return original_requests_request(self, method, url, **kwargs)
    
    httpx.Client = patched_httpx_client
    requests.Session.request = patched_requests_request
    requests.request = lambda method, url, **kwargs: requests.Session().request(method, url, **kwargs)
    
    try:
        yield
    finally:
        httpx.Client = original_httpx_client
        requests.Session.request = original_requests_request
        # Re-import to restore top-level request if needed, 
        # but Session.request is the core one used by most libs.


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

    return with_ssl_fallback(
        lambda: snapshot_download(repo_id=repo_id, **cfg.to_hf_kwargs()),
        cfg,
    )


def with_ssl_fallback(fn: Callable[[], T], config: Optional[DownloadConfig] = None) -> T:
    """Run a download function with automatic SSL fallback when possible.

    Behavior:
      1. If ``disable_ssl_verify=True``: execute directly in no-SSL mode.
      2. Otherwise, execute normally first.
      3. If failed and ``auto_disable_ssl_fallback=True``: retry once in no-SSL mode.
    """
    cfg = config or DownloadConfig()

    if cfg.disable_ssl_verify:
        with no_ssl_verification():
            return fn()

    try:
        return fn()
    except Exception as exc:
        if not cfg.auto_disable_ssl_fallback:
            raise

        message = str(exc).lower()
        looks_like_ssl_error = (
            isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
            or "ssl" in message
            or "certificate" in message
            or "self signed" in message
        )
        if not looks_like_ssl_error:
            raise

        with no_ssl_verification():
            return fn()
