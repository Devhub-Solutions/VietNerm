"""Tests for SDK download configuration helpers."""

import os

from vietnerm.download import DownloadConfig, clear_cache, no_ssl_verification


def test_no_ssl_verification_context_forces_verify_false(monkeypatch):
    calls = []

    class DummyClient:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("verify"))

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("verify"))

    import vietnerm.download as dl

    monkeypatch.setattr(dl.httpx, "Client", DummyClient)
    monkeypatch.setattr(dl.httpx, "AsyncClient", DummyAsyncClient)

    with no_ssl_verification():
        dl.httpx.Client(verify=True)
        dl.httpx.AsyncClient(verify=True)

    assert calls == [False, False]


def test_download_config_kwargs_and_ssl_env():
    cfg = DownloadConfig(
        cache_dir="/tmp/vietnerm-cache",
        local_files_only=True,
        force_download=True,
        revision="main",
        token="abc",
        disable_ssl_verify=True,
    )

    cfg.apply_environment()

    assert cfg.to_hf_kwargs() == {
        "cache_dir": "/tmp/vietnerm-cache",
        "local_files_only": True,
        "force_download": True,
        "revision": "main",
        "token": "abc",
    }
    assert os.environ.get("HF_HUB_DISABLE_SSL_VERIFY") == "1"


def test_clear_cache_repo_only(tmp_path):
    repo_path = tmp_path / "hub" / "models--owner--repo"
    repo_path.mkdir(parents=True)
    (repo_path / "dummy.bin").write_text("x", encoding="utf-8")

    deleted = clear_cache(cache_dir=str(tmp_path), repo_id="owner/repo")

    assert deleted == 1
    assert not repo_path.exists()
