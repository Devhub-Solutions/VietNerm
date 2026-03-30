"""Tests for SDK download configuration helpers."""

import os

from vietnerm.download import DownloadConfig, clear_cache, with_ssl_fallback


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
    assert os.environ.get("SSL_CERT_FILE") == ""
    assert os.environ.get("PYTHONHTTPSVERIFY") == "0"


def test_clear_cache_repo_only(tmp_path):
    repo_path = tmp_path / "hub" / "models--owner--repo"
    repo_path.mkdir(parents=True)
    (repo_path / "dummy.bin").write_text("x", encoding="utf-8")

    deleted = clear_cache(cache_dir=str(tmp_path), repo_id="owner/repo")

    assert deleted == 1
    assert not repo_path.exists()


def test_with_ssl_fallback_retries_on_ssl_error():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED")
        return "ok"

    out = with_ssl_fallback(flaky, DownloadConfig())
    assert out == "ok"
    assert calls["count"] == 2


def test_with_ssl_fallback_does_not_retry_non_ssl_error():
    calls = {"count": 0}

    def broken():
        calls["count"] += 1
        raise RuntimeError("model file not found")

    try:
        with_ssl_fallback(broken, DownloadConfig())
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "not found" in str(exc)
    assert calls["count"] == 1
