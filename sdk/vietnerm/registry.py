"""
Model Registry - Centralized discovery and management of VietNerm NER models.

The registry provides a single source of truth for available document types
and their corresponding models. It supports multiple discovery strategies:

  1. **HuggingFace Hub discovery**: Scans the Hub for repos matching the
     naming convention ``{hf_username}/phobert-{doc_type}-ner``.
  2. **Local registry file**: Reads from ``registry/documents.yaml`` in the
     project root (used during development and CI/CD).
  3. **Cached registry**: Stores discovered models in a local cache file
     to avoid repeated Hub API calls.

The registry is designed to be the ONLY place where doc types are listed.
All other components (detector, schema_mapper, ner) derive their knowledge
from the registry or from model configs (id2label).

Usage::

    >>> from vietnerm.registry import ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.list_doc_types()
    ['cccd', 'giay_khai_sinh', 'giay_ra_vien', 'gplx', 'vehicle_registration']
    >>> registry.get_model_info('cccd')
    {'doc_type': 'cccd', 'repo_id': 'ngocthanhdoan/phobert-cccd-ner', ...}
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cache TTL: 1 hour
_CACHE_TTL_SECONDS = 3600
_DEFAULT_HF_USERNAME = "ngocthanhdoan"
_CACHE_DIR = Path.home() / ".cache" / "vietnerm"
_CACHE_FILE = _CACHE_DIR / "model_registry.json"


class ModelRegistry:
    """Centralized registry for VietNerm NER models.

    Discovers available models from HuggingFace Hub and/or local registry
    file. Results are cached locally to avoid repeated API calls.

    Args:
        hf_username: HuggingFace username/org hosting the models.
        local_registry_path: Path to local ``documents.yaml`` registry file.
            If None, tries ``registry/documents.yaml`` relative to cwd.
        cache_ttl: Cache time-to-live in seconds. Default 3600 (1 hour).
        force_refresh: If True, bypass cache and re-discover from Hub.
    """

    def __init__(
        self,
        hf_username: str = _DEFAULT_HF_USERNAME,
        local_registry_path: Optional[str] = None,
        cache_ttl: int = _CACHE_TTL_SECONDS,
        force_refresh: bool = False,
    ) -> None:
        self.hf_username = hf_username
        self.cache_ttl = cache_ttl
        self._models: Dict[str, Dict[str, Any]] = {}

        # Try loading from cache first (unless force_refresh)
        if not force_refresh and self._load_from_cache():
            return

        # Try local registry file
        if local_registry_path:
            local_path = Path(local_registry_path)
        else:
            local_path = Path("registry") / "documents.yaml"

        if local_path.exists():
            self._load_from_local(local_path)

        # Always try Hub discovery to get the latest models
        self._discover_from_hub()

        # Save to cache
        self._save_to_cache()

    def _load_from_cache(self) -> bool:
        """Load registry from local cache file if fresh enough."""
        if not _CACHE_FILE.exists():
            return False

        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at > self.cache_ttl:
                return False

            self._models = data.get("models", {})
            return bool(self._models)
        except Exception:
            return False

    def _save_to_cache(self) -> None:
        """Save current registry state to local cache."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "cached_at": time.time(),
                "hf_username": self.hf_username,
                "models": self._models,
            }
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Cache write failure is non-critical

    def _load_from_local(self, path: Path) -> None:
        """Load doc types from local documents.yaml registry."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            documents = data.get("documents", {})
            for doc_type, info in documents.items():
                if doc_type not in self._models:
                    self._models[doc_type] = {
                        "doc_type": doc_type,
                        "name": info.get("name", doc_type.replace("_", " ").title()),
                        "repo_id": f"{self.hf_username}/phobert-{doc_type}-ner",
                        "templates_path": info.get("templates", ""),
                        "schema_path": info.get("schema", ""),
                        "source": "local_registry",
                        "hub_verified": False,
                    }
        except Exception:
            pass

    def _discover_from_hub(self) -> None:
        """Discover models from HuggingFace Hub."""
        try:
            from huggingface_hub import list_models

            models = list(list_models(author=self.hf_username, search="phobert-ner"))
            for model in models:
                repo_id = model.id if hasattr(model, "id") else str(model)
                name_part = repo_id.split("/")[-1]

                if not (name_part.startswith("phobert-") and name_part.endswith("-ner")):
                    continue

                doc_type = name_part[len("phobert-"):-len("-ner")]
                tags = model.tags if hasattr(model, "tags") else []

                # Update or create entry
                if doc_type in self._models:
                    self._models[doc_type]["hub_verified"] = True
                    self._models[doc_type]["repo_id"] = repo_id
                    self._models[doc_type]["tags"] = tags
                    self._models[doc_type]["source"] = "hub"
                else:
                    self._models[doc_type] = {
                        "doc_type": doc_type,
                        "name": doc_type.replace("_", " ").title(),
                        "repo_id": repo_id,
                        "tags": tags,
                        "source": "hub",
                        "hub_verified": True,
                    }
        except Exception:
            pass

    def list_doc_types(self) -> List[str]:
        """List all known document types.

        Returns:
            Sorted list of document type identifiers.
        """
        return sorted(self._models.keys())

    def get_model_info(self, doc_type: str) -> Optional[Dict[str, Any]]:
        """Get model info for a specific document type.

        Args:
            doc_type: Document type identifier.

        Returns:
            Dict with model info, or None if not found.
        """
        return self._models.get(doc_type)

    def get_repo_id(self, doc_type: str) -> Optional[str]:
        """Get HuggingFace Hub repo ID for a document type.

        Args:
            doc_type: Document type identifier.

        Returns:
            Repo ID string, or None if not found.
        """
        info = self._models.get(doc_type)
        return info["repo_id"] if info else None

    def is_available(self, doc_type: str) -> bool:
        """Check if a document type has a trained model available.

        Args:
            doc_type: Document type identifier.

        Returns:
            True if model exists (verified on Hub or in local registry).
        """
        info = self._models.get(doc_type)
        if info is None:
            return False
        return info.get("hub_verified", False) or info.get("source") == "local_registry"

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models with full info.

        Returns:
            List of model info dicts, sorted by doc_type.
        """
        return [self._models[dt] for dt in sorted(self._models.keys())]

    def refresh(self) -> None:
        """Force refresh the registry from HuggingFace Hub."""
        self._models.clear()

        # Re-discover
        local_path = Path("registry") / "documents.yaml"
        if local_path.exists():
            self._load_from_local(local_path)
        self._discover_from_hub()
        self._save_to_cache()

    def register_model(
        self,
        doc_type: str,
        repo_id: str,
        name: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Manually register a model (used by CI/CD after training).

        Args:
            doc_type: Document type identifier.
            repo_id: HuggingFace Hub repo ID.
            name: Human-readable name.
            **extra: Additional metadata.
        """
        self._models[doc_type] = {
            "doc_type": doc_type,
            "name": name or doc_type.replace("_", " ").title(),
            "repo_id": repo_id,
            "source": "manual",
            "hub_verified": True,
            **extra,
        }
        self._save_to_cache()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to a dict (for JSON export).

        Returns:
            Dict with 'models' key containing all model info.
        """
        return {
            "hf_username": self.hf_username,
            "models": self._models,
            "doc_types": self.list_doc_types(),
        }
