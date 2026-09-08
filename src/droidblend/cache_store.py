from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import CacheManifest
from .errors import ArtifactError


@dataclass
class SenderPrefixCache:
    """CPU-resident sender artifact for one exact tokenized prefix."""

    input_ids: Any
    attention_mask: Any
    legacy_kv: list[tuple[Any, Any]]
    e_caches: dict[int, Any]


class SenderCacheStore:
    """Disk layout that keeps every prompt independent and manifest-addressable."""

    def __init__(self, root: str | Path, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.overwrite = overwrite

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def initialize(self, manifest: CacheManifest) -> None:
        if self.root.exists() and self.manifest_path.exists() and not self.overwrite:
            existing = CacheManifest.read(self.manifest_path)
            if existing.digest != manifest.digest:
                raise ArtifactError(
                    f"Cache root already contains another experiment: {self.root}. "
                    "Use a new output directory or set cache.overwrite=true."
                )
            return
        if self.root.exists() and self.overwrite:
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest.write(self.manifest_path)

    def example_dir(self, identifier: str) -> Path:
        safe = identifier.replace("/", "_").replace("\\", "_")
        return self.root / "examples" / safe

    def exists(self, identifier: str) -> bool:
        directory = self.example_dir(identifier)
        return (directory / "complete.json").is_file()

    def save(self, identifier: str, payload: SenderPrefixCache, metadata: dict[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("torch is required to write sender caches") from exc

        destination = self.example_dir(identifier)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            torch.save(payload.input_ids, temp_dir / "input_ids.pt")
            torch.save(payload.attention_mask, temp_dir / "attention_mask.pt")
            torch.save(payload.legacy_kv, temp_dir / "kv_cache.pt")
            for layer, hidden in payload.e_caches.items():
                torch.save(hidden, temp_dir / f"e_cache_layer_{layer}.pt")
            with (temp_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            with (temp_dir / "complete.json").open("w", encoding="utf-8") as handle:
                json.dump({"ok": True}, handle)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temp_dir, destination)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def load(self, identifier: str, layers: list[int] | None = None, map_location: str = "cpu") -> SenderPrefixCache:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("torch is required to load sender caches") from exc
        directory = self.example_dir(identifier)
        if not (directory / "complete.json").is_file():
            raise ArtifactError(f"Sender cache is incomplete or missing for example '{identifier}'")
        selected = layers if layers is not None else [
            int(path.stem.rsplit("_", 1)[1]) for path in directory.glob("e_cache_layer_*.pt")
        ]
        e_caches = {
            layer: torch.load(directory / f"e_cache_layer_{layer}.pt", map_location=map_location, weights_only=True)
            for layer in selected
        }
        return SenderPrefixCache(
            input_ids=torch.load(directory / "input_ids.pt", map_location=map_location, weights_only=True),
            attention_mask=torch.load(directory / "attention_mask.pt", map_location=map_location, weights_only=True),
            legacy_kv=torch.load(directory / "kv_cache.pt", map_location=map_location, weights_only=True),
            e_caches=e_caches,
        )
