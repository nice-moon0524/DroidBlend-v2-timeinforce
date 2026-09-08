from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ArtifactError


SCHEMA_VERSION = 1


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temp_name = handle.name
    os.replace(temp_name, path)


@dataclass(frozen=True)
class ModelFingerprint:
    model_path: str
    checkpoint_signature: str
    model_type: str
    architectures: tuple[str, ...]
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float | None
    rope_scaling: Any
    tokenizer_fingerprint: str

    @property
    def digest(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    sender: ModelFingerprint
    receiver: ModelFingerprint
    common_tokenizer_path: str
    transition_layers: tuple[int, ...]
    prompt_suffix_tokens: int
    storage_dtype: str
    examples: tuple[dict[str, Any], ...]

    @property
    def digest(self) -> str:
        return stable_hash(asdict(self))

    def write(self, path: Path) -> None:
        atomic_json_write(path, asdict(self))

    @staticmethod
    def read(path: Path) -> "CacheManifest":
        if not path.is_file():
            raise ArtifactError(f"Cache manifest is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(f"Unsupported cache manifest version: {raw.get('schema_version')}")
        return CacheManifest(
            schema_version=raw["schema_version"],
            sender=ModelFingerprint(**raw["sender"]),
            receiver=ModelFingerprint(**raw["receiver"]),
            common_tokenizer_path=raw["common_tokenizer_path"],
            transition_layers=tuple(raw["transition_layers"]),
            prompt_suffix_tokens=raw["prompt_suffix_tokens"],
            storage_dtype=raw["storage_dtype"],
            examples=tuple(raw["examples"]),
        )


def require_same_fingerprint(expected: ModelFingerprint, observed: ModelFingerprint, role: str) -> None:
    if expected.digest != observed.digest:
        raise ArtifactError(
            f"{role} model fingerprint differs from cache manifest. "
            "Do not reuse caches across a changed checkpoint/config/tokenizer."
        )
