from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import ModelFingerprint
from .errors import CompatibilityError, ConfigurationError


@dataclass(frozen=True)
class CompatibilityReport:
    sender: ModelFingerprint
    receiver: ModelFingerprint
    compatible: bool
    mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mismatches": list(self.mismatches),
            "sender": asdict(self.sender),
            "receiver": asdict(self.receiver),
        }

    def require(self) -> None:
        if not self.compatible:
            raise CompatibilityError(
                "Sender and receiver are not valid for DroidSpeak cache reuse:\n- "
                + "\n- ".join(self.mismatches)
            )


def _require_dir(path: str, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"{role} model path does not exist or is not a directory: {resolved}")
    return resolved


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "length": len(tokenizer),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _checkpoint_signature(model_path: Path) -> str:
    """Cheap mutation detector for a local checkpoint directory.

    It intentionally hashes names, sizes and nanosecond mtimes rather than the
    multi-GB weight contents, so capture refuses obviously changed checkpoints
    without adding hours of preprocessing to every experimental run.
    """
    interesting = []
    prefixes = ("model", "pytorch_model", "adapter", "config", "generation_config")
    for item in model_path.iterdir():
        if item.is_file() and item.name.startswith(prefixes):
            stat = item.stat()
            interesting.append((item.name, stat.st_size, stat.st_mtime_ns))
    if not interesting:
        raise ConfigurationError(f"No checkpoint/config files found in model directory: {model_path}")
    return hashlib.sha256(json.dumps(sorted(interesting), separators=(",", ":")).encode("utf-8")).hexdigest()


def inspect_model(path: str, tokenizer_path: str, trust_remote_code: bool = False) -> ModelFingerprint:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("transformers is required: pip install transformers") from exc

    model_path = _require_dir(path, "model")
    tokenizer_dir = _require_dir(tokenizer_path, "tokenizer")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=True
    )

    hidden_size = int(getattr(config, "hidden_size", 0))
    attention_heads = int(getattr(config, "num_attention_heads", 0))
    head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads if attention_heads else 0))
    return ModelFingerprint(
        model_path=str(model_path),
        checkpoint_signature=_checkpoint_signature(model_path),
        model_type=str(getattr(config, "model_type", "")),
        architectures=tuple(getattr(config, "architectures", []) or []),
        num_layers=int(getattr(config, "num_hidden_layers", 0)),
        hidden_size=hidden_size,
        num_attention_heads=attention_heads,
        num_key_value_heads=int(getattr(config, "num_key_value_heads", attention_heads)),
        head_dim=head_dim,
        vocab_size=int(getattr(config, "vocab_size", 0)),
        rope_theta=getattr(config, "rope_theta", None),
        rope_scaling=getattr(config, "rope_scaling", None),
        tokenizer_fingerprint=_tokenizer_fingerprint(tokenizer),
    )


def validate_pair(sender: ModelFingerprint, receiver: ModelFingerprint) -> CompatibilityReport:
    fields = (
        "model_type",
        "architectures",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rope_theta",
        "rope_scaling",
        "tokenizer_fingerprint",
    )
    mismatches = tuple(
        f"{field}: sender={getattr(sender, field)!r}, receiver={getattr(receiver, field)!r}"
        for field in fields
        if getattr(sender, field) != getattr(receiver, field)
    )
    return CompatibilityReport(sender, receiver, not mismatches, mismatches)
