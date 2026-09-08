from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True)
class ModelConfig:
    sender_path: str
    receiver_path: str
    tokenizer_path: str
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    device_sender: str = "cuda:0"
    device_receiver: str = "cuda:0"


@dataclass(frozen=True)
class DataConfig:
    calibration_jsonl: str
    evaluation_jsonl: str
    max_prompt_tokens: int = 8192
    max_new_tokens: int = 20
    prompt_suffix_tokens: int = 1
    input_format: str = "plain"


@dataclass(frozen=True)
class CacheConfig:
    root: str
    capture_transition_layers: str | list[int] = "all"
    storage_dtype: str = "bfloat16"
    overwrite: bool = False


@dataclass(frozen=True)
class ProfilingConfig:
    recompute_lengths: str | list[int] = "all"
    min_recompute_layers: int = 1
    max_recompute_layers: int | None = None
    start_layers: str | list[int] = "all"
    quality_metric: str = "qa_f1"
    quality_tolerance_relative: float = 0.05
    measure_latency_repetitions: int = 5
    warmup_repetitions: int = 2
    max_profiles_per_run: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    selected_profile: str | None = None
    report_per_example: bool = True
    save_generations: bool = True
    collect_cache_error: bool = True
    collect_transfer_bytes: bool = True


@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    seed: int
    output_dir: str
    deterministic: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    root_dir: Path
    experiment: ExperimentMeta
    models: ModelConfig
    data: DataConfig
    cache: CacheConfig
    profiling: ProfilingConfig
    evaluation: EvaluationConfig

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.experiment.output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.cache.root)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["root_dir"] = str(self.root_dir)
        return result

    def assert_model_paths_configured(self) -> None:
        unresolved = [
            name
            for name, value in {
                "sender_path": self.models.sender_path,
                "receiver_path": self.models.receiver_path,
                "tokenizer_path": self.models.tokenizer_path,
            }.items()
            if not value or "__SET_" in value
        ]
        if unresolved:
            raise ConfigurationError(
                "Model paths are intentionally unconfigured. Set absolute paths for: "
                + ", ".join(unresolved)
            )


def _as_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Expected mapping for '{field_name}', got {type(raw).__name__}")
    return raw


def _build(cls: type, raw: Any, field_name: str):
    values = _as_mapping(raw, field_name)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid '{field_name}' section: {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Experiment config does not exist: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("PyYAML is required: pip install PyYAML") from exc

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _as_mapping(raw, "root")
    return ExperimentConfig(
        source_path=path,
        root_dir=path.parent.parent,
        experiment=_build(ExperimentMeta, raw.get("experiment"), "experiment"),
        models=_build(ModelConfig, raw.get("models"), "models"),
        data=_build(DataConfig, raw.get("data"), "data"),
        cache=_build(CacheConfig, raw.get("cache"), "cache"),
        profiling=_build(ProfilingConfig, raw.get("profiling"), "profiling"),
        evaluation=_build(EvaluationConfig, raw.get("evaluation"), "evaluation"),
    )
