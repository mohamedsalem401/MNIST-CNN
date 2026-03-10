from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 0
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081
    train_size: int = 55000
    val_size: int = 5000
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0


@dataclass
class ModelConfig:
    architecture: str = "mlp"  # mlp | cnn
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    latent_dim: int = 128
    num_classes: int = 10
    dropout: float = 0.0
    extra_parametric_layer: bool = False


@dataclass
class OptimConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9
    epochs: int = 8
    grad_clip_norm: float = 0.0


@dataclass
class RetrievalConfig:
    k: int = 8
    metric: str = "euclidean"  # euclidean | cosine
    weighting: str = "inverse_distance"  # uniform | inverse_distance | softmax
    temperature: float = 0.1
    normalize_keys: bool = False
    normalize_queries: bool = False
    normalize_values: bool = False
    class_conditional: bool = False
    filter_same_label: bool = False


@dataclass
class RefreshConfig:
    enabled: bool = False
    rate: float = 0.2
    match_distance_threshold: float = 0.5
    ema_key: float = 0.9
    ema_value: float = 0.9


@dataclass
class ForgettingConfig:
    policy: str = "none"  # none | ttl | fifo | reservoir | usage | helpfulness | helpfulness_age
    memory_size: int = 0
    ttl_steps: int = 0
    helpfulness_age_weight: float = 0.5


@dataclass
class TargetConfig:
    value_type: str = "delta"  # absolute | delta
    target_construction: str = "grad_delta"  # current_hidden | grad_delta | grad_target | class_centroid
    gradient_step_size: float = 0.1
    centroid_momentum: float = 0.95


@dataclass
class InterventionConfig:
    enabled: bool = True
    layer: str = "hidden_1"
    mode: str = "gated"  # overwrite | residual | gated
    affected_dims: int = 64
    affected_selection: str = "first"  # first | random
    query_mode: str = "full"  # full | untouched | projection
    query_projection_dims: int = 64
    gate_alpha: float = 0.4
    gate_schedule: str = "constant"  # constant | linear_warmup
    warmup_epochs: int = 1
    training_use: bool = True
    inference_use: bool = True
    populate_during_warmup: bool = True
    refresh: RefreshConfig = field(default_factory=RefreshConfig)


@dataclass
class MemoryConfig:
    enabled: bool = True
    insertion_policy: str = "always"  # always | loss_above_mean
    random_memory_control: bool = False
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)


@dataclass
class EvalConfig:
    compute_calibration: bool = True
    ece_bins: int = 15
    run_memory_toggle_eval: bool = True
    confusion_matrix: bool = True
    per_class_metrics: bool = True
    max_samples_for_latent_vis: int = 2000


@dataclass
class LoggingConfig:
    output_root: str = "results"
    experiment_name: str = "default"
    log_every_steps: int = 100
    save_checkpoints: bool = True
    save_memory_snapshots: bool = True


@dataclass
class ExperimentConfig:
    method: str = "hybrid"  # nn | nn_large | nn_extra_layer | embedding_knn | hybrid | hybrid_inactive | random_memory
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42])
    device: str = ""
    deterministic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigError(RuntimeError):
    pass


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_dataclass(datacls, values: Dict[str, Any]):
    fields = {f.name: f.type for f in datacls.__dataclass_fields__.values()}
    kwargs = {}
    for name, field_type in fields.items():
        if name not in values:
            continue
        value = values[name]
        default_obj = getattr(datacls(), name)
        if hasattr(default_obj, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _build_dataclass(type(default_obj), value)
        else:
            kwargs[name] = value
    return datacls(**kwargs)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none" or lowered == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        if value.startswith("[") and value.endswith("]"):
            return yaml.safe_load(value)
        return value


def _set_by_dotted_key(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(config_dict: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    if not overrides:
        return config_dict
    patch: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override '{override}'. Use key=value format.")
        key, raw_value = override.split("=", 1)
        _set_by_dotted_key(patch, key.strip(), _parse_scalar(raw_value.strip()))
    return _deep_update(config_dict, patch)


def load_config(path: str | Path, overrides: Optional[Sequence[str]] = None) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    merged = _deep_update(asdict(ExperimentConfig()), data)
    merged = apply_overrides(merged, overrides or [])
    return _build_dataclass(ExperimentConfig, merged)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)
