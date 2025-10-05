"""Utility helpers for constructing optimizers from configuration."""

from typing import Any, Dict, Iterable, Type
import inspect

import torch.optim as optim

from deepfake.config.schema import OptimizerConfig


def _canonicalize(name: str) -> str:
    """Normalize an optimizer name: case-insensitive, ignore punctuation.

    Examples:
      "AdamW" / "adam_w" / "adam-w" -> "adamw"
      "SGD" / "stochasticgradientdescent" -> "sgd"
    """
    key = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    aliases = {
        "stochasticgradientdescent": "sgd",
    }
    return aliases.get(key, key)


_OPTIMIZER_REGISTRY: Dict[str, Type[optim.Optimizer]] = {
    # Common
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    # Additional built-ins
    "adadelta": optim.Adadelta,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "rprop": optim.Rprop,
    "lbfgs": optim.LBFGS,
}

# Conditionally register optimizers that may not exist in older torch versions
if hasattr(optim, "RAdam"):
    _OPTIMIZER_REGISTRY["radam"] = optim.RAdam  # type: ignore[attr-defined]
if hasattr(optim, "NAdam"):
    _OPTIMIZER_REGISTRY["nadam"] = optim.NAdam  # type: ignore[attr-defined]
if hasattr(optim, "SparseAdam"):
    _OPTIMIZER_REGISTRY["sparseadam"] = optim.SparseAdam  # type: ignore[attr-defined]


def _validate_and_prepare_hyperparams(
    optimizer_cls: Type[optim.Optimizer], hyperparams: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify provided hyperparameters against the optimizer signature.

    - Ensures unexpected keys are surfaced with a helpful error.
    - Returns a filtered dict that only includes supported kwargs.
    """
    sig = inspect.signature(optimizer_cls.__init__)
    valid_keys = {k for k in sig.parameters.keys() if k != "self"}
    # 'params' is supplied positionally by the caller; disallow overriding it via config.
    if "params" in hyperparams:
        raise ValueError(
            "Do not set 'params' inside training.optimizer.params; it is provided by the model."
        )
    unexpected = [k for k in hyperparams.keys() if k not in valid_keys]
    if unexpected:
        raise ValueError(
            f"Unexpected optimizer parameter(s) {unexpected} for {optimizer_cls.__name__}. "
            f"Valid keys: {sorted(valid_keys)}"
        )
    return {k: v for k, v in hyperparams.items() if k in valid_keys}


def create_optimizer(
    params: Iterable,
    optimizer_cfg: OptimizerConfig,
    *,
    default_lr: float,
):
    """Instantiate an optimizer from config; ensures lr fallback for compatibility."""
    name = _canonicalize(optimizer_cfg.name or "adam")
    optimizer_cls = _OPTIMIZER_REGISTRY.get(name)
    if optimizer_cls is None:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_cfg.name}'. "
            f"Available: {sorted(k for k, v in _OPTIMIZER_REGISTRY.items() if v is not None)}"
        )

    hyperparams: Dict[str, Any] = dict(optimizer_cfg.params or {})
    hyperparams.setdefault("lr", default_lr)
    # Validate keys for nicer errors
    hyperparams = _validate_and_prepare_hyperparams(optimizer_cls, hyperparams)

    return optimizer_cls(params, **hyperparams)


def optimizer_requires_closure(optimizer: optim.Optimizer) -> bool:
    """Return True if the optimizer .step() requires a closure (e.g., LBFGS)."""
    return isinstance(optimizer, optim.LBFGS)
