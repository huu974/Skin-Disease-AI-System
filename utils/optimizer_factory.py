"""Factory for configurable PyTorch optimizers."""

import importlib
import inspect
from typing import Iterable

import torch

from utils.optimizer_Adam import CustomAdam


_CUSTOM_OPTIMIZERS = {
    "adam": CustomAdam,
    "customadam": CustomAdam,
    "custom_adam": CustomAdam,
    "adam_custom": CustomAdam,
}

_OPTIMIZER_ALIASES = {
    "sgd+m": ("SGD", {"momentum": 0.9}),
    "sgdm": ("SGD", {"momentum": 0.9}),
    "sgd_m": ("SGD", {"momentum": 0.9}),
}

_LEGACY_PARAM_NAMES = ("momentum", "nesterov", "dampening")


def _normalized_key(value):
    return str(value).strip().lower().replace("-", "_")


def _compact_key(value):
    return _normalized_key(value).replace("_", "")


def _available_torch_optimizers():
    optimizers = []
    for name in dir(torch.optim):
        value = getattr(torch.optim, name)
        if isinstance(value, type) and issubclass(value, torch.optim.Optimizer) and value is not torch.optim.Optimizer:
            optimizers.append(name)
    return sorted(optimizers)


def _load_optimizer_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    optimizer_class = getattr(module, class_name)
    if not isinstance(optimizer_class, type) or not issubclass(optimizer_class, torch.optim.Optimizer):
        raise ValueError(f"Optimizer '{path}' is not a torch.optim.Optimizer subclass")
    return optimizer_class


def _resolve_optimizer(optimizer_name):
    raw_name = str(optimizer_name or "AdamW").strip()
    lower_name = raw_name.lower()
    normalized_name = _normalized_key(raw_name)
    compact_name = _compact_key(raw_name)

    alias_defaults = {}
    if lower_name in _OPTIMIZER_ALIASES:
        raw_name, alias_defaults = _OPTIMIZER_ALIASES[lower_name]
    elif normalized_name in _OPTIMIZER_ALIASES:
        raw_name, alias_defaults = _OPTIMIZER_ALIASES[normalized_name]

    if normalized_name in _CUSTOM_OPTIMIZERS:
        return _CUSTOM_OPTIMIZERS[normalized_name], alias_defaults
    if compact_name in _CUSTOM_OPTIMIZERS:
        return _CUSTOM_OPTIMIZERS[compact_name], alias_defaults

    if "." in raw_name:
        return _load_optimizer_from_path(raw_name), alias_defaults

    for name in _available_torch_optimizers():
        if name.lower() == raw_name.lower():
            return getattr(torch.optim, name), alias_defaults

    supported = ", ".join(_available_torch_optimizers() + sorted(_CUSTOM_OPTIMIZERS))
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported built-ins/custom aliases: {supported}")


def _supports_kwarg(optimizer_class, name):
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        return True

    parameters = signature.parameters
    if name in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _normalize_param_keys(params):
    return {str(key).replace("-", "_"): value for key, value in (params or {}).items()}


def _build_optimizer_kwargs(args, optimizer_class, alias_defaults):
    kwargs = {}

    if _supports_kwarg(optimizer_class, "lr") and getattr(args, "lr", None) is not None:
        kwargs["lr"] = args.lr
    if _supports_kwarg(optimizer_class, "weight_decay") and getattr(args, "weight_decay", None) is not None:
        kwargs["weight_decay"] = args.weight_decay

    kwargs.update(_normalize_param_keys(alias_defaults))

    for name in _LEGACY_PARAM_NAMES:
        value = getattr(args, name, None)
        if value is not None and _supports_kwarg(optimizer_class, name):
            kwargs[name] = value

    kwargs.update(_normalize_param_keys(getattr(args, "optimizer_params", None)))

    unsupported = [name for name in kwargs if not _supports_kwarg(optimizer_class, name)]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"{optimizer_class.__name__} does not accept optimizer parameter(s): {names}")

    return kwargs


def build_optimizer(parameters: Iterable[torch.nn.Parameter], args):
    optimizer_class, alias_defaults = _resolve_optimizer(getattr(args, "optimizer", "AdamW"))
    optimizer_kwargs = _build_optimizer_kwargs(args, optimizer_class, alias_defaults)
    optimizer = optimizer_class(parameters, **optimizer_kwargs)

    args.optimizer = optimizer_class.__name__
    args.optimizer_params = optimizer_kwargs
    if optimizer.param_groups and "lr" in optimizer.param_groups[0]:
        args.lr = optimizer.param_groups[0]["lr"]

    return optimizer


def optimizer_requires_closure(optimizer):
    try:
        signature = inspect.signature(optimizer.step)
    except (TypeError, ValueError):
        return False

    closure = signature.parameters.get("closure")
    return closure is not None and closure.default == inspect.Parameter.empty
