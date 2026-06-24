from __future__ import annotations

import torch


def mlu_is_available() -> bool:
    try:
        import torch_mlu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "mlu") and torch.mlu.is_available()


def get_mlu_device_count() -> int:
    if not mlu_is_available():
        return 0
    get_count = getattr(torch.mlu, "device_count", None)
    return int(get_count()) if callable(get_count) else 1


def _parse_index(device: str, prefix: str) -> int:
    suffix = device[len(prefix) :].strip()
    if suffix.startswith(":"):
        suffix = suffix[1:]
    if not suffix:
        return 0
    if "," in suffix:
        raise ValueError("This training entry supports one device at a time.")
    if not suffix.isdigit():
        raise ValueError(f"Invalid device index: {device}")
    return int(suffix)


def resolve_device(device: str | None = "auto") -> torch.device:
    normalized = str(device or "auto").strip().lower()

    if normalized == "auto":
        if mlu_is_available():
            normalized = "mlu:0"
        elif torch.cuda.is_available():
            normalized = "cuda:0"
        else:
            normalized = "cpu"

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized in {"cuda", "gpu"} or normalized.isdigit():
        normalized = f"cuda:{0 if normalized in {'cuda', 'gpu'} else int(normalized)}"

    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but CUDA is not available.")
        index = _parse_index(normalized, "cuda")
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise RuntimeError(f"CUDA device index out of range: {index}. Available devices: {count}.")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    if normalized in {"mlu", "cambricon"}:
        normalized = "mlu:0"

    if normalized.startswith("mlu:"):
        if not mlu_is_available():
            raise RuntimeError("MLU device was requested, but torch_mlu is not available.")
        index = _parse_index(normalized, "mlu")
        count = get_mlu_device_count()
        if index < 0 or index >= count:
            raise RuntimeError(f"MLU device index out of range: {index}. Available devices: {count}.")
        set_device = getattr(torch.mlu, "set_device", None)
        if callable(set_device):
            set_device(index)
        return torch.device(f"mlu:{index}")

    raise ValueError(f"Unsupported device: {device}")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device.index or 0)
        total_gb = props.total_memory / 1024**3
        return f"device=cuda:{device.index or 0} name={props.name} total={total_gb:.1f}GB"

    if device.type == "mlu":
        index = device.index or 0
        get_name = getattr(torch.mlu, "get_device_name", None)
        name = get_name(index) if callable(get_name) else "Cambricon MLU"
        return f"device=mlu:{index} name={name}"

    return "device=cpu"
