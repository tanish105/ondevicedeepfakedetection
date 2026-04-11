from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class RuntimeContext:
    requested_device: str
    resolved_device: str
    use_cuda: bool
    use_mixed_precision: bool


def resolve_device(device: str = "auto") -> RuntimeContext:
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device setting: {device}")

    has_cuda = torch.cuda.is_available()
    if normalized == "cpu":
        resolved = "cpu"
    elif normalized == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA was requested but is not available in this environment")
        resolved = "cuda"
    else:
        resolved = "cuda" if has_cuda else "cpu"

    return RuntimeContext(
        requested_device=normalized,
        resolved_device=resolved,
        use_cuda=resolved == "cuda",
        use_mixed_precision=resolved == "cuda",
    )
