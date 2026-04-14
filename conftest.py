"""Root conftest.py — stub heavy dependencies not available in test environments."""
from __future__ import annotations

import sys
import types


def _make_torch_stub() -> types.ModuleType:
    """Return a minimal torch stub so src.common.runtime can be imported without PyTorch."""
    torch = types.ModuleType("torch")

    class _CudaStub:
        def is_available(self) -> bool:
            return False

    class _TensorStub:
        pass

    torch.cuda = _CudaStub()  # type: ignore[attr-defined]
    torch.Tensor = _TensorStub  # type: ignore[attr-defined]
    return torch


# Only install the stub when torch is genuinely absent.
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401 — real torch is present, nothing to do
    except ModuleNotFoundError:
        sys.modules["torch"] = _make_torch_stub()
