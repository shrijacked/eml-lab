"""Exact and training-safe EML operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

COMPLEX_DTYPE = torch.complex128
REAL_DTYPE = torch.float64


@dataclass(frozen=True)
class StabilityConfig:
    """Numerical guardrails for training only.

    The exact verifier never uses these guardrails. They exist to keep the optimizer
    from falling into NaN-heavy regions while it is still exploring soft mixtures.
    """

    exp_real_clip: float = 18.0
    imag_clip: float = 40.0
    log_min_abs: float = 1e-10
    output_real_clip: float = 1e12
    output_imag_clip: float = 1e12


@dataclass(frozen=True)
class StabilityStats:
    """Counts of guardrail activations during one EML call."""

    exp_clipped: int = 0
    imag_clipped: int = 0
    log_nudged: int = 0
    output_clipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def as_complex_tensor(value: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
    """Convert a value to complex128, preserving device when a reference tensor is given."""

    if isinstance(value, torch.Tensor):
        tensor = value
        if like is not None:
            tensor = tensor.to(device=like.device)
        return tensor.to(dtype=COMPLEX_DTYPE)
    device = like.device if like is not None else None
    return torch.as_tensor(value, dtype=COMPLEX_DTYPE, device=device)


def eml_exact(x: Any, y: Any) -> torch.Tensor:
    """Faithful EML operator: exp(x) - log(y), evaluated as complex128."""

    x_tensor = as_complex_tensor(x)
    y_tensor = as_complex_tensor(y, like=x_tensor)
    return torch.exp(x_tensor) - torch.log(y_tensor)


def _clip_complex(
    value: torch.Tensor, real_clip: float, imag_clip: float
) -> tuple[torch.Tensor, int]:
    real = value.real
    imag = value.imag
    clipped_real = real.clamp(-real_clip, real_clip)
    clipped_imag = imag.clamp(-imag_clip, imag_clip)
    changed = torch.count_nonzero((clipped_real != real) | (clipped_imag != imag)).item()
    return torch.complex(clipped_real, clipped_imag), int(changed)


def _nudge_away_from_zero(value: torch.Tensor, min_abs: float) -> tuple[torch.Tensor, int]:
    magnitude = torch.abs(value)
    mask = magnitude < min_abs
    safe = torch.where(
        mask, value + torch.as_tensor(min_abs, dtype=COMPLEX_DTYPE, device=value.device), value
    )
    return safe, int(torch.count_nonzero(mask).item())


def eml_train(
    x: Any,
    y: Any,
    stability_config: StabilityConfig | None = None,
    *,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, StabilityStats]:
    """Training-safe EML helper.

    This is deliberately labeled as a training helper. It may clip or nudge values.
    Use :func:`eml_exact` for final verification.
    """

    config = stability_config or StabilityConfig()
    x_tensor = as_complex_tensor(x)
    y_tensor = as_complex_tensor(y, like=x_tensor)

    x_safe, exp_clipped = _clip_complex(x_tensor, config.exp_real_clip, config.imag_clip)
    y_safe, log_nudged = _nudge_away_from_zero(y_tensor, config.log_min_abs)
    y_safe, imag_clipped = _clip_complex(y_safe, config.output_real_clip, config.imag_clip)

    output = torch.exp(x_safe) - torch.log(y_safe)
    output_safe, output_clipped = _clip_complex(
        output,
        config.output_real_clip,
        config.output_imag_clip,
    )
    stats = StabilityStats(
        exp_clipped=exp_clipped,
        imag_clipped=imag_clipped,
        log_nudged=log_nudged,
        output_clipped=output_clipped,
    )
    if return_stats:
        return output_safe, stats
    return output_safe
