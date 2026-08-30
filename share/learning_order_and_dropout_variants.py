#!/usr/bin/env python3
"""Learning-order and gradient-to-dropout variants used in the City project.

This file is intentionally self-contained so it can be shared independently.
It contains:

1. Learning-order variants:
   normal, domain_reverse, temporal_reverse, random, reverse.
2. Gradient-to-dropout schedules:
   linear, inverse, logarithmic, plus the original exponential reference.
3. The exact three coefficient sets used by the experiments.

Notation for the non-exponential schedules
------------------------------------------
x       : gradient magnitude of the current item.
d_max   : maximum/reference gradient magnitude.
z       : normalized difficulty, clip(x / d_max, 0, 1).
s = 1-z : reversed difficulty; a larger s produces a larger dropout value.

The training code clips the final dropout probability to [0.0, 0.9].
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Literal


OrderMode = Literal[
    "normal",
    "domain_reverse",
    "temporal_reverse",
    "random",
    "reverse",
]
ScheduleKind = Literal["exp", "linear", "inverse", "log"]
TaskPair = tuple[int, int]  # (domain_index, temporal_task_index)


# Exact coefficient sets from tools/gradient_compute.py.
# coeff_idx selects row 0, 1, or 2.
COEFFICIENT_SETS: dict[str, tuple[dict[str, float], ...]] = {
    "linear": (
        {"a": 0.05, "b": 0.0},
        {"a": 0.10, "b": 0.0},
        {"a": 0.20, "b": 0.0},
    ),
    "inverse": (
        {"a": 0.02, "b": 0.0, "c": 0.40},
        {"a": 0.04, "b": 0.0, "c": 0.40},
        {"a": 0.08, "b": 0.0, "c": 0.40},
    ),
    "log": (
        {"a": 0.04, "b": 0.0, "c": 2.00},
        {"a": 0.08, "b": 0.0, "c": 2.00},
        {"a": 0.16, "b": 0.0, "c": 2.00},
    ),
}


def ordered_domain_indices(
    domain_count: int,
    mode: OrderMode = "normal",
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """Return the domain traversal order used in curriculum acquisition."""
    _validate_order_mode(mode)
    indices = list(range(domain_count))
    if mode in {"domain_reverse", "reverse"}:
        indices.reverse()
    elif mode == "random":
        (rng or random).shuffle(indices)
    return indices


def ordered_task_indices(
    task_count: int,
    mode: OrderMode = "normal",
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """Return the temporal-task order within one domain."""
    _validate_order_mode(mode)
    indices = list(range(task_count))
    if mode in {"temporal_reverse", "reverse"}:
        indices.reverse()
    elif mode == "random":
        (rng or random).shuffle(indices)
    return indices


def apply_source_order(
    pairs: Iterable[TaskPair],
    mode: OrderMode = "normal",
    *,
    rng: random.Random | None = None,
) -> list[TaskPair]:
    """Apply the experiment's order override to a list of (domain, task) pairs.

    The input may already be gradient-ranked. ``normal`` preserves that order,
    while the four ablations replace or reverse it as follows:

    - domain_reverse: descending domain, ascending task.
    - temporal_reverse: ascending domain, descending task.
    - reverse: reverse the complete existing list.
    - random: shuffle the complete existing list.
    """
    _validate_order_mode(mode)
    result = list(pairs)
    if mode == "domain_reverse":
        return sorted(result, key=lambda pair: (-pair[0], pair[1]))
    if mode == "temporal_reverse":
        return sorted(result, key=lambda pair: (pair[0], -pair[1]))
    if mode == "reverse":
        return list(reversed(result))
    if mode == "random":
        (rng or random).shuffle(result)
    return result


def dropout_schedule(
    kind: ScheduleKind,
    coeff_idx: int,
    x: float,
    d_max: float,
    *,
    p0: float = 0.1,
    clip_output: bool = True,
) -> float:
    """Map a gradient magnitude to a dropout probability.

    Formulas
    --------
    Let z = clip(x / d_max, 0, 1), and s = 1 - z.

    linear : p = a*s + b
    inverse: p = a/(z+c) + b
    log    : p = a*ln(1+c*s) + b

    The original exponential reference does not use ``coeff_idx``:
    exp    : p = p0 * (1 - exp(x-d_max))
    """
    if kind == "exp":
        value = p0 * (1.0 - math.exp(float(x) - float(d_max)))
        return _clip_dropout(value) if clip_output else value

    if kind not in COEFFICIENT_SETS:
        raise ValueError(f"unsupported schedule: {kind!r}")
    if coeff_idx not in (0, 1, 2):
        raise ValueError(f"coeff_idx must be 0, 1, or 2; got {coeff_idx}")

    safe_d_max = max(float(d_max), 1e-8)
    z = _clip(float(x) / safe_d_max, 0.0, 1.0)
    s = 1.0 - z
    params = COEFFICIENT_SETS[kind][coeff_idx]

    if kind == "linear":
        value = params["a"] * s + params["b"]
    elif kind == "inverse":
        value = params["a"] / (z + params["c"]) + params["b"]
    else:  # kind == "log"
        value = params["a"] * math.log1p(params["c"] * s) + params["b"]

    return _clip_dropout(value) if clip_output else value


def schedule_curve(
    kind: ScheduleKind,
    coeff_idx: int,
    normalized_gradients: Sequence[float],
) -> list[float]:
    """Convenience helper: evaluate a schedule with d_max=1."""
    return [dropout_schedule(kind, coeff_idx, z, 1.0) for z in normalized_gradients]


def _validate_order_mode(mode: str) -> None:
    valid = {"normal", "domain_reverse", "temporal_reverse", "random", "reverse"}
    if mode not in valid:
        raise ValueError(f"unsupported order mode: {mode!r}; expected one of {sorted(valid)}")


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _clip_dropout(value: float) -> float:
    return _clip(value, 0.0, 0.9)


def demo() -> None:
    """Print small reproducible examples when this file is run directly."""
    pairs: list[TaskPair] = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    print("Learning-order variants")
    for mode in ("normal", "domain_reverse", "temporal_reverse", "random", "reverse"):
        order = apply_source_order(pairs, mode, rng=random.Random(2025))
        print(f"  {mode:16s}: {order}")

    normalized_gradients = (0.0, 0.25, 0.5, 0.75, 1.0)
    print("\nGradient-to-dropout schedules (d_max=1)")
    for kind in ("linear", "inverse", "log"):
        for coeff_idx in range(3):
            values = schedule_curve(kind, coeff_idx, normalized_gradients)
            formatted = ", ".join(f"{value:.5f}" for value in values)
            print(f"  {kind:7s} coeff_idx={coeff_idx}: [{formatted}]")


if __name__ == "__main__":
    demo()
