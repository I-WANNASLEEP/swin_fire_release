"""Batch-size-invariant learning-rate schedules for the JEI experiments."""

from __future__ import annotations

import math
from typing import Any


def cosine_restart_position(
    completed_epochs: int,
    *,
    t0_epochs: int,
    t_mult: int,
) -> tuple[int, int, int]:
    """Return ``(cycle_index, epoch_in_cycle, cycle_length)``.

    ``completed_epochs`` is the number of fully completed training epochs.
    A completed epoch exactly on a cycle boundary starts the next cycle.  The
    function intentionally knows nothing about batches, samples, or world size.
    """

    if completed_epochs < 0:
        raise ValueError("completed_epochs must be non-negative.")
    if t0_epochs <= 0:
        raise ValueError("t0_epochs must be positive.")
    if t_mult < 1:
        raise ValueError("t_mult must be at least 1.")

    remaining = int(completed_epochs)
    cycle_index = 0
    cycle_length = int(t0_epochs)
    while remaining >= cycle_length:
        remaining -= cycle_length
        cycle_index += 1
        cycle_length *= int(t_mult)
    return cycle_index, remaining, cycle_length


class EpochCosineAnnealingWarmRestartsWithDecay:
    """Cosine warm restarts whose peak decays once per completed cycle.

    The optimizer starts at its configured base learning rate.  Call
    :meth:`step` once after each validation pass with the one-based number of
    completed epochs.  The resulting learning rate is used by the next epoch.
    """

    step_unit = "epoch"

    def __init__(
        self,
        optimizer: Any,
        *,
        t0_epochs: int,
        t_mult: int = 1,
        decay_factor: float = 0.99,
        eta_min: float = 0.0,
    ) -> None:
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1].")
        if eta_min < 0.0:
            raise ValueError("eta_min must be non-negative.")
        cosine_restart_position(0, t0_epochs=t0_epochs, t_mult=t_mult)

        self.optimizer = optimizer
        self.t0_epochs = int(t0_epochs)
        self.t_mult = int(t_mult)
        self.decay_factor = float(decay_factor)
        self.eta_min = float(eta_min)
        self.initial_base_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if any(base_lr < self.eta_min for base_lr in self.initial_base_lrs):
            raise ValueError("Every base learning rate must be >= eta_min.")
        self.completed_epochs = 0
        self._last_lr = list(self.initial_base_lrs)

    def _learning_rates(self, completed_epochs: int) -> list[float]:
        cycle_index, epoch_in_cycle, cycle_length = cosine_restart_position(
            completed_epochs,
            t0_epochs=self.t0_epochs,
            t_mult=self.t_mult,
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * epoch_in_cycle / cycle_length)
        )
        return [
            self.eta_min
            + (
                base_lr * (self.decay_factor**cycle_index)
                - self.eta_min
            )
            * cosine
            for base_lr in self.initial_base_lrs
        ]

    def step(self, completed_epochs: int | None = None) -> None:
        if completed_epochs is None:
            completed_epochs = self.completed_epochs + 1
        completed_epochs = int(completed_epochs)
        if completed_epochs < self.completed_epochs:
            raise ValueError("Scheduler epochs must be monotonic.")
        learning_rates = self._learning_rates(completed_epochs)
        for param_group, learning_rate in zip(
            self.optimizer.param_groups, learning_rates
        ):
            param_group["lr"] = learning_rate
        self.completed_epochs = completed_epochs
        self._last_lr = learning_rates

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "t0_epochs": self.t0_epochs,
            "t_mult": self.t_mult,
            "decay_factor": self.decay_factor,
            "eta_min": self.eta_min,
            "initial_base_lrs": list(self.initial_base_lrs),
            "completed_epochs": self.completed_epochs,
            "last_lr": list(self._last_lr),
            "step_unit": self.step_unit,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        required = {
            "t0_epochs",
            "t_mult",
            "decay_factor",
            "eta_min",
            "initial_base_lrs",
            "completed_epochs",
        }
        missing = required - set(state_dict)
        if missing:
            raise ValueError(
                f"Scheduler state misses required fields: {sorted(missing)}"
            )
        if int(state_dict["t0_epochs"]) != self.t0_epochs:
            raise ValueError("Scheduler t0_epochs differs from checkpoint.")
        if int(state_dict["t_mult"]) != self.t_mult:
            raise ValueError("Scheduler t_mult differs from checkpoint.")
        if float(state_dict["decay_factor"]) != self.decay_factor:
            raise ValueError("Scheduler decay_factor differs from checkpoint.")
        if float(state_dict["eta_min"]) != self.eta_min:
            raise ValueError("Scheduler eta_min differs from checkpoint.")
        base_lrs = [float(value) for value in state_dict["initial_base_lrs"]]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler parameter-group count differs.")
        self.initial_base_lrs = base_lrs
        self.completed_epochs = int(state_dict["completed_epochs"])
        self._last_lr = self._learning_rates(self.completed_epochs)
        for param_group, learning_rate in zip(
            self.optimizer.param_groups, self._last_lr
        ):
            param_group["lr"] = learning_rate
