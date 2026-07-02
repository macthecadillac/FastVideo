# SPDX-License-Identifier: Apache-2.0
"""Gradient norm clipping callback.

Clips gradients on modules returned by
``method.get_grad_clip_targets()`` before the optimizer step.
Optionally logs per-module grad norms to the tracker.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.optimizer import (
    clip_grad_norm_if_needed, )

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class GradNormClipCallback(Callback):
    """Clip gradient norms before the optimizer step.

    ``max_grad_norm`` must be set explicitly in the callback
    config (``callbacks.grad_clip.max_grad_norm``).
    """

    def __init__(
        self,
        *,
        max_grad_norm: float = 1.0,
        log_grad_norms: bool = True,
        debug_log: bool = False,
        debug_log_steps: list[int] | None = None,
    ) -> None:
        self._max_grad_norm = float(max_grad_norm)
        self._log_grad_norms = bool(log_grad_norms)
        self._debug_log = bool(debug_log)
        self._debug_log_steps = ({int(step) for step in debug_log_steps} if debug_log_steps is not None else None)

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        max_norm = self._max_grad_norm
        if max_norm <= 0.0:
            return

        targets = method.get_grad_clip_targets(iteration)
        tracker = getattr(method, "tracker", None)

        for name, module in targets.items():
            should_debug_log = self._should_debug_log(iteration)
            if should_debug_log:
                self._log_clip_debug(
                    "begin",
                    name,
                    module,
                    iteration,
                )
                clip_t0 = time.perf_counter()
            grad_norm = clip_grad_norm_if_needed(
                module,
                max_norm,
            )
            if should_debug_log:
                self._log_clip_debug(
                    "end",
                    name,
                    module,
                    iteration,
                    grad_norm=grad_norm,
                    elapsed_sec=(time.perf_counter() - clip_t0),
                )
            if (self._log_grad_norms and tracker is not None and grad_norm > 0.0):
                tracker.log(
                    {f"grad_norm/{name}": grad_norm},
                    iteration,
                )

    def _should_debug_log(
        self,
        iteration: int,
    ) -> bool:
        if not self._debug_log:
            return False
        if self._debug_log_steps is None:
            return True
        return int(iteration) in self._debug_log_steps

    def _log_clip_debug(
        self,
        phase: str,
        name: str,
        module: torch.nn.Module,
        iteration: int,
        *,
        grad_norm: float | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        params = list(module.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        grad_types = sorted({type(g).__name__ for g in grads})
        grad_dtypes = sorted({str(g.dtype) for g in grads})
        grad_devices = sorted({str(g.device) for g in grads})
        logger.info(
            "Grad clip debug %s step=%s role=%s rank=%s "
            "params=%d grad_params=%d grad_types=%s grad_dtypes=%s "
            "grad_devices=%s grad_norm=%s elapsed_sec=%s",
            phase,
            iteration,
            name,
            self._rank(),
            len(params),
            len(grads),
            grad_types,
            grad_dtypes,
            grad_devices,
            grad_norm,
            elapsed_sec,
        )

    @staticmethod
    def _rank() -> int:
        if (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return int(torch.distributed.get_rank())
        return -1
