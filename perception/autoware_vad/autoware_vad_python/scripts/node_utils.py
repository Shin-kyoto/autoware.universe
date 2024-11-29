from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
from typing_extensions import Self


@dataclass
class ModelInput:
    uuids: list[str]
    actor: NDArray
    lane: NDArray
    rpe: NDArray
    rpe_mask: NDArray | None = None

    def cuda(self, device: int | torch.device | None = None) -> Self:
        self.actor = torch.from_numpy(self.actor).cuda(device)
        self.lane = torch.from_numpy(self.lane).cuda(device)
        self.rpe = torch.from_numpy(self.rpe).cuda(device)
        if self.rpe_mask is not None:
            self.rpe_mask = torch.from_numpy(self.rpe_mask).cuda(device)

        return self


def softmax(x: NDArray, axis: int) -> NDArray:
    """Apply softmax.

    Args:
        x (NDArray): Input array.
        axis (int): Axis to apply softmax.

    Returns:
        NDArray: Softmax result.
    """
    x -= x.max(axis=axis, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=axis, keepdims=True)
