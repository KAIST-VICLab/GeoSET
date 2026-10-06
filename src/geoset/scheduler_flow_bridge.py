"""Explicit-Euler solver for the GeoSET flow bridge (t = 0: Gaussian noise, t = 1: EO latent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class FlowBridgeSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class FlowBridgeScheduler(SchedulerMixin, ConfigMixin):
    """Uniform grid ``linspace(0, 1, N + 1)``; the model is queried at the first ``N`` points, ``t`` ascending.

    ``step`` integrates the velocity ``dz/dt`` over one grid interval in float32.
    """

    order = 1

    @register_to_config
    def __init__(self):
        self.grid: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.step_index = 0

    @property
    def timesteps(self) -> torch.Tensor:
        return self.grid[:-1]

    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[str, torch.device]] = None) -> None:
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        self.num_inference_steps = num_inference_steps
        self.grid = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        self.step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowBridgeSchedulerOutput, tuple]:
        """``z + (t_next - t) * v``; steps must follow ``timesteps`` in order."""
        t, t_next = self.grid[self.step_index], self.grid[self.step_index + 1]
        if float(timestep) != float(t):
            raise ValueError(f"expected timestep {float(t)} at step {self.step_index}, got {float(timestep)}")
        prev_sample = sample.float() + (t_next - t) * model_output.float()
        self.step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowBridgeSchedulerOutput(prev_sample=prev_sample)
