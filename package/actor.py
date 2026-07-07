import torch
from torch import nn


class DistributionActor(nn.Module):
    atoms: torch.Tensor

    def __init__(self, num_atoms: int = 51, v_min: float | int = -10.0, v_max: float | int = 10.0):
        super().__init__()
        self.register_buffer("atoms", torch.linspace(v_min, v_max, num_atoms))

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        expected_q_value: torch.Tensor = (torch.softmax(logits, dim=-1) * self.atoms).sum(dim=-1)
        action: torch.Tensor = expected_q_value.argmax(dim=-1)
        return action
