import torch
from torch import nn
import torch.nn.functional as nf
from torchrl.modules import NoisyLazyLinear
from functools import wraps
from typing import Callable


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def reset_noise(net: nn.Module) -> None:
    for module in net.modules():
        if isinstance(module, Rainbow):
            module.reset_noise()


def auto_batching(unbatched_ndim: int = 1, keep_ndim: bool = False):
    def decorator(function: Callable[[nn.Module, torch.Tensor], torch.Tensor]):
        @wraps(function)
        def wrapper(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            assert x.ndim in (unbatched_ndim, unbatched_ndim + 1), f"Arguments Compatibility Error..."
            is_batch: bool = (x.ndim == unbatched_ndim + 1)
            x: torch.Tensor = x.unsqueeze(0) if not is_batch else x
            output: torch.Tensor = function(self, x)
            if keep_ndim:
                if not is_batch:
                    output: torch.Tensor = output.squeeze(0)
            return output

        return wrapper

    return decorator


class ScaleModule(nn.Module):
    def __init__(self, factor: float | int = 1.0 / 255.0):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32) * self.factor


class BreakoutBackbone(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                         nn.ReLU(),
                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, stride=1),
                         nn.ReLU(),
                         nn.Flatten())
        self.apply(init_weights)

    @auto_batching(unbatched_ndim=3, keep_ndim=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(BreakoutBackbone, self).forward(x)


class InverseDynamicsHead(nn.Module):
    def __init__(self, n_actions: int, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([x, y, x - y, x * y], dim=-1)
        return self.mlp(x)


class InverseDynamicsModel(nn.Module):
    def __init__(self, n_actions: int, hidden: int = 512, scale: int | float | None = None):
        super().__init__()
        self.scale = ScaleModule(scale) if (scale is not None) else nn.Identity()
        self.encoder = BreakoutBackbone()
        self.head = InverseDynamicsHead(n_actions, hidden=hidden)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder(self.scale(x))
        y: torch.Tensor = self.encoder(self.scale(y))
        logits: torch.Tensor = self.head(x, y)
        return logits


class DuelingDistributionHead(nn.Module):
    def __init__(self, n_actions: int, n_atoms: int = 51):
        super().__init__()
        self.action_dim: int = n_actions
        self.num_atoms: int = n_atoms
        self.value_hidden = NoisyLazyLinear(512, std_init=0.5, use_exploration_type=False)
        self.value_output = NoisyLazyLinear(n_atoms, std_init=0.5, use_exploration_type=False)
        self.advantage_hidden = NoisyLazyLinear(512, std_init=0.5, use_exploration_type=False)
        self.advantage_output = NoisyLazyLinear(self.action_dim * self.num_atoms, std_init=0.5,
                                                use_exploration_type=False)

    def reset_noise(self):
        for layer in self.children():
            layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------------------------
        # Распределение ценности (V)
        value = nf.relu(self.value_hidden(x))
        value = self.value_output(value).view(-1, 1, self.num_atoms)
        # ----------------------------
        # Распределение преимущества (A)
        advantage = nf.relu(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage).view(-1, self.action_dim, self.num_atoms)
        logits: torch.Tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return logits


class Rainbow(nn.Module):
    def __init__(self, backbone: nn.Module, dueling: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.dueling = dueling

    @auto_batching(unbatched_ndim=3, keep_ndim=True)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.backbone(state)
        logits: torch.Tensor = self.dueling(features)
        return logits

    def reset_noise(self):
        self.dueling.reset_noise()
