import torch
from torch import nn
from copy import deepcopy
from td_view import Transition
from module import reset_noise


def categorical_projection(next_probs: torch.Tensor,
                           reward: torch.Tensor,
                           done: torch.Tensor,
                           gamma: torch.Tensor,
                           support: torch.Tensor,
                           v_min: float | int,
                           v_max: float | int) -> torch.Tensor:
    """ Проекция беллмановской цели обратно на фиксированную сетку атомов. """
    batch_size, n_atoms = next_probs.shape
    delta_z: float = (v_max - v_min) / (n_atoms - 1)  # шаг сетки между соседними атомами
    # Приводим всё к [B, 1], чтобы корректно бродкастить по оси атомов [1, N].
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    gamma = gamma.reshape(-1, 1)
    support = support.view(1, -1)  # [1, N]
    # --- Шаг 2.3: Bellman-сдвиг каждого атома z_i -> Tz_i = r + gamma^n * (1 - done) * z_i ---
    # [B,1] + [B,1] * [1,N] -> [B, N]. При done=1 остаётся только r (дельта в награде).
    tz: torch.Tensor = reward + (1.0 - done) * gamma * support
    tz = tz.clamp(v_min, v_max)  # не даём выйти за пределы линейки
    # --- Шаг 2.4a: переводим Tz в "дробный индекс" на сетке [0, N-1] ---
    b: torch.Tensor = (tz - v_min) / delta_z  # [B, N]
    l: torch.Tensor = b.floor().long()  # нижний сосед-узел
    u: torch.Tensor = b.ceil().long()  # верхний сосед-узел
    # Если Tz попал точно в узел (l == u), доли (u-b) и (b-l) обнулятся и масса "исчезнет".
    # Раздвигаем соседей, аккуратно обрабатывая края линейки (0 и N-1).
    l = torch.where((u > 0) & (l == u), l - 1, l)
    u = torch.where((l < n_atoms - 1) & (l == u), u + 1, u)
    # --- Шаг 2.4b: "размазываем" массу next_probs между узлами l и u (линейная интерполяция) ---
    weight_l: torch.Tensor = (u.float() - b)  # доля массы, уходящая в НИЖНИЙ узел l
    weight_u: torch.Tensor = (b - l.float())  # доля массы, уходящая в ВЕРХНИЙ узел u
    # scatter_add по плоскому тензору: смещаем индексы каждого примера на его строку (offset).
    m: torch.Tensor = torch.zeros(batch_size, n_atoms, device=next_probs.device, dtype=next_probs.dtype)
    offset: torch.Tensor = (torch.arange(batch_size, device=next_probs.device) * n_atoms).unsqueeze(1)
    m_flat: torch.Tensor = m.view(-1)
    m_flat.scatter_add_(0, (l + offset).view(-1), (next_probs * weight_l).view(-1))
    m_flat.scatter_add_(0, (u + offset).view(-1), (next_probs * weight_u).view(-1))
    return m  # [B, N], каждая строка суммируется в 1


class DistributionalLoss(nn.Module):
    atoms: torch.Tensor

    def __init__(self, module: nn.Module, n_atoms: int = 51, v_min: float | int = -10.0, v_max: float | int = +10.0):
        super().__init__()
        self.online: nn.Module = module
        self.target: nn.Module = deepcopy(module).requires_grad_(False)
        self.n_atoms: int = n_atoms
        self.v_min: float | int = v_min
        self.v_max: float | int = v_max
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))

    @staticmethod
    def _distribution(module: nn.Module, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(module(observation), dim=-1)

    def _greedy_action(self, probs: torch.Tensor) -> torch.Tensor:
        q_value: torch.Tensor = torch.sum(probs * self.atoms, dim=-1)  # [B, n_actions]
        return q_value.argmax(dim=-1)  # [B]

    def _pick_action(self, probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        index: torch.Tensor = actions.long().view(-1, 1, 1).expand(-1, 1, self.n_atoms)
        return probs.gather(1, index).squeeze(1)  # [B, n_atoms]

    @torch.no_grad()
    def target_distribution(self, x: Transition) -> torch.Tensor:
        """ Беллмановская цель m: [B, n_atoms]. Double DQN + проекция на сетку. """
        next_obs: torch.Tensor = x.raw.next.observation
        # 2.1 действие a* выбирает ONLINE-сеть...
        next_actions: torch.Tensor = self._greedy_action(self._distribution(self.online, next_obs))
        # 2.2 ...а распределение для a* берём у TARGET-сети
        next_probs: torch.Tensor = self._pick_action(self._distribution(self.target, next_obs), next_actions)
        # 2.3-2.4 Bellman-сдвиг + проекция обратно на линейку атомов
        return categorical_projection(next_probs,
                                      x.raw.next.reward,
                                      x.raw.next.done.float(),
                                      x.raw.gamma,
                                      self.atoms,
                                      self.v_min,
                                      self.v_max)

    def forward(self, x: Transition) -> tuple[torch.Tensor, torch.Tensor]:
        # Шаг 1: распределение online-сети для реально сделанного действия a_t.
        online_probs: torch.Tensor = self._distribution(self.online, x.raw.observation)
        dist: torch.Tensor = self._pick_action(online_probs, x.raw.action)  # [B, n_atoms]
        # Шаг 2: фиксированная беллмановская цель.
        target: torch.Tensor = self.target_distribution(x)  # [B, n_atoms]
        # Шаг 3: кросс-энтропия -sum(m * log p) по атомам — per-sample лосс. [B].
        log_dist: torch.Tensor = torch.log(dist.clamp(min=1e-8))
        td_errors: torch.Tensor = -torch.sum(target * log_dist, dim=1)
        # Шаг 4: взвешиваем по importance-sampling весам PER и усредняем.
        loss: torch.Tensor = (td_errors * x.raw.priority_weight).mean()
        return loss, td_errors.detach()  # td_errors -> новые приоритеты в буфере


class SoftUpdate:
    def __init__(self, loss_module: DistributionalLoss, tau: float | int = 0.05):
        assert 0.0 < tau <= 1.0, f"tau must be in (0, 1], got {tau}"
        self.online: nn.Module = loss_module.online
        self.target: nn.Module = loss_module.target
        self.tau: float | int = tau

    @torch.no_grad()
    def step(self) -> None:
        for p_online, p_target in zip(self.online.parameters(), self.target.parameters()):
            p_target.mul_(1.0 - self.tau).add_(p_online, alpha=self.tau)
        for b_online, b_target in zip(self.online.buffers(), self.target.buffers()):
            b_target.copy_(b_online)


class HardUpdate:
    def __init__(self, loss_module: DistributionalLoss, period: int):
        assert period >= 1, f"period must be a positive integer, got {period}"
        self.online: nn.Module = loss_module.online
        self.target: nn.Module = loss_module.target
        self.period: int = period
        self.counter: int = 0

    @torch.no_grad()
    def step(self) -> None:
        self.counter += 1
        if (self.counter % self.period) == 0:
            for p_online, p_target in zip(self.online.parameters(), self.target.parameters()):
                p_target.copy_(p_online)
            for b_online, b_target in zip(self.online.buffers(), self.target.buffers()):
                b_target.copy_(b_online)


class DQNOptimizer:
    def __init__(self,
                 loss_function: DistributionalLoss,
                 target_updater: SoftUpdate | HardUpdate,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 grad_norm: int | float = 1.0):
        self.loss_function: DistributionalLoss = loss_function
        self.target_updater: SoftUpdate | HardUpdate = target_updater
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = scheduler
        self.grad_norm: int | float = grad_norm

    def step(self, sample: Transition) -> tuple[torch.Tensor, torch.Tensor]:
        # Свежий шум для online и target один раз перед лоссом
        # (внутри лосса online прогоняется дважды — ресет там сломал бы граф in-place операцией).
        reset_noise(self.loss_function.online)
        reset_noise(self.loss_function.target)
        self.optimizer.zero_grad(set_to_none=True)
        loss, td_errors = self.loss_function(sample)
        loss.backward()
        nn.utils.clip_grad_norm_(self.loss_function.online.parameters(), self.grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.target_updater.step()
        return loss, td_errors
