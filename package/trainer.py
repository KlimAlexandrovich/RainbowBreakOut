import sys
from dataclasses import dataclass
from typing import Callable
from collections.abc import Iterator

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import GymWrapper
from torchrl.collectors import Collector

from environment import build_environment
from module import Rainbow, DuelingDistributionHead, BreakoutBackbone, ScaleModule, reset_noise
from actor import DistributionActor
from utils import RewardDeque, build_buffer, except_keyboard_interrupt, build_linear_scheduler
from td_view import to_transition, Transition
from optim import DistributionalLoss, HardUpdate, DQNOptimizer
from Logger import SmartLogger


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    # --- Цикл обучения ---
    frames_per_batch: int = 32  # Оригинальный 4. Сколько шагов среды собираем за одну итерацию.
    batch_size: int = 256  # Размер батча из буфера.
    action_repeat: int = 4  # = frame_skip в AtariPreprocessing; перевод кадры эмулятора -> шаги среды.
    total_frames_steps: int = 50_000_000 // action_repeat  # 12.5M шагов среды = 50M кадров эмулятора.
    updates_per_batch: int = 1  # Градиентных шагов на один собранный батч.
    min_buffer_size: int = 80_000  # Прогрев: не учимся, пока в буфере меньше переходов.
    # --- Буфер / N-step / PER ---
    buffer_size: int = int(1.5 * 250_000)  # Ёмкость реплей-буфера (x * ≈ 14 ГБ).
    n_steps: int = 3  # Горизонт multi-step возврата.
    gamma: float | int = 0.99  # Ставка дисконтирования.
    alpha: float | int = 0.5  # Приоритизация PER.
    beta: float | int = 0.4  # Importance-sampling PER: стартовое значение β₀.
    beta_end: float | int = 1.0  # β линейно растёт β₀ -> beta_end к концу обучения (как в Rainbow)
    # --- Оптимизатор / Target ---
    lr: float | int = 1.76e-4  # SQRT-масштабирование под batch=256 (6.25e-5 * sqrt(256/32) ≈ 1.76e-4).
    eps: float | int = 1.5e-4
    target_period: int = int((32000 / action_repeat / frames_per_batch) * updates_per_batch)
    grad_norm: float | int = 10.0  # Максимум нормы градиента.
    # --- Распределение / Архитектура ---
    n_actions: int = 4  # Число действий среды.
    n_atoms: int = 51  # Атомы категориального распределения (C51).
    v_min: float | int = -10.0
    v_max: float | int = +10.0
    # --- Ввод-вывод ---
    buffer_dir: str = "/tmp/staff/buffer"
    log_dir: str = "/kaggle/working/staff/logs"
    log_interval: int = 20_000  # Сохранять статистику и веса раз в N итераций.
    show_interval: int = 100  # Печатать статистику раз в N итераций.
    # --- Устройство (None -> авто: cuda/mps/cpu) ---
    device: torch.device | str | None = None


class Trainer:
    def __init__(self, config: TrainConfig | None = None):
        self.cfg: TrainConfig = config or TrainConfig()
        # ------------------------------------
        cfg = self.cfg
        self.device: torch.device = (torch.device(cfg.device) if (cfg.device is not None) else resolve_device())
        # ---------------------------------------------
        self.dqn = nn.Sequential(ScaleModule(1. / 255.),
                                 Rainbow(BreakoutBackbone(),
                                         DuelingDistributionHead(cfg.n_actions, cfg.n_atoms)))
        self.policy = TensorDictSequential(
            TensorDictModule(module=self.dqn, in_keys=["observation"], out_keys=["logits"]),
            TensorDictModule(module=DistributionActor(cfg.n_atoms, cfg.v_min, cfg.v_max),
                             in_keys=["logits"],
                             out_keys=["action"]),
        ).to(self.device)
        # ---------------------------------------------
        self.logger: SmartLogger = SmartLogger(self.dqn.__class__.__name__, log_dir=self.cfg.log_dir, exp_name="main")
        self.initialize_weights()
        # ---------------------------------------------
        loss_module = DistributionalLoss(self.dqn, cfg.n_atoms, cfg.v_min, cfg.v_max).to(self.device)
        self.optim = DQNOptimizer(loss_function=loss_module,
                                  target_updater=HardUpdate(loss_module, period=cfg.target_period),
                                  optimizer=torch.optim.Adam(self.dqn.parameters(), lr=cfg.lr, eps=cfg.eps),
                                  grad_norm=cfg.grad_norm)
        self.collector: Iterator[TensorDict] = Collector(build_environment,
                                                         policy=self.policy,
                                                         frames_per_batch=cfg.frames_per_batch,
                                                         total_frames=cfg.total_frames_steps,
                                                         policy_device=self.device)
        self.buffer = build_buffer(size=cfg.buffer_size,
                                   device=self.device,
                                   directory=cfg.buffer_dir,
                                   alpha=cfg.alpha,
                                   beta=cfg.beta,
                                   gamma=cfg.gamma,
                                   n_steps=cfg.n_steps)
        # ---------------------------------------------
        self.reward_window: RewardDeque = RewardDeque(window_size=100, default=0.0)
        self.beta_scheduler: Callable[[int], float | int] = build_linear_scheduler(cfg.beta,
                                                                                   cfg.beta_end,
                                                                                   cfg.total_frames_steps)

    def initialize_weights(self) -> None:
        env: GymWrapper = build_environment()
        _ = self.policy(env.fake_tensordict().to(self.device))
        last_update: str | None = self.logger.get_last_update(self.dqn.__class__.__name__)
        if last_update is not None:
            self.dqn.load_state_dict(torch.load(last_update))
            print(f"Loaded -> {last_update}")
        else:
            print("Weights initialized.")

    def buffer_step(self, td: TensorDict) -> None:
        reset_noise(self.dqn)
        self.buffer.extend(to_transition(td).td)
        self.track_returns_step(td)
        self.buffer.sampler.beta = self.beta_scheduler(self.reward_window.collected)

    def track_returns_step(self, td: TensorDict) -> None:
        transition: Transition = to_transition(td)
        rewards: list[float | int] = transition.raw.next.reward.reshape(-1).tolist()
        dones: list[bool] = transition.raw.next.done.reshape(-1).tolist()
        self.reward_window.add_frames(td.numel())
        self.reward_window.update_batch(rewards, dones)

    def logger_step(self, step: int, cumulative_loss: float | int) -> None:
        if ((step % self.cfg.log_interval) == 0) and (self.logger is not None):
            mean_loss: int | float = cumulative_loss / self.cfg.updates_per_batch
            self.logger.set_scalars(mean_loss=mean_loss,
                                    avg_return=self.reward_window.mean_reward,
                                    collected_frames=self.reward_window.collected,
                                    buffer_len=len(self.buffer))
            self.logger.checkpoint(weights=self.dqn.state_dict(), model=self.dqn.__class__.__name__)

    def loss_step(self) -> float | int:
        cumulative_loss: float | int = 0
        for _ in range(self.cfg.updates_per_batch):
            reset_noise(self.dqn)
            sample: Transition = to_transition(self.buffer.sample(self.cfg.batch_size))
            loss, td_errors = self.optim.step(sample)
            self.buffer.update_priority(sample.raw.index, td_errors)
            cumulative_loss += loss.item()
        return cumulative_loss

    @except_keyboard_interrupt()
    def run(self) -> None:
        sys.stderr.write("Training. Filling buffer step...")
        sys.stderr.flush()
        for step, td in enumerate(self.collector, start=1):
            # ----------------------------------------
            self.buffer_step(td=td)
            # ----------------------------------------
            if len(self.buffer) < self.cfg.min_buffer_size:
                if (step % self.cfg.show_interval) == 0:
                    log_string: str = f"Iteration: {step}/{len(self.collector)}; Buffer len: {len(self.buffer)}.\n"
                    sys.stderr.write(log_string), sys.stderr.flush()
                continue
            # ----------------------------------------
            cumulative_loss: float | int = self.loss_step()
            self.logger_step(step=step, cumulative_loss=cumulative_loss)
            # ----------------------------------------
            if (step % self.cfg.show_interval) == 0:
                mean_loss: int | float = cumulative_loss / self.cfg.updates_per_batch
                log_string = (f"Iteration: {step}/{len(self.collector)}; "
                              f"Loss: {mean_loss:.4f}; "
                              f"Avg. return: {self.reward_window.mean_reward:.2f};"
                              f"Collected frames: {self.reward_window.collected}; "
                              f"Buffer len: {len(self.buffer)}.\n")
                sys.stderr.write(log_string)
                sys.stderr.flush()
        self.logger.checkpoint(weights=self.dqn.state_dict(), model=self.dqn.__class__.__name__)
