from tqdm.auto import tqdm
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import GymWrapper
from torchrl.collectors import Collector

from environment import build_environment
from module import Rainbow, DuelingDistributionHead, BreakoutBackbone, ScaleModule, reset_noise
from actor import DistributionActor
from utils import build_buffer, except_keyboard_interrupt
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
    # --- цикл обучения ---
    frames_per_batch: int = 4  # сколько кадров собираем за одну итерацию
    batch_size: int = 32  # размер батча из буфера
    total_frames_steps: int = 200_000_000 // frames_per_batch  # всего кадров среды за прогон
    updates_per_batch: int = 1  # градиентных шагов на один собранный батч
    min_buffer_steps: int = 80_000 // frames_per_batch  # прогрев: не учимся, пока в буфере меньше переходов
    # --- буфер / n-step / PER ---
    buffer_size: int = 250_000 * 4  # ёмкость реплей-буфера
    n_steps: int = 3  # горизонт multi-step возврата
    gamma: float | int = 0.99  # дисконт
    alpha: float | int = 0.5  # приоритизация PER
    beta: float | int = 0.4  # importance-sampling PER: стартовое значение β₀
    beta_end: float | int = 1.0  # β линейно анилится β₀ -> beta_end к концу обучения (как в Rainbow)
    # --- оптимизатор / target ---
    lr: float | int = 6.25e-5
    eps: float | int = 1.5e-4
    target_period: int = int((32000 / 4 / frames_per_batch) * updates_per_batch)
    grad_norm: float | int = 10.0  # клип нормы градиента
    # --- распределение / архитектура ---
    n_actions: int = 4  # число действий Breakout
    n_atoms: int = 51  # атомы категориального распределения (C51)
    v_min: float | int = -10.0
    v_max: float | int = +10.0
    # --- ввод-вывод ---
    buffer_dir: str = "staff/buffer"
    log_dir: str = "staff/logs"
    log_interval: int = 5_000  # печатать статистику и сохранять веса раз в N итераций
    # --- устройство (None -> авто: cuda/mps/cpu) ---
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
        self.collector = Collector(build_environment,
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
        self.reward_window: deque = deque(maxlen=100)
        self.episode_return: float | int = 0.0
        self.collected: int = 0

    def initialize_weights(self) -> None:
        env: GymWrapper = build_environment()
        _ = self.policy(env.fake_tensordict().to(self.device))
        last_update: str | None = self.logger.get_last_update(self.dqn.__class__.__name__)
        if last_update is not None:
            self.dqn.load_state_dict(torch.load(last_update))
            print(f"Loaded -> {last_update}")
        else:
            print("Weights initialized.")

    def mean_reward(self, default: int | float = 0.0) -> int | float:
        return sum(self.reward_window) / len(self.reward_window) if self.reward_window else default

    def annealed_beta(self) -> float | int:
        fraction: float | int = min(1.0, self.collected / self.cfg.total_frames_steps)
        return self.cfg.beta + fraction * (self.cfg.beta_end - self.cfg.beta)

    def _track_returns(self, td: Transition) -> None:
        rewards: list[float | int] = td.raw.next.reward.reshape(-1).tolist()
        dones: list[bool] = td.raw.next.done.reshape(-1).tolist()
        for reward, done in zip(rewards, dones):
            self.episode_return += reward
            if done:
                self.reward_window.append(self.episode_return)
                self.episode_return = 0.0

    def logger_step(self, mean_loss: float | int) -> None:
        self.logger.set_scalars(mean_loss=mean_loss,
                                avg_return=self.mean_reward(default=0.),
                                collected_frames=self.collected,
                                buffer_len=len(self.buffer))
        self.logger.checkpoint(weights=self.dqn.state_dict(), model=self.dqn.__class__.__name__)
        self.logger.draw_scalars()

    @except_keyboard_interrupt()
    def run(self) -> None:
        cfg: TrainConfig = self.cfg
        with tqdm(iterable=self.collector, total=len(self.collector)) as progress_bar:
            progress_bar.set_description("Filling the buffer...")
            for it, td in enumerate(progress_bar, start=1):
                # ----------------------------------------
                progress_bar.set_description(f"Filling the buffer...")
                reset_noise(self.dqn)
                self.buffer.extend(to_transition(td).td)
                self.collected += td.numel()
                self._track_returns(to_transition(td))
                self.buffer.sampler.beta = self.annealed_beta()
                # ----------------------------------------
                if len(self.buffer) < cfg.min_buffer_steps: continue
                # ----------------------------------------
                if progress_bar.desc == "Filling the buffer...":
                    progress_bar.set_description("Makes gradient descent steps...")
                cum_loss: float | int = 0
                for _ in range(cfg.updates_per_batch):
                    reset_noise(self.dqn)
                    sample: Transition = to_transition(self.buffer.sample(cfg.batch_size))
                    loss, td_errors = self.optim.step(sample)
                    self.buffer.update_priority(sample.raw.index, td_errors)
                    cum_loss += loss.item()
                # ----------------------------------------
                if ((it % cfg.log_interval) == 0) and (self.logger is not None):
                    progress_bar.set_description(f"Makes logger step...")
                    mean_loss: int | float = cum_loss / cfg.updates_per_batch
                    self.logger_step(mean_loss)
                    progress_bar.container = progress_bar.status_printer
                    progress_bar.display()
            self.logger.checkpoint(weights=self.dqn.state_dict(), model=self.dqn.__class__.__name__)
            progress_bar.set_description(f"Model saved -> {self.logger.get_last_update(self.dqn.__class__.__name__)}")
