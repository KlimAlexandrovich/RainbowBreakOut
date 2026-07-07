import os
import sys
import numpy as np
import cv2
import torch
from tensordict import TensorDict
from torchrl.envs.transforms.rb_transforms import MultiStepTransform
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from functools import wraps
from typing import Optional, Callable, Any, Deque
from collections import deque


def build_buffer(size: int,
                 directory: str,
                 alpha: float | int,
                 beta: float | int,
                 gamma: float | int,
                 n_steps: int = 1,
                 device: torch.device = torch.device("cpu")) -> TensorDictReplayBuffer:
    storage = LazyMemmapStorage(max_size=size, scratch_dir=directory, existsok=True)
    sampler = PrioritizedSampler(max_capacity=size, alpha=alpha, beta=beta)
    multi_step = MultiStepTransform(n_steps=n_steps, gamma=gamma)
    buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler, transform=multi_step)
    transform: Callable[[TensorDict], TensorDict] = lambda tensordict: tensordict.to(device)
    buffer.append_transform(transform)
    return buffer


def progress(step: int, total: int, description: str = "") -> None:
    width: int = 40
    filled: int = int(width * step / total)
    bar: str = "█" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {step}/{total};" + " " + description)
    sys.stdout.flush()


def save_video_opencv(frames: torch.Tensor, out_path: str = "video.mp4", fps: int = 15):
    if frames.dtype != torch.uint8:
        frames: torch.Tensor = (frames.clamp(0, 1) * 255).to(torch.uint8)
    frames: np.ndarray = frames.permute(0, 2, 3, 1).cpu().numpy()
    height, width = frames.shape[1:3]
    fourcc: int = cv2.VideoWriter.fourcc(*"mp4v")
    vw: cv2.VideoWriter = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        bgr: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()
    print(f"Saved in {out_path}")


def except_keyboard_interrupt(hook: Callable[[], None] = None):
    def decorator(function: Callable[[Any, ...], Any]):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except KeyboardInterrupt:
                if hook:
                    hook()
                print("Process interrupted by user.")

        return wrapper

    return decorator


def get_last_update(directory: str) -> Optional[str]:
    """ Finds and returns the path to the most recently modified file in a directory. """
    assert os.path.isdir(directory), f"Directory {directory} does not exist."
    listdir: list[str] = os.listdir(directory)
    abs_listdir: list[str] = [os.path.join(directory, path) for path in listdir]
    # "os.path.isfile" left directories if an absolute path exists.
    files: list[str] = list(filter(os.path.isfile, abs_listdir))
    if len(files) == 0: return None
    # Return the newest file.
    last: str = max(files, key=os.path.getmtime)
    return last


def build_linear_scheduler(start: float | int, end: float | int, total_steps: int) -> Callable[[int], float | int]:
    delta: int | float = end - start

    def schedule(step: int) -> float | int:
        fraction: int | float = min(1.0, step / total_steps)
        return start + fraction * delta

    return schedule


class RewardDeque:
    def __init__(self, window_size: int = 100, default: int | float = 0.0) -> None:
        self.reward_window: Deque[int | float] = deque(maxlen=window_size)
        self.episode_return: int | float = 0.0
        self.collected: int = 0
        self.default: int | float = default

    def add_frames(self, n: int) -> None:
        self.collected += n

    def step(self, reward: int | float, done: bool) -> None:
        self.episode_return += reward
        if done:
            self.reward_window.append(self.episode_return)
            self.episode_return = 0.0

    def update_batch(self, rewards: list[float | int], dones: list[bool]) -> None:
        for reward, done in zip(rewards, dones):
            self.step(float(reward), bool(done))

    @property
    def mean_reward(self) -> int | float:
        if len(self.reward_window) == 0:
            return self.default
        return sum(self.reward_window) / len(self.reward_window)

    def last_reward(self) -> int | float:
        if len(self.reward_window) == 0:
            return self.default
        return self.reward_window[-1]

    def reset(self) -> None:
        self.reward_window.clear()
        self.episode_return = 0.0
        self.collected = 0

    def __len__(self) -> int:
        return len(self.reward_window)

    def __repr__(self) -> str:
        return (f"WindowDeque("
                f"episodes={len(self.reward_window)}, "
                f"mean_reward={self.mean_reward:.2f}, "
                f"collected={self.collected}"
                f")")
