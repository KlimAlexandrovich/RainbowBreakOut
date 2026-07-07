import ale_py  # регистрирует ALE-среды в gymnasium
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

import torch
from torchrl.envs import GymWrapper
from torchrl.envs.transforms import InitTracker, TransformedEnv


class FireResetEnv(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if hasattr(env.unwrapped, "get_action_meanings"):
            assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        else:
            raise ValueError("Environment must have get_action_meanings method.")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


def soft_log_scale(value: int | float | np.ndarray) -> float | int:
    return np.sign(value) * np.log1p(np.abs(value))


def to_torchrl(device: torch.device, cat: bool = True):
    def decorator(function):
        def wrapper(*args, **kwargs) -> GymWrapper:
            envir = function(*args, **kwargs)
            envir = GymWrapper(envir, device=device, categorical_action_encoding=cat)
            envir = TransformedEnv(envir, InitTracker())
            return envir

        return wrapper

    return decorator


@to_torchrl(device="cpu")
def build_environment() -> gym.Env:
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env,
                             noop_max=30,
                             frame_skip=4,
                             screen_size=84,
                             terminal_on_life_loss=True,
                             grayscale_obs=True,
                             scale_obs=False)
    env = FireResetEnv(env)
    env = TransformReward(env, np.sign)
    env = FrameStackObservation(env, stack_size=4)
    return env
