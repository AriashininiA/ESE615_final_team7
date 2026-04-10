"""
Self-Play Wrapper
==================
Wraps the F1TENTH environment for RL vs RL training.

The opponent is controlled by a past version of the ego policy.
The opponent policy is updated periodically during training.

Usage:
    env = SelfPlayWrapper(config)
    # During training, call update_opponent(model) periodically
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy


class SelfPlayWrapper(gym.Wrapper):
    """
    Self-play wrapper: opponent uses a frozen copy of the ego policy.

    The opponent policy is updated every `update_freq` episodes with
    the current ego policy weights.

    Parameters
    ----------
    env : gym.Env
        The F1TenthWrapper with num_agents=2.
    update_freq : int
        Update opponent every N episodes.
    """

    def __init__(self, env: gym.Env, update_freq: int = 50):
        super().__init__(env)
        self.update_freq = update_freq
        self.episode_count = 0
        self.opponent_policy = None
        self.opponent_obs_rms = None

        # Access the inner wrapper's opponent controller slot
        inner = env
        while hasattr(inner, "env"):
            if hasattr(inner, "opponent_controller"):
                break
            inner = inner.env
        self.inner_wrapper = inner

    def set_opponent_policy(self, model):
        """Set the opponent policy from an SB3 model."""
        try:
            # Deep copy the policy network only (not the full model)
            self.opponent_policy = deepcopy(model.policy)
            self.opponent_policy.eval()
            # Freeze gradients
            for param in self.opponent_policy.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"[SelfPlay] Failed to copy policy: {e}")

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def get_opponent_action(self, obs_dict: Dict) -> Tuple[float, float]:
        """Get action from the frozen opponent policy."""
        if self.opponent_policy is None:
            # Fallback to driving slowly
            return 0.0, 1.0

        try:
            # Build observation for the opponent (agent index 1)
            obs_builder = self.inner_wrapper.obs_builder
            prev_action = np.zeros(2, dtype=np.float32)
            obs = obs_builder.build(obs_dict, ego_idx=1, prev_action=prev_action)

            # Run inference
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = self.opponent_policy._predict(obs_tensor, deterministic=True)

            action = action.cpu().numpy().squeeze()

            # Scale from [-1,1] to [steer, speed]
            steer = float(action[0]) * self.inner_wrapper.max_steer
            speed = (float(action[1]) + 1.0) * 0.5 * (
                self.inner_wrapper.max_speed - self.inner_wrapper.min_speed
            ) + self.inner_wrapper.min_speed

            return steer, speed
        except Exception:
            return 0.0, 1.0
