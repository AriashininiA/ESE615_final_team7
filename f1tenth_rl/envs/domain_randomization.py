"""
Domain Randomization
====================
The key ingredient for sim-to-real transfer.

The problem: a policy trained in simulation learns to exploit the
exact physics of the simulator. When you put it on the real car,
the physics are slightly different (different friction, different
motor response, different sensor noise) and the policy crashes.

The solution: during training, randomly change the physics each
episode. The friction is different, the mass is different, the
lidar has noise. The agent can't memorize one set of dynamics —
it has to learn a robust policy that works across a RANGE of
conditions. The real world is just another sample from that range.

Three modes (set in config: domain_randomization.mode):

    "off"        → No randomization. Fast training, but the policy
                   won't work on real hardware.

    "fixed"      → Full randomization from the start. Can make early
                   training unstable because the agent faces randomized
                   physics before it even knows how to drive.

    "curriculum" → RECOMMENDED for sim2real. Start with clean physics
                   so the agent learns the basics, then gradually
                   increase randomization over training.

                   Timeline with warmup=0.2, full=0.6:
                   ├─── 0-20% ───┤─── 20-60% ───┤─── 60-100% ───┤
                   │  no DR       │  ramp up      │  full DR       │
                   │  learn basic │  get robust   │  stay robust   │

What gets randomized:
    - Tire friction:        Most important. Real friction varies with
                            floor surface, tire wear, and temperature.
    - Vehicle mass:         Affects acceleration and cornering.
    - Lidar noise:          Gaussian noise on beam readings.
    - Lidar dropout:        Random beams return zero (sensor failures).
    - Action delay:         Holds each action for extra steps, simulating
                            real-world latency (~30-50ms on hardware).
    - Cornering stiffness:  How the tires grip in turns.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


# Nominal vehicle parameters (center of randomization ranges)
NOMINAL = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "m": 3.74,
}


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wrapper that randomizes physical parameters on each reset.

    Supports fixed and curriculum (progressive) randomization.

    Parameters
    ----------
    env : gym.Env
        The F1TenthWrapper environment.
    config : dict
        Domain randomization configuration section.
    """

    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.dr_config = config
        self.mode = config.get("mode", "fixed")  # "off", "fixed", "curriculum"

        # Curriculum schedule (fraction of total training steps)
        self.curriculum_warmup = config.get("curriculum_warmup", 0.2)
        self.curriculum_full = config.get("curriculum_full", 0.6)

        # Current curriculum strength: 0.0 = no DR, 1.0 = full DR
        self._strength = 0.0 if self.mode == "curriculum" else 1.0
        if self.mode == "off":
            self._strength = 0.0

        # Self-tracking curriculum: each subprocess counts its own steps
        self._total_steps = 0
        self._curriculum_total = config.get("curriculum_total_steps", 2_000_000)

        # Lidar noise
        self.lidar_noise_cfg = config.get("lidar_noise", {})
        self.lidar_dropout_cfg = config.get("lidar_dropout", {})

        # Action delay
        self.action_delay_cfg = config.get("action_delay", {})
        self.action_buffer = deque(maxlen=10)
        self.current_delay = 0

    @property
    def strength(self) -> float:
        """Current randomization strength [0.0, 1.0]."""
        return self._strength

    def set_strength(self, strength: float):
        """Set randomization strength manually (used by curriculum callback)."""
        self._strength = np.clip(strength, 0.0, 1.0)

    # --- Methods callable via SB3 env_method() across process boundaries ---

    def set_dr_strength(self, strength: float):
        """Called by CurriculumDRCallback via env_method()."""
        self._strength = np.clip(strength, 0.0, 1.0)

    def get_dr_schedule(self):
        """Return (warmup, full) schedule for the callback."""
        return (self.curriculum_warmup, self.curriculum_full)

    def get_dr_strength(self):
        """Return current strength — callable via env_method for logging."""
        return self._strength

    def update_curriculum(self, progress: float):
        """
        Update curriculum strength based on training progress.

        Parameters
        ----------
        progress : float
            Training progress [0.0, 1.0] (current_step / total_steps).
        """
        if self.mode != "curriculum":
            return

        if progress < self.curriculum_warmup:
            self._strength = 0.0
        elif progress > self.curriculum_full:
            self._strength = 1.0
        else:
            # Linear ramp
            ramp = (progress - self.curriculum_warmup) / (self.curriculum_full - self.curriculum_warmup)
            self._strength = np.clip(ramp, 0.0, 1.0)

    def _scale_range(self, nominal: float, low: float, high: float) -> float:
        """Scale a randomization range by current strength."""
        if self._strength <= 0:
            return nominal
        # Interpolate between nominal and full range
        scaled_low = nominal + (low - nominal) * self._strength
        scaled_high = nominal + (high - nominal) * self._strength
        return np.random.uniform(scaled_low, scaled_high)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset with randomized parameters scaled by current strength."""

        if self._strength > 0:
            self._randomize_physics()

        # Action delay
        if self.action_delay_cfg.get("enabled", False) and self._strength > 0:
            max_delay = self.action_delay_cfg.get("max_steps", 2)
            # Scale delay by strength
            scaled_max = max(0, int(max_delay * self._strength))
            self.current_delay = np.random.randint(0, scaled_max + 1)
            self.action_buffer.clear()
        else:
            self.current_delay = 0

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._apply_lidar_effects(obs)

        info["dr_strength"] = self._strength
        info["dr_mode"] = self.mode
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with optional action delay and lidar noise."""
        self._total_steps += 1

        # Auto-update curriculum strength from step count
        if self.mode == "curriculum":
            progress = self._total_steps / self._curriculum_total
            self.update_curriculum(progress)

        # Apply action delay
        if self.current_delay > 0:
            self.action_buffer.append(action.copy())
            if len(self.action_buffer) > self.current_delay:
                action = self.action_buffer[0]
            else:
                action = np.zeros_like(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_lidar_effects(obs)

        info["dr_strength"] = self._strength
        return obs, reward, terminated, truncated, info

    def _randomize_physics(self):
        """Randomize vehicle dynamics, scaled by current strength."""
        base_env = self.env
        while hasattr(base_env, "env"):
            if hasattr(base_env, "base_env"):
                break
            base_env = base_env.env

        if not hasattr(base_env, "base_env"):
            return

        params = {}

        # Friction
        friction_cfg = self.dr_config.get("friction", {})
        if friction_cfg.get("enabled", False):
            r = friction_cfg.get("range", [0.7, 1.3])
            params["mu"] = self._scale_range(NOMINAL["mu"], r[0], r[1])

        # Mass
        mass_cfg = self.dr_config.get("mass", {})
        if mass_cfg.get("enabled", False):
            r = mass_cfg.get("range", [3.2, 4.2])
            params["m"] = self._scale_range(NOMINAL["m"], r[0], r[1])

        # Cornering stiffness
        cs_cfg = self.dr_config.get("cornering_stiffness", {})
        if cs_cfg.get("enabled", False):
            sr = cs_cfg.get("scale_range", [0.8, 1.2])
            scale = self._scale_range(1.0, sr[0], sr[1])
            params["C_Sf"] = NOMINAL["C_Sf"] * scale
            params["C_Sr"] = NOMINAL["C_Sr"] * scale

        if params:
            try:
                f1_env = base_env.base_env
                if hasattr(f1_env, "update_params"):
                    f1_env.update_params(params, index=0)
                elif hasattr(f1_env, "unwrapped") and hasattr(f1_env.unwrapped, "update_params"):
                    f1_env.unwrapped.update_params(params, index=0)
            except Exception:
                pass

    def _apply_lidar_effects(self, obs: np.ndarray) -> np.ndarray:
        """Add noise and dropout to lidar, scaled by strength."""
        if self._strength <= 0:
            return obs

        obs = obs.copy()

        # Determine lidar portion size
        base_env = self.env
        while hasattr(base_env, "env"):
            if hasattr(base_env, "obs_builder"):
                break
            base_env = base_env.env
        n_beams = getattr(base_env, "obs_builder", None)
        n_beams = n_beams.actual_beams if n_beams else min(108, len(obs))

        # Gaussian noise (scaled by strength)
        if self.lidar_noise_cfg.get("enabled", False):
            std = self.lidar_noise_cfg.get("std", 0.04) * self._strength
            if hasattr(base_env, "obs_builder") and base_env.obs_builder.lidar_normalize:
                std = std / base_env.obs_builder.lidar_clip
            obs[:n_beams] += np.random.normal(0, std, n_beams).astype(np.float32)

        # Dropout (scaled by strength)
        if self.lidar_dropout_cfg.get("enabled", False):
            rate = self.lidar_dropout_cfg.get("rate", 0.02) * self._strength
            mask = np.random.random(n_beams) < rate
            obs[:n_beams][mask] = 0.0

        return obs
