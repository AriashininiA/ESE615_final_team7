"""
Environment wrappers for F1TENTH Gym.

Provides gymnasium-compatible wrappers with configurable observation spaces,
action spaces, reward functions, and domain randomization.
"""

from f1tenth_rl.envs.wrapper import F1TenthWrapper, make_env, make_vec_env
from f1tenth_rl.envs.observations import ObservationBuilder
from f1tenth_rl.envs.rewards import RewardFunction, ProgressReward, CTHReward, SpeedReward
from f1tenth_rl.envs.domain_randomization import DomainRandomizationWrapper

__all__ = [
    "F1TenthWrapper",
    "make_env",
    "make_vec_env",
    "ObservationBuilder",
    "RewardFunction",
    "ProgressReward",
    "CTHReward",
    "SpeedReward",
    "DomainRandomizationWrapper",
]
