"""
RL Agents for F1TENTH.

Provides:
    - SB3Trainer: Stable Baselines 3 training (PPO, SAC, TD3)
    - CustomPPO: CleanRL-style PPO for full control
    - ImitationTrainer: Behavioral cloning + RL fine-tuning
"""

from f1tenth_rl.agents.sb3_trainer import SB3Trainer
from f1tenth_rl.agents.custom_ppo import CustomPPO
from f1tenth_rl.agents.imitation import ImitationTrainer
from f1tenth_rl.agents.networks import LidarCNN, RacingMLP

__all__ = [
    "SB3Trainer",
    "CustomPPO",
    "ImitationTrainer",
    "LidarCNN",
    "RacingMLP",
]
