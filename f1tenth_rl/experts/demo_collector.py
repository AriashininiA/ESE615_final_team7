"""
Demonstration Collector
========================
Collects expert demonstrations using pure pursuit in the F1TENTH Gym.

Usage:
    python scripts/collect_demos.py --episodes 100 --output demos/expert.npz
"""

import numpy as np
from typing import Dict, Any
from tqdm import tqdm
import os

from f1tenth_rl.envs.wrapper import F1TenthWrapper
from f1tenth_rl.experts.pure_pursuit import PurePursuitController


class DemoCollector:
    """
    Collect expert demonstrations for imitation learning.

    Runs pure pursuit and records (observation, action) pairs.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = F1TenthWrapper(config)

        # Get waypoints: custom raceline first, then env waypoints
        expert_wp = self._load_expert_waypoints(config)
        if expert_wp is not None:
            expert_cfg = dict(config.get("expert", {}))
            expert_cfg["_action_config"] = config.get("action", {})
            self.expert = PurePursuitController(
                expert_wp, expert_cfg
            )
        else:
            raise RuntimeError("No waypoints available for expert controller")

        # Set waypoints in obs builder
        self.env.obs_builder.set_waypoints(self.env.waypoints[:, :2])

        self.observations = []
        self.actions = []
        self.episode_returns = []
        self.episode_lengths = []

    def _load_expert_waypoints(self, config):
        """Load custom raceline if specified, otherwise use env waypoints."""
        wp_path = config.get("expert", {}).get("waypoint_path", None)
        if wp_path and os.path.exists(wp_path):
            data = np.loadtxt(wp_path, delimiter=",", skiprows=1)
            if data.shape[1] >= 3:
                wp = data[:, :3]
            else:
                speed = config.get("expert", {}).get("pure_pursuit", {}).get("target_speed", 5.0)
                wp = np.column_stack([data[:, :2], np.ones(len(data)) * speed])
            print(f"  Using custom raceline: {wp_path} ({len(wp)} waypoints)")
            return wp
        return self.env.waypoints

    def collect(self, num_episodes=100, max_steps=3000, render=False):
        """Collect demonstrations."""
        print(f"\nCollecting {num_episodes} expert demonstrations...")
        print(f"  Track waypoints: {len(self.env.waypoints)}")
        print(f"  Expert speed range: {self.expert.velocities.min():.1f} - {self.expert.velocities.max():.1f} m/s")

        for ep in tqdm(range(num_episodes), desc="Episodes"):
            obs, info = self.env.reset()
            flat_obs = info.get("raw_obs", {})
            ep_obs, ep_acts = [], []
            ep_return = 0
            done = False
            step = 0

            while not done and step < max_steps:
                action = self.expert.get_normalized_action(flat_obs, ego_idx=0)
                ep_obs.append(obs.copy())
                ep_acts.append(action.copy())

                obs, reward, terminated, truncated, info = self.env.step(action)
                flat_obs = info.get("raw_obs", {})
                ep_return += reward
                done = terminated or truncated
                step += 1

                if render:
                    self.env.render()

            self.observations.extend(ep_obs)
            self.actions.extend(ep_acts)
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(step)

        print(f"\nCollection complete!")
        print(f"  Total transitions: {len(self.observations)}")
        print(f"  Avg episode return: {np.mean(self.episode_returns):.2f}")
        print(f"  Avg episode length: {np.mean(self.episode_lengths):.0f}")
        crash_rate = sum(1 for l in self.episode_lengths if l < max_steps * 0.1) / num_episodes
        print(f"  Crash rate: {crash_rate:.1%}")

    def save(self, path: str):
        """Save demonstrations to NPZ file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(
            path,
            observations=np.array(self.observations, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            episode_returns=np.array(self.episode_returns),
            episode_lengths=np.array(self.episode_lengths),
        )
        print(f"  Saved {len(self.observations)} transitions to {path}")

    def close(self):
        self.env.close()
