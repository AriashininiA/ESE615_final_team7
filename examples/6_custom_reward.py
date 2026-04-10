#!/usr/bin/env python3
"""
=============================================================================
Example 6: Custom Reward Functions
=============================================================================

Your reward function is the MOST important design decision in RL.
It tells the agent what "good driving" means. Get it wrong, and
the agent learns the wrong behavior — or nothing at all.

This example shows how to create custom reward functions:
    1. Trajectory-Aided Learning (TAL): reward matching an expert's
       speed and steering decisions
    2. Frenet-Frame reward: reward based on the car's position
       relative to the raceline (lateral error, heading error)
    3. How to plug them into the framework

These reward types are useful for:
    - Racing faster by following an optimal speed profile
    - Staying precisely on a raceline
    - Any task where you have a reference trajectory

Usage:
    python3 examples/6_custom_reward.py                    # Train with TAL reward
    python3 examples/6_custom_reward.py --reward frenet    # Train with Frenet reward
    python3 examples/6_custom_reward.py --eval --render    # Watch it drive
=============================================================================
"""

import gymnasium as gym
import numpy as np
import argparse
import os

# ============================================================
# THE ENVIRONMENT (same as Example 1)
# ============================================================

class F1TenthRLWrapper(gym.Env):
    """F1TENTH wrapper with pluggable reward functions."""

    metadata = {"render_modes": ["human", "human_fast"]}

    def __init__(self, map_name="Spielberg", render_mode=None, reward_type="tal"):
        super().__init__()
        from f1tenth_gym.envs.env_config import (
            EnvConfig, SimulationConfig, ObservationConfig,
            ResetConfig,
        )
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.dynamic_models import DynamicModel

        env_config = EnvConfig(
            map_name=map_name,
            num_agents=1,
            simulation_config=SimulationConfig(
                timestep=0.01,
                integrator=IntegratorType.RK4,
                dynamics_model=DynamicModel.ST,
                max_laps=1,
            ),
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_GRID_STATIC),
            render_enabled=(render_mode is not None),
        )
        self.env = gym.make("f1tenth_gym:f1tenth-v0",
                            config=env_config, render_mode=render_mode)
        self.render_mode = render_mode

        # Get track info for reward computation
        self.track = self.env.unwrapped.track
        self.centerline = self.track.centerline
        waypoints = np.column_stack([self.centerline.xs, self.centerline.ys])
        self.waypoints = waypoints
        self.track_length = self.centerline.length

        # Pure pursuit for TAL reference actions
        from f1tenth_rl.experts.pure_pursuit import PurePursuitController
        self.expert = PurePursuitController(
            np.column_stack([waypoints[:, 0], waypoints[:, 1],
                             np.full(len(waypoints), 3.0)]),
            {"pure_pursuit": {"target_speed": 3.0},
             "_action_config": {"max_speed": 5.0, "min_speed": 0.5}}
        )

        # Reward type
        self.reward_type = reward_type

        # Spaces
        self.num_beams = 108
        self.max_speed = 5.0
        self.min_speed = 0.5
        obs_dim = self.num_beams + 4  # lidar + vel + yaw_rate + prev_action
        self.observation_space = gym.spaces.Box(-1, 1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.prev_action = np.zeros(2)
        self.prev_progress = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self._update_state(obs)
        self.prev_action = np.zeros(2)

        # Reset progress tracking
        x, y = self._get_pose(obs)
        self.prev_progress = self._get_progress(x, y)

        return self._make_obs(obs), info

    def step(self, action):
        # Scale action to physical units
        steer = float(action[0]) * 0.4189
        speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
        physical_action = np.array([[steer, speed]])

        obs, _, terminated, truncated, info = self.env.step(physical_action)
        self._update_state(obs)

        # ============================================================
        # REWARD COMPUTATION — This is the interesting part!
        # ============================================================
        collision = bool(obs["agent_0"]["collision"])

        if self.reward_type == "tal":
            reward = self._tal_reward(obs, action, collision)
        elif self.reward_type == "frenet":
            reward = self._frenet_reward(obs, action, collision)
        else:
            reward = self._progress_reward(obs, collision)

        self.prev_action = action.copy()
        return self._make_obs(obs), reward, terminated, truncated, info

    # ============================================================
    # REWARD TYPE 1: Trajectory-Aided Learning (TAL)
    # ============================================================
    # Instead of just rewarding forward progress, TAL rewards the
    # agent for choosing actions SIMILAR to what an expert would do.
    #
    # The idea: if the expert slows down in a corner, the agent
    # should learn to slow down there too. If the expert goes full
    # speed on a straight, the agent should match that.
    #
    # r_TAL = 1 - |v_agent - v_expert| / v_max - |δ_agent - δ_expert| / δ_max
    #
    # This biases the agent toward expert-like behavior while still
    # allowing it to discover improvements through RL exploration.

    def _tal_reward(self, obs, action, collision):
        if collision:
            return -10.0

        # What would the expert do right now?
        flat_obs = self._flatten_obs(obs)
        expert_action = self.expert.get_normalized_action(flat_obs, ego_idx=0)

        # How close is the agent's action to the expert's?
        steer_error = abs(action[0] - expert_action[0])   # both in [-1, 1]
        speed_error = abs(action[1] - expert_action[1])   # both in [-1, 1]

        # TAL reward: 1.0 when perfectly matching, lower when diverging
        tal_reward = 1.0 - 0.5 * steer_error - 0.5 * speed_error

        # Add a small progress bonus so the agent still cares about going forward
        x, y = self._get_pose(obs)
        progress = self._get_progress(x, y)
        delta = progress - self.prev_progress
        if delta < -self.track_length * 0.5:
            delta += self.track_length
        elif delta > self.track_length * 0.5:
            delta -= self.track_length
        self.prev_progress = progress

        progress_reward = 5.0 * (delta / self.track_length)

        return tal_reward + progress_reward

    # ============================================================
    # REWARD TYPE 2: Frenet Frame Reward
    # ============================================================
    # The Frenet frame describes the car's position RELATIVE to
    # the raceline:
    #   s    = how far along the track (arc length)
    #   ey   = lateral error (how far from the centerline)
    #   ephi = heading error (angle between car and track direction)
    #
    # This is extremely useful because:
    #   - ey tells you if the car is drifting off the racing line
    #   - ephi tells you if the car is pointing the wrong way
    #   - Changes in s tell you forward progress
    #
    # Perfect driving: ey ≈ 0, ephi ≈ 0, Δs > 0

    def _frenet_reward(self, obs, action, collision):
        if collision:
            return -10.0

        x, y = self._get_pose(obs)
        theta = float(obs["agent_0"]["std_state"][4])

        # Convert to Frenet coordinates using the Track object
        s, ey, ephi = self.track.cartesian_to_frenet(x, y, theta)

        # Reward components:
        # 1. Penalize lateral deviation from centerline
        lateral_penalty = -2.0 * abs(ey)

        # 2. Penalize heading error (not pointing along track)
        heading_penalty = -1.0 * abs(ephi)

        # 3. Reward forward progress
        progress = s
        delta = progress - self.prev_progress
        if delta < -self.track_length * 0.5:
            delta += self.track_length
        elif delta > self.track_length * 0.5:
            delta -= self.track_length
        self.prev_progress = progress
        progress_reward = 10.0 * (delta / self.track_length)

        # 4. Penalize jerky steering
        steer_change = abs(action[0] - self.prev_action[0])
        smoothness_penalty = -0.5 * steer_change

        return progress_reward + lateral_penalty + heading_penalty + smoothness_penalty

    # ============================================================
    # REWARD TYPE 3: Simple Progress (fallback)
    # ============================================================

    def _progress_reward(self, obs, collision):
        if collision:
            return -10.0
        x, y = self._get_pose(obs)
        progress = self._get_progress(x, y)
        delta = progress - self.prev_progress
        if delta < -self.track_length * 0.5:
            delta += self.track_length
        elif delta > self.track_length * 0.5:
            delta -= self.track_length
        self.prev_progress = progress
        return 10.0 * (delta / self.track_length)

    # ============================================================
    # Helper methods
    # ============================================================

    def _make_obs(self, obs):
        scan = np.array(obs["agent_0"]["scan"], dtype=np.float32)
        scan = np.clip(scan, 0, 10.0) / 10.0
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        if len(scan) < self.num_beams:
            scan = np.pad(scan, (0, self.num_beams - len(scan)))

        vel = float(obs["agent_0"]["std_state"][3]) / 10.0
        yaw_rate = float(obs["agent_0"]["std_state"][5]) / 3.14
        extra = np.array([vel, yaw_rate, self.prev_action[0], self.prev_action[1]], dtype=np.float32)
        return np.concatenate([scan, extra])

    def _get_pose(self, obs):
        return float(obs["agent_0"]["std_state"][0]), float(obs["agent_0"]["std_state"][1])

    def _get_progress(self, x, y):
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        cumulative = np.concatenate([[0], np.cumsum(
            np.sqrt(np.sum(np.diff(self.waypoints, axis=0)**2, axis=1)))])
        return cumulative[np.argmin(dists)]

    def _flatten_obs(self, obs):
        """Convert to legacy dict format for pure pursuit."""
        std = obs["agent_0"]["std_state"]
        vel = float(std[3])
        beta = float(std[6])
        return {
            "scans": [np.array(obs["agent_0"]["scan"])],
            "poses_x": [float(std[0])],
            "poses_y": [float(std[1])],
            "poses_theta": [float(std[4])],
            "linear_vels_x": [vel * np.cos(beta)],
            "ang_vels_z": [float(std[5])],
        }

    def _update_state(self, obs):
        self._current_obs = obs

    def render(self):
        if self.render_mode:
            return self.env.render()

    def close(self):
        self.env.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Example 6: Custom Rewards")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--reward", type=str, default="tal",
                        choices=["tal", "frenet", "progress"],
                        help="Which reward function to use")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model or f"models/ex6_{args.reward}_{args.map}"

    if args.eval:
        # ---- Evaluate ----
        from stable_baselines3 import PPO
        render_mode = "human" if args.render else None
        env = F1TenthRLWrapper(map_name=args.map, render_mode=render_mode,
                               reward_type=args.reward)
        model = PPO.load(model_path, device="cpu")
        import time

        for ep in range(args.episodes):
            obs, _ = env.reset()
            total_reward, done = 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if args.render:
                    env.render()
                    time.sleep(0.005)
            print(f"Episode {ep+1}: reward={total_reward:.1f}")
        env.close()
    else:
        # ---- Train ----
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        print(f"\nTraining with {args.reward.upper()} reward on {args.map}")
        print(f"  Steps: {args.steps:,}")
        print()

        env = DummyVecEnv([lambda: F1TenthRLWrapper(
            map_name=args.map, reward_type=args.reward)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        model = PPO("MlpPolicy", env, verbose=1, device=args.device,
                     learning_rate=3e-4, n_steps=2048, batch_size=128,
                     policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
        model.learn(total_timesteps=args.steps)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
