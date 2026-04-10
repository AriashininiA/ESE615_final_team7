#!/usr/bin/env python3
"""
=============================================================================
Example 1: Simple PPO Training for F1TENTH (dev-humble branch)
=============================================================================

This is the SIMPLEST possible RL training script for F1TENTH.
It trains a PPO policy that learns to drive around a track using only
lidar scans as input.

What it does:
    1. Creates the F1TENTH Gym environment
    2. Wraps it so the RL algorithm can use it
    3. Trains PPO for 500k steps
    4. Saves the model
    5. Evaluates it with rendering

Key concepts:
    - Observation: Raw lidar scan (1080 beams, clipped to 10m, normalized to [0,1])
    - Action: [steering_angle, speed] normalized to [-1, 1]
    - Reward: Progress along the track centerline

Usage:
    python3 examples/1_simple_ppo.py              # Train
    python3 examples/1_simple_ppo.py --eval        # Evaluate saved model
    python3 examples/1_simple_ppo.py --eval --render  # Evaluate with rendering
=============================================================================
"""

import gymnasium as gym
import numpy as np
import argparse
import os

# ============================================================
# STEP 1: Create a Gym wrapper for F1TENTH (dev-humble API)
# ============================================================
# The F1TENTH Gym uses a nested dict observation format.
# RL algorithms (like SB3's PPO) need a flat Box observation.
# This wrapper converts between the two.

class F1TenthRLWrapper(gym.Env):
    """
    Minimal wrapper that converts F1TENTH Gym into a standard RL environment.

    Observation: 1080 lidar beams → downsampled to `num_beams`, normalized to [0,1]
    Action: [steer, speed] in [-1, 1] → scaled to physical units
    Reward: Progress along track centerline + collision penalty
    """

    def __init__(self, map_name="Spielberg", num_beams=108, render_mode=None):
        super().__init__()

        # ---- Import F1TENTH Gym (dev-humble) ----
        import f1tenth_gym
        from f1tenth_gym.envs.env_config import (
            EnvConfig, SimulationConfig, ObservationConfig,
            ResetConfig, ControlConfig,
        )
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.dynamic_models import DynamicModel

        # ---- Create the base environment ----
        env_config = EnvConfig(
            map_name=map_name,
            num_agents=1,
            simulation_config=SimulationConfig(
                timestep=0.01,                    # 100 Hz simulation
                integrator=IntegratorType.RK4,    # Runge-Kutta 4th order
                dynamics_model=DynamicModel.ST,   # Single-track model
                max_laps=1,
            ),
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_GRID_STATIC),
            render_enabled=(render_mode is not None),
        )

        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=env_config,
            render_mode=render_mode,
        )

        # ---- Configuration ----
        self.num_beams = num_beams       # Downsample lidar to this many beams
        self.lidar_clip = 10.0           # Clip lidar range to 10m
        self.max_speed = 8.0             # Maximum speed (m/s)
        self.min_speed = 0.5             # Minimum speed (m/s)
        self.max_steer = 0.4189          # Maximum steering angle (rad, ~24°)
        self.max_steps = 6000            # Max steps per episode (~60 seconds)
        self.current_step = 0

        # ---- Spaces ----
        # Observation: downsampled + normalized lidar
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_beams,), dtype=np.float32
        )
        # Action: [steer, speed] both in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ---- Track waypoints for reward computation ----
        self.waypoints = self._extract_waypoints()
        self.prev_progress = 0.0

    def _extract_waypoints(self):
        """Get track centerline from the gym's Track object."""
        try:
            track = self.env.unwrapped.track
            line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
            if line is not None:
                return np.column_stack([line.xs, line.ys])
        except Exception:
            pass
        # Fallback: dummy circle
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        return np.column_stack([5 * np.cos(t), 5 * np.sin(t)])

    def _get_progress(self, x, y):
        """Compute how far along the track the car is (0 to 1)."""
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest = np.argmin(dists)
        return closest / len(self.waypoints)

    def _process_obs(self, raw_obs):
        """Convert nested dict observation to flat lidar array."""
        scan = raw_obs["agent_0"]["scan"]

        # Downsample: 1080 beams → num_beams
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]

        # Clip and normalize to [0, 1]
        scan = np.clip(scan, 0.0, self.lidar_clip) / self.lidar_clip

        return scan.astype(np.float32)

    def _scale_action(self, action):
        """Convert [-1, 1] action to physical [steer, speed]."""
        steer = float(action[0]) * self.max_steer
        speed = (float(action[1]) + 1.0) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
        return np.array([[steer, speed]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_progress = 0.0
        obs = self._process_obs(raw_obs)
        return obs, info

    def step(self, action):
        # Scale action to physical units
        physical_action = self._scale_action(action)

        # Step the simulator
        raw_obs, _, done, truncated, info = self.env.step(physical_action)
        self.current_step += 1

        # Extract state
        obs = self._process_obs(raw_obs)
        state = raw_obs["agent_0"]["std_state"]
        x, y = float(state[0]), float(state[1])
        collision = bool(raw_obs["agent_0"].get("collision", False))

        # ---- Compute reward ----
        progress = self._get_progress(x, y)

        # Progress reward: reward for moving forward along the track
        delta_progress = progress - self.prev_progress
        if delta_progress < -0.5:
            delta_progress += 1.0  # Handle wrap-around
        reward = delta_progress * 10.0

        # Collision penalty
        if collision:
            reward = -10.0
            done = True

        # Truncate if too many steps
        if self.current_step >= self.max_steps:
            truncated = True

        self.prev_progress = progress
        return obs, float(reward), done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================
# STEP 2: Train or Evaluate
# ============================================================

def train(args):
    """Train a PPO policy."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    print("=" * 60)
    print("  F1TENTH PPO Training")
    print("=" * 60)
    print(f"  Map:        {args.map}")
    print(f"  Steps:      {args.steps:,}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Create vectorized environment (SB3 requirement)
    env = DummyVecEnv([lambda: F1TenthRLWrapper(map_name=args.map)])
    env = VecMonitor(env)

    # Create PPO agent
    model = PPO(
        "MlpPolicy",                # Standard MLP policy network
        env,
        learning_rate=3e-4,          # Adam optimizer learning rate
        n_steps=2048,                # Steps per rollout
        batch_size=128,              # Minibatch size for SGD
        n_epochs=10,                 # SGD epochs per rollout
        gamma=0.99,                  # Discount factor
        gae_lambda=0.95,             # GAE lambda
        clip_range=0.2,              # PPO clipping parameter
        ent_coef=0.01,               # Entropy bonus (encourages exploration)
        verbose=1,
        device=args.device,
        seed=42,
        tensorboard_log="runs/",
    )

    # Train!
    model.learn(total_timesteps=args.steps, progress_bar=True)

    # Save
    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_{args.map}")
    print(f"\nModel saved to models/ppo_{args.map}.zip")

    env.close()


def evaluate(args):
    """Evaluate a trained policy."""
    from stable_baselines3 import PPO

    model_path = args.model or f"models/ppo_{args.map}"
    print(f"Loading model: {model_path}")

    model = PPO.load(model_path, device="cpu")
    render_mode = "human" if args.render else None
    env = F1TenthRLWrapper(map_name=args.map, render_mode=render_mode)

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if render_mode:
                env.render()

        print(f"  Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH PPO Example")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--map", type=str, default="Spielberg", help="Track name")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Training steps")
    parser.add_argument("--episodes", type=int, default=5, help="Eval episodes")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
