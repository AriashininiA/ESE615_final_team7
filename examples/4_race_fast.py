#!/usr/bin/env python3
"""
=============================================================================
Example 4: Racing Fast — Train a Speed Demon
=============================================================================

Trains a policy that maximizes lap time by going as fast as possible.
Uses a reward that balances speed, progress, and survival.

Key differences from Example 1:
    - Higher max speed (8 m/s instead of 3 m/s)
    - Speed bonus in the reward function
    - Steering smoothness penalty (prevents oscillation at high speed)
    - Domain randomization for robustness

Reward design:
    reward = progress_reward + speed_bonus - steering_penalty - collision_penalty

    progress_reward:   Moving forward along the track (+)
    speed_bonus:       Going faster than min_speed (+)
    steering_penalty:  Changing steering rapidly (-)
    collision_penalty: Hitting a wall (big -)

Usage:
    python3 examples/4_race_fast.py                    # Train
    python3 examples/4_race_fast.py --eval --render    # Watch it race
    python3 examples/4_race_fast.py --steps 2000000    # Train longer
=============================================================================
"""

import gymnasium as gym
import numpy as np
import argparse
import os


class FastRacingEnv(gym.Env):
    """
    F1TENTH environment optimized for fast lap times.

    Observation: [108 lidar beams, velocity, yaw_rate, prev_steer, prev_speed]
    Action: [steer, speed] in [-1, 1]
    Reward: progress + speed_bonus - steering_penalty
    """

    def __init__(self, map_name="Spielberg", max_speed=8.0, render_mode=None):
        super().__init__()

        import f1tenth_gym
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

        self.env = gym.make("f1tenth_gym:f1tenth-v0", config=env_config, render_mode=render_mode)

        # Config
        self.num_beams = 108
        self.lidar_clip = 10.0
        self.max_speed = max_speed
        self.min_speed = 0.5
        self.max_steer = 0.4189
        self.max_steps = 6000
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_steer = 0.0

        # Observation: lidar + vel + yaw_rate + prev_action = 112
        obs_dim = self.num_beams + 1 + 1 + 2
        self.observation_space = gym.spaces.Box(-1, 1, (obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        # Track
        self.waypoints = self._extract_waypoints()
        self.prev_progress = 0.0

        # Domain randomization (simple version)
        self.friction_range = [0.8, 1.2]
        self.lidar_noise_std = 0.02

    def _extract_waypoints(self):
        try:
            track = self.env.unwrapped.track
            line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
            if line:
                return np.column_stack([line.xs, line.ys])
        except Exception:
            pass
        t = np.linspace(0, 2*np.pi, 100, endpoint=False)
        return np.column_stack([5*np.cos(t), 5*np.sin(t)])

    def _get_progress(self, x, y):
        dists = np.sqrt((self.waypoints[:,0]-x)**2 + (self.waypoints[:,1]-y)**2)
        return np.argmin(dists) / len(self.waypoints)

    def _build_obs(self, raw_obs):
        ego = raw_obs["agent_0"]
        scan = ego["scan"]

        # Downsample and normalize
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        scan = np.clip(scan, 0, self.lidar_clip)

        # Add lidar noise (domain randomization)
        scan += np.random.normal(0, self.lidar_noise_std, len(scan))
        scan = np.clip(scan, 0, self.lidar_clip) / self.lidar_clip

        state = ego["std_state"]
        vel = float(state[3]) / 10.0
        yaw_rate = float(state[5]) / 3.14

        return np.concatenate([
            scan, [vel, yaw_rate], self.prev_action
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_steer = 0.0
        self.prev_progress = 0.0
        return self._build_obs(raw_obs), info

    def step(self, action):
        # Scale action
        steer = float(action[0]) * self.max_steer
        speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed

        # Apply action smoothing (EMA)
        smoothing = 0.3
        steer = smoothing * steer + (1 - smoothing) * self.prev_steer

        raw_obs, _, done, truncated, info = self.env.step(np.array([[steer, speed]]))
        self.current_step += 1

        ego = raw_obs["agent_0"]
        state = ego["std_state"]
        x, y, vel = float(state[0]), float(state[1]), float(state[3])
        collision = bool(ego.get("collision", False))

        # ---- REWARD FUNCTION ----
        # 1. Progress reward (moving forward)
        progress = self._get_progress(x, y)
        delta = progress - self.prev_progress
        if delta < -0.5:
            delta += 1.0
        progress_reward = delta * 15.0

        # 2. Speed bonus (reward going fast)
        speed_bonus = vel / self.max_speed * 0.1

        # 3. Steering smoothness penalty (prevent oscillation)
        steer_change = abs(steer - self.prev_steer)
        steering_penalty = steer_change * 0.5

        # 4. Collision penalty
        if collision:
            reward = -10.0
            done = True
        else:
            reward = progress_reward + speed_bonus - steering_penalty

        if self.current_step >= self.max_steps:
            truncated = True

        self.prev_progress = progress
        self.prev_steer = steer
        self.prev_action = np.array([action[0], action[1]], dtype=np.float32)

        return self._build_obs(raw_obs), float(reward), done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

    print("=" * 60)
    print("  F1TENTH: Train to Race FAST")
    print("=" * 60)
    print(f"  Max speed: {args.max_speed} m/s")
    print(f"  Steps:     {args.steps:,}")
    print("=" * 60)

    # Use multiple environments for faster training
    num_envs = args.num_envs
    if num_envs > 1:
        env = SubprocVecEnv([
            lambda i=i: FastRacingEnv(args.map, max_speed=args.max_speed)
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([lambda: FastRacingEnv(args.map, max_speed=args.max_speed)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,        # Low entropy — exploit speed
        verbose=1,
        device=args.device,
        seed=42,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Bigger network
        ),
    )

    model.learn(total_timesteps=args.steps, progress_bar=True)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_fast_{args.map}")
    print(f"\nModel saved to models/ppo_fast_{args.map}.zip")
    env.close()


def evaluate(args):
    from stable_baselines3 import PPO
    import time

    model_path = args.model or f"models/ppo_fast_{args.map}"
    model = PPO.load(model_path, device="cpu")

    render_mode = "human" if args.render else None
    env = FastRacingEnv(args.map, max_speed=args.max_speed, render_mode=render_mode)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward, done, speeds = 0, False, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            #state = env.env.unwrapped.sim.agent_states[0]
            #speeds.append(float(state[3]))
            
            speeds.append(obs[108] * 10.0)  # vel was normalized by /10

            if render_mode:
                env.render()

        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, "
              f"avg_speed={avg_speed:.1f} m/s, max_speed={max_speed:.1f} m/s, "
              f"steps={env.current_step}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH: Race Fast")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-speed", type=float, default=8.0)
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
