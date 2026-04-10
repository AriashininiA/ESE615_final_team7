#!/usr/bin/env python3
"""
=============================================================================
Example 2: Racing Against a Pure Pursuit Opponent
=============================================================================

Trains an RL agent to race against a pure pursuit waypoint follower.
The RL agent learns to:
    - Overtake the opponent on straights
    - Defend position in corners
    - Avoid collisions with the other car
    - React to dynamic obstacles (the opponent)

Architecture:
    Agent 0 (ego):      RL policy (PPO) — learns from experience
    Agent 1 (opponent):  Pure pursuit — follows the track centerline

Observation: [108 lidar beams + velocity + yaw_rate + prev_action] = 112 dims
    The lidar naturally "sees" the opponent car as an obstacle.

Usage:
    python3 examples/2_race_against_opponent.py              # Train
    python3 examples/2_race_against_opponent.py --eval --render  # Watch it race
=============================================================================
"""

import gymnasium as gym
import numpy as np
import argparse
import os


# ============================================================
# Pure Pursuit Controller (the opponent)
# ============================================================

class PurePursuitOpponent:
    """
    Simple pure pursuit controller that follows the track centerline.
    This is the opponent that the RL agent races against.

    Pure pursuit works by:
        1. Find the closest waypoint to the car
        2. Look ahead by `lookahead_distance` meters
        3. Compute the steering angle to drive toward that point
        4. Drive at a constant speed
    """

    def __init__(self, waypoints, target_speed=3.0, lookahead=1.5):
        self.waypoints = waypoints[:, :2]  # [x, y]
        self.target_speed = target_speed
        self.lookahead = lookahead
        self.wheelbase = 0.33  # F1TENTH wheelbase (meters)

    def get_action(self, x, y, theta):
        """Compute steering angle and speed given car pose."""
        # Find closest waypoint
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest = np.argmin(dists)

        # Find lookahead point
        n = len(self.waypoints)
        lookahead_idx = closest
        for i in range(1, n):
            idx = (closest + i) % n
            d = np.sqrt((self.waypoints[idx, 0] - x)**2 + (self.waypoints[idx, 1] - y)**2)
            if d >= self.lookahead:
                lookahead_idx = idx
                break

        # Compute steering via pure pursuit formula
        target = self.waypoints[lookahead_idx]
        dx = target[0] - x
        dy = target[1] - y

        # Transform to car frame
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        # Pure pursuit steering
        L = np.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            return 0.0, self.target_speed

        curvature = 2.0 * local_y / (L * L)
        steer = np.clip(self.wheelbase * curvature, -0.4189, 0.4189)

        return float(steer), self.target_speed


# ============================================================
# Multi-Agent Racing Environment
# ============================================================

class RacingEnv(gym.Env):
    """
    Two-car racing environment.
    Agent 0 (ego) is controlled by the RL policy.
    Agent 1 (opponent) is controlled by pure pursuit.
    """

    def __init__(self, map_name="Spielberg", opponent_speed=3.0, render_mode=None):
        super().__init__()

        import f1tenth_gym
        from f1tenth_gym.envs.env_config import (
            EnvConfig, SimulationConfig, ObservationConfig,
            ResetConfig, ControlConfig,
        )
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.dynamic_models import DynamicModel

        env_config = EnvConfig(
            map_name=map_name,
            num_agents=2,                # TWO CARS
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

        # ---- Config ----
        self.num_beams = 108
        self.lidar_clip = 10.0
        self.max_speed = 5.0       # RL agent can go faster than opponent
        self.min_speed = 0.5
        self.max_steer = 0.4189
        self.max_steps = 6000
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        # ---- Observation: lidar + velocity + yaw_rate + prev_action ----
        obs_dim = self.num_beams + 1 + 1 + 2  # 112
        self.observation_space = gym.spaces.Box(-1, 1, (obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        # ---- Track waypoints ----
        self.waypoints = self._extract_waypoints()
        self.prev_progress = 0.0

        # ---- Opponent controller ----
        self.opponent = PurePursuitOpponent(
            self.waypoints, target_speed=opponent_speed
        )

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
        """Build observation for the ego agent."""
        ego = raw_obs["agent_0"]

        # Lidar (sees both walls AND the opponent car)
        scan = ego["scan"]
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        scan = np.clip(scan, 0, self.lidar_clip) / self.lidar_clip

        # Velocity and yaw rate
        state = ego["std_state"]
        vel = float(state[3]) / 10.0           # Normalize by max ~10 m/s
        yaw_rate = float(state[5]) / 3.14      # Normalize by pi

        return np.concatenate([
            scan,
            [vel, yaw_rate],
            self.prev_action,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed)
        self._prev_raw_obs = raw_obs  # Initialize for opponent
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_progress = 0.0
        return self._build_obs(raw_obs), info

    def step(self, action):
        # ---- Ego action (RL agent) ----
        ego_steer = float(action[0]) * self.max_steer
        ego_speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed

        # ---- Opponent action (pure pursuit) ----
        # Get opponent's pose from previous step
        try:
            opp_state = self._prev_raw_obs["agent_1"]["std_state"]
            opp_x, opp_y = float(opp_state[0]), float(opp_state[1])
            opp_theta = float(opp_state[4])
            opp_steer, opp_speed = self.opponent.get_action(opp_x, opp_y, opp_theta)
        except Exception:
            opp_steer, opp_speed = 0.0, 1.0

        # ---- Step both cars ----
        all_actions = np.array([
            [ego_steer, ego_speed],      # Agent 0: RL
            [opp_steer, opp_speed],      # Agent 1: Pure pursuit
        ], dtype=np.float32)

        raw_obs, _, done, truncated, info = self.env.step(all_actions)
        self._prev_raw_obs = raw_obs
        self.current_step += 1

        # ---- Ego state ----
        ego_state = raw_obs["agent_0"]["std_state"]
        ego_x, ego_y = float(ego_state[0]), float(ego_state[1])
        ego_collision = bool(raw_obs["agent_0"].get("collision", False))

        # ---- Reward ----
        progress = self._get_progress(ego_x, ego_y)
        delta = progress - self.prev_progress
        if delta < -0.5:
            delta += 1.0
        reward = delta * 15.0  # Progress reward

        if ego_collision:
            reward = -10.0
            done = True

        if self.current_step >= self.max_steps:
            truncated = True

        self.prev_progress = progress
        self.prev_action = np.array([action[0], action[1]], dtype=np.float32)
        return self._build_obs(raw_obs), float(reward), done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================
# Train & Evaluate
# ============================================================

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    print("=" * 60)
    print("  F1TENTH: RL vs Pure Pursuit Racing")
    print("=" * 60)
    print(f"  Opponent speed: {args.opp_speed} m/s")
    print(f"  RL max speed:   {5.0} m/s (can overtake!)")
    print("=" * 60)

    env = DummyVecEnv([lambda: RacingEnv(args.map, opponent_speed=args.opp_speed)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.005,       # Slightly more exploration for racing
        verbose=1,
        device=args.device,
        seed=42,
    )

    model.learn(total_timesteps=args.steps, progress_bar=True)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_vs_pp_{args.map}")
    print(f"\nModel saved to models/ppo_vs_pp_{args.map}.zip")
    env.close()


def evaluate(args):
    from stable_baselines3 import PPO

    model_path = args.model or f"models/ppo_vs_pp_{args.map}"
    model = PPO.load(model_path, device="cpu")

    render_mode = "human" if args.render else None
    env = RacingEnv(args.map, opponent_speed=args.opp_speed, render_mode=render_mode)

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if render_mode:
                env.render()

        print(f"  Episode {ep+1}: reward={total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1TENTH: RL vs Pure Pursuit")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--opp-speed", type=float, default=3.0, help="Opponent speed (m/s)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
