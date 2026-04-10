#!/usr/bin/env python3
"""
=============================================================================
Example 7: Custom Observations — Teach the Agent to See Different Things
=============================================================================

The observation is what the neural network receives as input at each step.
By default, the framework gives the agent lidar beams + velocity + yaw rate.
But you can add anything:
    - Distance to the nearest opponent (for racing)
    - Vehicle dynamic state (steering angle, slip angle, lateral velocity)
    - Frenet coordinates (how far from centerline, heading error)
    - Curvature of upcoming track section
    - Time elapsed, lap count, etc.

This example shows THREE custom observations:
    1. Opponent-aware: lidar + velocity + distance/bearing to opponent
    2. Dynamics-aware: lidar + velocity + steering angle + slip angle + lateral velocity
    3. Track-aware: lidar + velocity + Frenet frame (s, ey, ephi) + curvature

Key principle: ALWAYS NORMALIZE your observations to roughly [-1, 1] or [0, 1].
Neural networks learn much faster when all inputs are on similar scales.

Usage:
    python3 examples/7_custom_observations.py --obs opponent    # Opponent-aware
    python3 examples/7_custom_observations.py --obs dynamics    # Vehicle dynamics state
    python3 examples/7_custom_observations.py --obs track       # Track-aware
    python3 examples/7_custom_observations.py --eval --render   # Watch
=============================================================================
"""

import gymnasium as gym
import numpy as np
import argparse
import os


class F1TenthCustomObsWrapper(gym.Env):
    """F1TENTH wrapper demonstrating custom observation spaces."""

    metadata = {"render_modes": ["human", "human_fast"]}

    def __init__(self, map_name="Spielberg", render_mode=None,
                 obs_type="opponent", num_agents=2):
        super().__init__()

        self.obs_type = obs_type
        self.num_agents = num_agents if obs_type == "opponent" else 1

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
            num_agents=self.num_agents,
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
        self.track = self.env.unwrapped.track

        # Get waypoints for expert / opponent
        waypoints = np.column_stack([self.track.centerline.xs,
                                     self.track.centerline.ys])
        self.waypoints = waypoints

        # Pure pursuit for opponent
        if self.num_agents > 1:
            from f1tenth_rl.experts.pure_pursuit import PurePursuitController
            self.opponent = PurePursuitController(
                np.column_stack([waypoints[:, 0], waypoints[:, 1],
                                 np.full(len(waypoints), 3.0)]),
                {"pure_pursuit": {"target_speed": 3.0},
                 "_action_config": {"max_speed": 5.0, "min_speed": 0.5}}
            )

        # ============================================================
        # OBSERVATION SPACE — different sizes for different obs types
        # ============================================================
        self.num_beams = 108
        self.max_speed = 5.0
        self.min_speed = 0.5

        if obs_type == "opponent":
            # lidar (108) + vel + yaw + prev_action (2) + opponent_dist + opponent_bearing
            obs_dim = self.num_beams + 2 + 2 + 2
        elif obs_type == "dynamics":
            # lidar (108) + vel + yaw + prev_action (2) + steering_angle + slip_angle + lateral_vel
            obs_dim = self.num_beams + 2 + 2 + 3
        elif obs_type == "track":
            # lidar (108) + vel + yaw + prev_action (2) + s + ey + ephi + curvature
            obs_dim = self.num_beams + 2 + 2 + 4
        else:
            obs_dim = self.num_beams + 4

        self.observation_space = gym.spaces.Box(-5, 5, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.prev_action = np.zeros(2)

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.prev_action = np.zeros(2)
        return self._make_obs(obs), info

    def step(self, action):
        # Scale ego action
        steer = float(action[0]) * 0.4189
        speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed

        # Build multi-agent action
        if self.num_agents > 1:
            flat_obs = self._flatten_for_pp(obs=self._last_obs)
            opp_action = self.opponent.get_action(flat_obs, ego_idx=1)
            all_actions = np.array([[steer, speed],
                                    [opp_action[0], opp_action[1]]])
        else:
            all_actions = np.array([[steer, speed]])

        obs, _, terminated, truncated, info = self.env.step(all_actions)
        self._last_obs = obs

        collision = bool(obs["agent_0"]["collision"])
        reward = -10.0 if collision else 0.1  # Simple reward; replace with yours

        self.prev_action = action.copy()
        return self._make_obs(obs), reward, terminated, truncated, info

    # ============================================================
    # OBSERVATION BUILDERS
    # ============================================================

    def _make_obs(self, obs):
        self._last_obs = obs

        # Base: downsampled lidar + velocity + yaw rate + prev action
        scan = np.array(obs["agent_0"]["scan"], dtype=np.float32)
        scan = np.clip(scan, 0, 10.0) / 10.0
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        if len(scan) < self.num_beams:
            scan = np.pad(scan, (0, self.num_beams - len(scan)))

        std = obs["agent_0"]["std_state"]
        vel = float(std[3]) / 10.0           # Normalize: 10 m/s → 1.0
        yaw_rate = float(std[5]) / 3.14      # Normalize: π rad/s → 1.0

        base = np.concatenate([scan, [vel, yaw_rate], self.prev_action])

        # ---- OPPONENT-AWARE: add distance and bearing to opponent ----
        if self.obs_type == "opponent":
            ego_x, ego_y = float(std[0]), float(std[1])
            ego_theta = float(std[4])

            if "agent_1" in obs:
                opp_std = obs["agent_1"]["std_state"]
                opp_x, opp_y = float(opp_std[0]), float(opp_std[1])

                # Distance to opponent (normalized by 10m)
                dx, dy = opp_x - ego_x, opp_y - ego_y
                dist = np.sqrt(dx**2 + dy**2)
                opponent_dist = min(dist / 10.0, 1.0)  # 0-1 range

                # Bearing to opponent relative to ego heading
                angle_to_opp = np.arctan2(dy, dx) - ego_theta
                opponent_bearing = angle_to_opp / np.pi   # -1 to 1
            else:
                opponent_dist = 1.0    # Far away
                opponent_bearing = 0.0  # Straight ahead

            return np.concatenate([base, [opponent_dist, opponent_bearing]]).astype(np.float32)

        # ---- DYNAMICS-AWARE: add steering angle, slip angle, lateral velocity ----
        # These tell the policy about the car's current dynamic state.
        # Useful when the car is cornering aggressively or drifting —
        # the policy can learn to recover from slides if it knows
        # its slip angle and lateral velocity.
        elif self.obs_type == "dynamics":
            # std_state indices: [x, y, delta, vel, theta, omega, beta]
            delta = float(std[2]) / 0.4189    # Steering angle, normalized by max steer
            beta = float(std[6]) / 0.5        # Slip angle, normalized (typical range ±0.5 rad)
            vel = float(std[3])
            beta_rad = float(std[6])
            lateral_vel = vel * np.sin(beta_rad)  # Sideways velocity (m/s)
            lateral_vel_norm = lateral_vel / 3.0   # Normalize by ~max expected value

            return np.concatenate([base, [delta, beta, lateral_vel_norm]]).astype(np.float32)

        # ---- TRACK-AWARE: add Frenet frame + curvature ----
        elif self.obs_type == "track":
            ego_x, ego_y = float(std[0]), float(std[1])
            ego_theta = float(std[4])

            # Convert car position to Frenet coordinates
            s, ey, ephi = self.track.cartesian_to_frenet(ego_x, ego_y, ego_theta)

            # Normalize Frenet coordinates
            s_norm = (s % self.track.centerline.length) / self.track.centerline.length  # [0, 1]
            ey_norm = np.clip(ey / 2.0, -1, 1)        # ±2m → [-1, 1]
            ephi_norm = np.clip(ephi / np.pi, -1, 1)   # ±π → [-1, 1]

            # Get track curvature at current position
            if self.track.centerline.ks is not None:
                idx = int(s_norm * len(self.track.centerline.ks)) % len(self.track.centerline.ks)
                curvature = float(self.track.centerline.ks[idx])
                curv_norm = np.clip(curvature * 5.0, -1, 1)  # Typical curvature ~0.2
            else:
                curv_norm = 0.0

            return np.concatenate([base, [s_norm, ey_norm, ephi_norm, curv_norm]]).astype(np.float32)

        return base.astype(np.float32)

    def _flatten_for_pp(self, obs):
        """Convert to dict format for pure pursuit."""
        result = {"scans": [], "poses_x": [], "poses_y": [],
                  "poses_theta": [], "linear_vels_x": [], "ang_vels_z": []}
        for i in range(self.num_agents):
            agent_key = f"agent_{i}"
            if agent_key in obs:
                std = obs[agent_key]["std_state"]
                result["scans"].append(np.array(obs[agent_key]["scan"]))
                result["poses_x"].append(float(std[0]))
                result["poses_y"].append(float(std[1]))
                result["poses_theta"].append(float(std[4]))
                vel, beta = float(std[3]), float(std[6])
                result["linear_vels_x"].append(vel * np.cos(beta))
                result["ang_vels_z"].append(float(std[5]))
        return result

    def render(self):
        if self.render_mode:
            return self.env.render()

    def close(self):
        self.env.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Example 7: Custom Observations")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--obs", type=str, default="opponent",
                        choices=["opponent", "dynamics", "track"])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model or f"models/ex7_{args.obs}_{args.map}"
    num_agents = 2 if args.obs == "opponent" else 1

    if args.eval:
        from stable_baselines3 import PPO
        import time
        render_mode = "human" if args.render else None
        env = F1TenthCustomObsWrapper(map_name=args.map, render_mode=render_mode,
                                       obs_type=args.obs, num_agents=num_agents)
        model = PPO.load(model_path, device="cpu")
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
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        print(f"\nTraining with {args.obs.upper()} observations on {args.map}")
        env = DummyVecEnv([lambda: F1TenthCustomObsWrapper(
            map_name=args.map, obs_type=args.obs, num_agents=num_agents)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        model = PPO("MlpPolicy", env, verbose=1, device=args.device,
                     learning_rate=3e-4, n_steps=2048, batch_size=128)
        model.learn(total_timesteps=args.steps)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
