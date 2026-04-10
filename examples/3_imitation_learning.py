#!/usr/bin/env python3
"""
=============================================================================
Example 3: Imitation Learning from a Waypoint Follower
=============================================================================

Teaches an RL policy by first imitating an expert (pure pursuit),
then fine-tuning with RL. This is called "Behavioral Cloning → RL".

Why this works:
    - Pure RL from scratch takes millions of steps to learn basic driving
    - Behavioral cloning gives the policy a "warm start" in ~2 minutes
    - RL fine-tuning then improves beyond the expert

Pipeline:
    Step 1: Expert drives → we record (observation, action) pairs
    Step 2: Train a neural network to predict actions from observations (BC)
    Step 3: Use BC weights to initialize PPO, then fine-tune with RL

The expert here is a pure pursuit controller that follows the track
centerline. Students can replace it with any controller.

Usage:
    python3 examples/3_imitation_learning.py                    # Full pipeline
    python3 examples/3_imitation_learning.py --collect-only      # Just collect demos
    python3 examples/3_imitation_learning.py --bc-only           # Just train BC
    python3 examples/3_imitation_learning.py --eval --render     # Evaluate
=============================================================================
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Reuse the wrapper from Example 1
from importlib.util import spec_from_file_location, module_from_spec
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# STEP 1: Collect Expert Demonstrations
# ============================================================

class ExpertCollector:
    """
    Drives the car with pure pursuit and records everything.

    Saves:
        observations: what the policy would see (lidar + state)
        actions: what the expert decided to do (steer, speed)
    """

    def __init__(self, map_name="Spielberg", target_speed=3.0):
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
            render_enabled=False,
        )

        self.env = gym.make("f1tenth_gym:f1tenth-v0", config=env_config)
        self.target_speed = target_speed
        self.num_beams = 108
        self.lidar_clip = 10.0
        self.max_speed = 8.0
        self.min_speed = 0.5
        self.max_steer = 0.4189

        # Get waypoints
        track = self.env.unwrapped.track
        line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
        self.waypoints = np.column_stack([line.xs, line.ys])
        self.wheelbase = 0.33

    def _process_obs(self, raw_obs):
        """Same preprocessing as the RL wrapper."""
        scan = raw_obs["agent_0"]["scan"]
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        scan = np.clip(scan, 0, self.lidar_clip) / self.lidar_clip

        state = raw_obs["agent_0"]["std_state"]
        vel = float(state[3]) / 10.0
        yaw_rate = float(state[5]) / 3.14

        return np.concatenate([scan, [vel, yaw_rate]]).astype(np.float32)

    def _pure_pursuit(self, raw_obs):
        """Get expert action using pure pursuit."""
        state = raw_obs["agent_0"]["std_state"]
        x, y, theta = float(state[0]), float(state[1]), float(state[4])

        # Find closest waypoint
        dists = np.sqrt((self.waypoints[:,0]-x)**2 + (self.waypoints[:,1]-y)**2)
        closest = np.argmin(dists)

        # Lookahead
        lookahead = 1.5
        n = len(self.waypoints)
        target_idx = closest
        for i in range(1, n):
            idx = (closest + i) % n
            d = np.sqrt((self.waypoints[idx,0]-x)**2 + (self.waypoints[idx,1]-y)**2)
            if d >= lookahead:
                target_idx = idx
                break

        # Pure pursuit steering
        target = self.waypoints[target_idx]
        dx, dy = target[0] - x, target[1] - y
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        L = np.sqrt(local_x**2 + local_y**2)
        if L < 0.01:
            steer = 0.0
        else:
            steer = np.clip(self.wheelbase * 2 * local_y / (L*L), -self.max_steer, self.max_steer)

        # Normalize to [-1, 1] (same as RL action space)
        norm_steer = steer / self.max_steer
        norm_speed = 2.0 * (self.target_speed - self.min_speed) / (self.max_speed - self.min_speed) - 1.0

        return np.array([norm_steer, norm_speed], dtype=np.float32)

    def collect(self, num_episodes=100, max_steps=6000):
        """Run expert and record demonstrations."""
        all_obs, all_actions = [], []

        print(f"\nCollecting {num_episodes} expert demonstrations...")
        print(f"  Expert speed: {self.target_speed} m/s")

        for ep in range(num_episodes):
            raw_obs, _ = self.env.reset()
            ep_reward = 0

            for step in range(max_steps):
                obs = self._process_obs(raw_obs)
                action = self._pure_pursuit(raw_obs)

                all_obs.append(obs)
                all_actions.append(action)

                # Step with physical action
                phys_steer = float(action[0]) * self.max_steer
                phys_speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
                raw_obs, _, done, truncated, _ = self.env.step(np.array([[phys_steer, phys_speed]]))

                collision = bool(raw_obs["agent_0"].get("collision", False))
                if collision or done or truncated:
                    break

            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep+1}/{num_episodes}: {step+1} steps")

        self.env.close()

        obs_array = np.array(all_obs, dtype=np.float32)
        act_array = np.array(all_actions, dtype=np.float32)
        print(f"\nCollected {len(obs_array)} transitions")
        return obs_array, act_array


# ============================================================
# STEP 2: Behavioral Cloning (supervised learning)
# ============================================================

class BCPolicy(nn.Module):
    """
    Simple neural network that predicts actions from observations.
    Same architecture as the RL policy so weights transfer directly.
    """

    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


def train_bc(obs, actions, epochs=50, batch_size=256, lr=1e-3):
    """
    Train a policy via behavioral cloning (supervised learning).

    This is just standard supervised learning:
        Input:  observation
        Output: action
        Loss:   MSE between predicted and expert actions
    """
    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]

    # Create PyTorch dataset
    dataset = TensorDataset(
        torch.FloatTensor(obs),
        torch.FloatTensor(actions),
    )

    # 90% train, 10% validation
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Create and train model
    policy = BCPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining Behavioral Cloning...")
    print(f"  Obs dim:  {obs_dim}")
    print(f"  Act dim:  {act_dim}")
    print(f"  Samples:  {len(dataset)} ({n_train} train, {n_val} val)")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Train
        policy.train()
        train_loss = 0
        for batch_obs, batch_act in train_loader:
            pred = policy(batch_obs)
            loss = loss_fn(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                pred = policy(batch_obs)
                val_loss += loss_fn(pred, batch_act).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(policy.state_dict(), "models/bc_policy.pt")

    print(f"  Best val loss: {best_val_loss:.6f}")
    return policy


# ============================================================
# STEP 3: Fine-tune with RL (BC → PPO)
# ============================================================

def finetune_with_rl(bc_policy, map_name, steps=500_000, device="cpu"):
    """
    Initialize PPO with behavioral cloning weights, then fine-tune.

    This gives PPO a massive head start — instead of learning from
    random actions, it starts from a policy that already knows how
    to drive (from imitating the expert).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    # Import wrapper from example 1
    from importlib import import_module
    example1 = {}
    exec(open(os.path.join(os.path.dirname(__file__), "1_simple_ppo.py")).read(), example1)
    F1TenthRLWrapper = example1["F1TenthRLWrapper"]

    # Create environment
    env = DummyVecEnv([lambda: F1TenthRLWrapper(map_name=map_name)])
    env = VecMonitor(env)

    # Create PPO with same architecture
    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-4,     # Lower LR for fine-tuning (don't destroy BC weights)
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.005,
        clip_range=0.15,        # Tighter clipping for fine-tuning
        verbose=1,
        device=device,
    )

    # ---- Transfer BC weights to PPO policy ----
    # PPO's policy network has the same structure as our BC policy
    bc_state = bc_policy.state_dict()

    # Map BC weights to SB3 policy
    sb3_policy = model.policy.mlp_extractor.policy_net
    print(f"\n  BC layers:  {list(bc_state.keys())}")
    print(f"  PPO layers: {[n for n, _ in sb3_policy.named_parameters()]}")

    # Copy weights for matching layers
    try:
        sb3_state = sb3_policy.state_dict()
        for (bc_key, bc_val), (sb3_key, sb3_val) in zip(
            list(bc_state.items())[:-2],  # Skip last layer (different output)
            sb3_state.items()
        ):
            if bc_val.shape == sb3_val.shape:
                sb3_state[sb3_key] = bc_val
                print(f"  Copied: {bc_key} → {sb3_key}")
        sb3_policy.load_state_dict(sb3_state)
        print("  ✓ BC weights transferred to PPO!")
    except Exception as e:
        print(f"  ⚠ Weight transfer failed: {e}")
        print("  Training PPO from scratch instead")

    # Fine-tune
    print(f"\nFine-tuning PPO for {steps:,} steps...")
    model.learn(total_timesteps=steps, progress_bar=True)

    model.save("models/bc_then_ppo")
    print(f"\nModel saved to models/bc_then_ppo.zip")
    env.close()
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="F1TENTH Imitation Learning")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Only collect demos")
    parser.add_argument("--bc-only", action="store_true", help="Only train BC")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("demos", exist_ok=True)

    if args.eval:
        from stable_baselines3 import PPO
        model = PPO.load("models/bc_then_ppo", device="cpu")
        sys.path.insert(0, os.path.dirname(__file__))
        exec_globals = {}
        exec(open(os.path.join(os.path.dirname(__file__), "1_simple_ppo.py")).read(), exec_globals)
        F1TenthRLWrapper = exec_globals["F1TenthRLWrapper"]

        render_mode = "human" if args.render else None
        env = F1TenthRLWrapper(map_name=args.map, render_mode=render_mode)
        for ep in range(5):
            obs, _ = env.reset()
            total_reward, done = 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if render_mode: env.render()
            print(f"  Episode {ep+1}: reward={total_reward:.1f}")
        env.close()
        return

    # ---- Step 1: Collect demonstrations ----
    demo_path = "demos/expert_demos.npz"
    if not os.path.exists(demo_path) or args.collect_only:
        collector = ExpertCollector(args.map, target_speed=args.speed)
        obs, actions = collector.collect(num_episodes=args.episodes)
        np.savez(demo_path, observations=obs, actions=actions)
        print(f"Saved demos to {demo_path}")
        if args.collect_only:
            return

    # ---- Step 2: Train BC ----
    data = np.load(demo_path)
    obs, actions = data["observations"], data["actions"]
    bc_policy = train_bc(obs, actions)

    if args.bc_only:
        return

    # ---- Step 3: Fine-tune with RL ----
    bc_policy = BCPolicy(obs.shape[1], actions.shape[1])
    bc_policy.load_state_dict(torch.load("models/bc_policy.pt", weights_only=True))
    finetune_with_rl(bc_policy, args.map, steps=args.steps, device=args.device)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  1. Expert demos collected")
    print("  2. BC policy trained")
    print("  3. PPO fine-tuned from BC initialization")
    print(f"\n  Evaluate: python3 {__file__} --eval --render")
    print("=" * 60)


if __name__ == "__main__":
    main()
