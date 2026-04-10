#!/usr/bin/env python3
"""
=============================================================================
PPO From Scratch for F1TENTH — Educational Implementation
=============================================================================

This file implements Proximal Policy Optimization (PPO) from scratch.
No Stable-Baselines3 — just PyTorch, numpy, and the math.

The goal is for students to understand EVERY LINE of PPO:
    1. How the policy network works (actor-critic)
    2. How rollouts are collected
    3. How advantages are computed (GAE)
    4. How the policy is updated (clipped objective)
    5. How the value function is trained

You can WATCH the car learn in real-time with --render.
Early on it crashes everywhere. After ~100k steps it starts surviving.
By 500k steps it drives full laps.

Usage:
    python3 examples/5_ppo_from_scratch.py                    # Train
    python3 examples/5_ppo_from_scratch.py --render           # Train WITH rendering
    python3 examples/5_ppo_from_scratch.py --eval --render    # Evaluate saved model
    python3 examples/5_ppo_from_scratch.py --steps 1000000    # Train longer

Architecture:
    ┌─────────────────────────────────────────┐
    │           Observation (110 dims)        │
    │  [108 lidar beams, velocity, yaw_rate]  │
    └─────────────────┬───────────────────────┘
                      │
              ┌───────▼───────┐
              │  Shared MLP   │
              │  128 → ReLU   │
              │  128 → ReLU   │
              └───┬───────┬───┘
                  │       │
           ┌──────▼──┐ ┌──▼──────┐
           │  Actor  │ │  Critic │
           │ (policy)│ │ (value) │
           │  64→64  │ │  64→64  │
           │  → 2    │ │  → 1    │
           └─────────┘ └─────────┘
           action mean   state value
           [steer,speed]

PPO Algorithm:
    1. Collect N steps of experience using current policy
    2. Compute advantages using GAE (how much better was each action?)
    3. For K epochs:
        a. Sample minibatches from collected experience
        b. Compute new log probabilities and values
        c. Compute ratio = new_prob / old_prob
        d. Clip ratio to [1-ε, 1+ε] to prevent too-large updates
        e. Update policy to maximize clipped objective
        f. Update value function to minimize prediction error

=============================================================================
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import argparse
import os
import time


# ============================================================
# PART 1: The Neural Network (Actor-Critic)
# ============================================================

class ActorCritic(nn.Module):
    """
    The brain of the agent. Two heads on one body:

    Actor (policy):  Given an observation, what action should I take?
                     Outputs a mean and std for each action dimension.
                     Actions are sampled from Normal(mean, std).

    Critic (value):  Given an observation, how good is this state?
                     Outputs a single number V(s) estimating future reward.

    Why both? The critic helps reduce variance in policy gradient estimates.
    Without it, learning is very noisy and slow.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head: outputs action mean
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),   # Squash to [-1, 1]
        )

        # Critic head: outputs state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Log standard deviation (learnable parameter)
        # Starts at -0.5 → std ≈ 0.6 (lots of exploration)
        # During training, this decreases → std → 0 (less exploration)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        """Forward pass through the network."""
        features = self.shared(obs)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(self, obs, deterministic=False):
        """
        Sample an action from the policy.

        Returns:
            action:   The chosen action
            log_prob: Log probability of that action (needed for PPO update)
            value:    Estimated value of the current state
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()  # Convert log_std to std

        if deterministic:
            # At evaluation time: just use the mean (no randomness)
            return action_mean, None, value

        # Create a Normal distribution and sample from it
        dist = Normal(action_mean, std)
        action = dist.sample()

        # Clamp to [-1, 1] to stay in valid action range
        action = action.clamp(-1.0, 1.0)

        # Log probability of this action (sum over action dims)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """
        Re-evaluate actions that were taken in the past.
        Used during the PPO update to compare old vs new policy.

        Returns:
            log_prob: Log probability under CURRENT policy
            value:    Value estimate under CURRENT critic
            entropy:  Entropy of the policy (measure of exploration)
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()

        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy


# ============================================================
# PART 2: The Rollout Buffer (Experience Storage)
# ============================================================

class RolloutBuffer:
    """
    Stores experience collected during rollouts.

    Think of it as a notebook where the agent writes down:
        "I was in state s, took action a, got reward r, ended up in s'"

    After collecting enough experience, we use it to update the policy.
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        """Record one step of experience."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear all stored experience."""
        self.__init__()

    def get(self):
        """Convert lists to tensors for training."""
        return (
            torch.stack(self.observations),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
        )


# ============================================================
# PART 3: Computing Advantages (GAE)
# ============================================================

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation (GAE).

    The advantage A(s, a) answers: "How much better was this action
    compared to what I usually do in this state?"

    If A > 0: This action was better than average → reinforce it
    If A < 0: This action was worse than average → discourage it

    GAE blends 1-step and multi-step advantages using lambda:
        lambda=0: Only look 1 step ahead (low variance, high bias)
        lambda=1: Look all the way to end of episode (high variance, low bias)
        lambda=0.95: Good balance (what most people use)

    Args:
        rewards: [T] tensor of rewards
        values:  [T] tensor of value estimates V(s)
        dones:   [T] tensor of done flags (1 if episode ended)
        gamma:   Discount factor (how much to care about future rewards)
        gae_lambda: GAE smoothing parameter

    Returns:
        advantages: [T] tensor — how good each action was
        returns:    [T] tensor — target for value function (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # No future value at last step
        else:
            next_value = values[t + 1]

        # TD error: how much better was reality than our prediction?
        # delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE: exponentially-weighted sum of TD errors
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        last_gae = advantages[t]

    # Returns = advantages + values (target for critic training)
    returns = advantages + values

    return advantages, returns


# ============================================================
# PART 4: The F1TENTH Environment Wrapper
# ============================================================

class F1TenthEnv(gym.Env):
    """Minimal F1TENTH wrapper for PPO training."""

    def __init__(self, map_name="Spielberg", render_mode=None):
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

        self.env = gym.make("f1tenth_gym:f1tenth-v0", config=env_config,
                            render_mode=render_mode)

        self.num_beams = 108
        self.lidar_clip = 10.0
        self.max_speed = 8.0
        self.min_speed = 0.5
        self.max_steer = 0.4189
        self.max_steps = 3000
        self.current_step = 0

        # Obs: lidar + vel + yaw_rate = 110
        self.observation_space = gym.spaces.Box(-1, 1, (110,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        # Waypoints for progress reward
        try:
            track = self.env.unwrapped.track
            line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
            self.waypoints = np.column_stack([line.xs, line.ys])
        except Exception:
            t = np.linspace(0, 2*np.pi, 100, endpoint=False)
            self.waypoints = np.column_stack([5*np.cos(t), 5*np.sin(t)])
        self.prev_progress = 0.0

    def _process_obs(self, raw_obs):
        scan = raw_obs["agent_0"]["scan"]
        stride = max(1, len(scan) // self.num_beams)
        scan = scan[::stride][:self.num_beams]
        scan = np.clip(scan, 0, self.lidar_clip) / self.lidar_clip

        state = raw_obs["agent_0"]["std_state"]
        vel = float(state[3]) / 10.0
        yaw_rate = float(state[5]) / np.pi

        return np.concatenate([scan, [vel, yaw_rate]]).astype(np.float32)

    def _get_progress(self, x, y):
        dists = np.sqrt((self.waypoints[:,0]-x)**2 + (self.waypoints[:,1]-y)**2)
        return np.argmin(dists) / len(self.waypoints)

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_progress = 0.0
        return self._process_obs(raw_obs), info

    def step(self, action):
        steer = float(action[0]) * self.max_steer
        speed = (float(action[1]) + 1) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed

        raw_obs, _, done, truncated, info = self.env.step(np.array([[steer, speed]]))
        self.current_step += 1

        obs = self._process_obs(raw_obs)
        state = raw_obs["agent_0"]["std_state"]
        x, y = float(state[0]), float(state[1])
        collision = bool(raw_obs["agent_0"].get("collision", False))

        progress = self._get_progress(x, y)
        delta = progress - self.prev_progress
        if delta < -0.5:
            delta += 1.0
        reward = delta * 10.0

        if collision:
            reward = -10.0
            done = True
        if self.current_step >= self.max_steps:
            truncated = True

        self.prev_progress = progress
        return obs, float(reward), done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================
# PART 5: PPO Training Loop
# ============================================================

def train_ppo(args):
    """
    The main PPO training loop.

    Alternates between:
        1. COLLECT: Run the policy in the environment, store experience
        2. UPDATE:  Use collected experience to improve the policy
    """

    # ---- Setup ----
    render_mode = "human" if args.render else None
    env = F1TenthEnv(map_name=args.map, render_mode=render_mode)

    obs_dim = 110  # 108 lidar + vel + yaw_rate
    act_dim = 2    # steer + speed

    policy = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    buffer = RolloutBuffer()

    # ---- Hyperparameters ----
    gamma = 0.99               # Discount factor
    gae_lambda = 0.95          # GAE lambda
    clip_range = 0.2           # PPO clipping parameter (ε)
    entropy_coef = 0.01        # Entropy bonus coefficient
    value_coef = 0.5           # Value loss coefficient
    max_grad_norm = 0.5        # Gradient clipping
    n_steps = 2048             # Steps per rollout
    n_epochs = 10              # SGD epochs per update
    batch_size = 128           # Minibatch size
    total_steps = args.steps

    # ---- Training ----
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs)
    total_timesteps = 0
    episode_reward = 0
    episode_rewards = []
    episode_count = 0
    start_time = time.time()

    print("=" * 60)
    print("  PPO From Scratch — F1TENTH Training")
    print("=" * 60)
    print(f"  Obs dim:     {obs_dim}")
    print(f"  Act dim:     {act_dim}")
    print(f"  Steps:       {total_steps:,}")
    print(f"  Render:      {args.render}")
    print(f"  Parameters:  {sum(p.numel() for p in policy.parameters()):,}")
    print("=" * 60)

    while total_timesteps < total_steps:
        # ==================================================
        # PHASE 1: COLLECT ROLLOUT
        # ==================================================
        # Run the current policy for n_steps and store everything

        policy.eval()  # Set to eval mode (no gradient tracking)

        for step in range(n_steps):
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_tensor)

            # Step the environment
            action_np = action.numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            if args.render:
                env.render()

            # Store experience
            buffer.add(obs_tensor, action, log_prob, reward, value.item(), float(done))

            # Track episode stats
            episode_reward += reward
            total_timesteps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0
                next_obs, _ = env.reset()

            obs = next_obs
            obs_tensor = torch.FloatTensor(obs)

        # ==================================================
        # PHASE 2: COMPUTE ADVANTAGES
        # ==================================================
        # Figure out which actions were good and which were bad

        observations, actions, old_log_probs, rewards, values, dones = buffer.get()
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # Normalize advantages (important for stable training!)
        # This makes the gradient updates more consistent
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================================================
        # PHASE 3: PPO UPDATE
        # ==================================================
        # Update the policy using the clipped objective

        policy.train()  # Set to train mode
        n_samples = len(observations)

        for epoch in range(n_epochs):
            # Shuffle and create minibatches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                # Get batch data
                b_obs = observations[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Evaluate actions under CURRENT policy
                new_log_probs, new_values, entropy = policy.evaluate_actions(b_obs, b_actions)

                # ---- POLICY LOSS (the core of PPO) ----
                # Ratio = π_new(a|s) / π_old(a|s)
                # If ratio > 1: new policy likes this action MORE
                # If ratio < 1: new policy likes this action LESS
                ratio = (new_log_probs - b_old_log_probs).exp()

                # Unclipped objective: ratio * advantage
                surr1 = ratio * b_advantages

                # Clipped objective: clip ratio to [1-ε, 1+ε]
                # This prevents the policy from changing too much in one update
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_advantages

                # Take the minimum (pessimistic bound)
                # This is the key insight of PPO: conservative updates
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- VALUE LOSS ----
                # Train the critic to predict actual returns
                value_loss = ((new_values - b_returns) ** 2).mean()

                # ---- ENTROPY BONUS ----
                # Encourages exploration by keeping the policy "spread out"
                # Without this, the policy might converge to a bad local optimum
                entropy_loss = -entropy.mean()

                # ---- TOTAL LOSS ----
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                # ---- GRADIENT UPDATE ----
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding updates
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        # Clear buffer for next rollout
        buffer.clear()

        # ==================================================
        # LOGGING
        # ==================================================
        if episode_rewards:
            recent = episode_rewards[-10:]
            avg_reward = np.mean(recent)
            elapsed = time.time() - start_time
            fps = total_timesteps / elapsed
            std = policy.log_std.exp().mean().item()

            print(f"  Steps: {total_timesteps:>8,} | "
                  f"Episodes: {episode_count:>4} | "
                  f"Avg reward: {avg_reward:>8.1f} | "
                  f"Std: {std:.3f} | "
                  f"FPS: {fps:.0f}")

    # ---- Save model ----
    os.makedirs("models", exist_ok=True)
    save_path = f"models/ppo_scratch_{args.map}.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    env.close()
    return policy


# ============================================================
# PART 6: Evaluate
# ============================================================

def evaluate(args):
    """Load and evaluate a trained policy."""
    save_path = args.model or f"models/ppo_scratch_{args.map}.pt"
    checkpoint = torch.load(save_path, weights_only=True)

    policy = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    render_mode = "human" if args.render else None
    env = F1TenthEnv(map_name=args.map, render_mode=render_mode)

    print(f"\nEvaluating {save_path}")
    print(f"  Policy std: {policy.log_std.exp().mean().item():.4f}")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_t, deterministic=True)
            action_np = action.numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if render_mode:
                env.render()

        print(f"  Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO From Scratch — F1TENTH")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    parser.add_argument("--render", action="store_true", help="Show real-time rendering")
    parser.add_argument("--map", type=str, default="Spielberg", help="Track name")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--episodes", type=int, default=5, help="Eval episodes")
    parser.add_argument("--model", type=str, default=None, help="Model path for eval")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train_ppo(args)
