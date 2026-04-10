"""
PPO From Scratch (No SB3)
=========================
This is a clean, readable implementation of Proximal Policy Optimization
with no library dependencies beyond PyTorch and numpy.

Why does this exist? The SB3 PPO is great for getting results, but it
hides most of the algorithm behind abstractions. If you're a student
learning RL, you want to see what's actually happening. This file
lets you trace the entire PPO algorithm step by step.

How PPO works (in plain English):
    1. The agent drives around collecting experience (observations,
       actions, rewards) for a fixed number of steps
    2. For each step, it computes "how much better was the action I
       took compared to what I expected?" (these are called advantages)
    3. It then updates the neural network to make good actions more
       likely and bad actions less likely
    4. The "proximal" part: it limits how much the policy can change
       in a single update, which prevents catastrophic forgetting

This implementation is based on CleanRL's ppo_continuous_action.py.
It produces identical results to SB3's PPO but in ~300 lines of
transparent, modifiable code.

Usage:
    python scripts/train.py --algo custom_ppo --total-steps 500000
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

from f1tenth_rl.envs.wrapper import make_env
from f1tenth_rl.agents.networks import ActorCritic


class CustomPPO:
    """
    CleanRL-style PPO for continuous action spaces.

    Everything happens in this class - no hidden abstractions.
    Students can read, understand, and modify every line.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        algo_cfg = config["algorithm"].get("custom_ppo", {})

        # ---- Hyperparameters ----
        self.learning_rate = algo_cfg.get("learning_rate", 3e-4)
        self.num_steps = algo_cfg.get("num_steps", 2048)
        self.num_minibatches = algo_cfg.get("num_minibatches", 32)
        self.update_epochs = algo_cfg.get("update_epochs", 10)
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.gae_lambda = algo_cfg.get("gae_lambda", 0.95)
        self.clip_coef = algo_cfg.get("clip_coef", 0.2)
        self.ent_coef = algo_cfg.get("ent_coef", 0.01)
        self.vf_coef = algo_cfg.get("vf_coef", 0.5)
        self.max_grad_norm = algo_cfg.get("max_grad_norm", 0.5)
        self.anneal_lr = algo_cfg.get("anneal_lr", True)
        self.normalize_advantages = algo_cfg.get("normalize_advantages", True)

        self.total_timesteps = config["algorithm"]["total_timesteps"]
        self.seed = config["experiment"].get("seed", 42)

        # ---- Device ----
        device_str = config["experiment"].get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # ---- Environments ----
        self.num_envs = config["env"].get("num_envs", 8)

        # ---- Logging ----
        self.exp_name = config["experiment"].get("name", "custom_ppo")
        self.log_dir = config["experiment"].get("log_dir", "logs")
        self.save_dir = config["experiment"].get("save_dir", "checkpoints")

        # ---- State ----
        self.agent = None
        self.optimizer = None

    def train(self):
        """Run the full PPO training loop."""

        # ---- Seed everything ----
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ---- Create vectorized environment ----
        import gymnasium
        envs = gymnasium.vector.SyncVectorEnv(
            [make_env(self.config, rank=i, seed=self.seed) for i in range(self.num_envs)]
        )

        obs_dim = envs.single_observation_space.shape[0]
        act_dim = envs.single_action_space.shape[0]

        # ---- Create actor-critic network ----
        self.agent = ActorCritic(
            obs_dim, act_dim, self.config["network"]
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.learning_rate, eps=1e-5
        )

        # ---- Logging ----
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(self.log_dir, self.exp_name))

        # ---- Rollout storage ----
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // self.num_minibatches
        num_updates = self.total_timesteps // batch_size

        obs_buf = torch.zeros((self.num_steps, self.num_envs, obs_dim)).to(self.device)
        act_buf = torch.zeros((self.num_steps, self.num_envs, act_dim)).to(self.device)
        logprob_buf = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        reward_buf = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        done_buf = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        value_buf = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # ---- Training loop ----
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=self.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        # Episode tracking
        episode_returns = []
        episode_lengths = []

        print(f"\n{'='*60}")
        print(f"  Custom PPO Training")
        print(f"{'='*60}")
        print(f"  Obs dim:      {obs_dim}")
        print(f"  Act dim:      {act_dim}")
        print(f"  Envs:         {self.num_envs}")
        print(f"  Batch size:   {batch_size}")
        print(f"  Updates:      {num_updates}")
        print(f"  Device:       {self.device}")
        print(f"{'='*60}\n")

        for update in range(1, num_updates + 1):
            # ---- Learning rate annealing ----
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lr = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr

            # ============================================================
            # PHASE 1: Collect rollout
            # ============================================================
            for step in range(self.num_steps):
                global_step += self.num_envs
                obs_buf[step] = next_obs
                done_buf[step] = next_done

                # Get action from policy
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    value_buf[step] = value.flatten()

                act_buf[step] = action
                logprob_buf[step] = logprob

                # Step environment
                next_obs_np, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                reward_buf[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(self.device)
                next_done = torch.tensor(done, dtype=torch.float32).to(self.device)

                # Track episode statistics
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is not None and "episode" in info:
                            ep_return = info["episode"]["r"]
                            ep_length = info["episode"]["l"]
                            episode_returns.append(ep_return)
                            episode_lengths.append(ep_length)
                            writer.add_scalar("rollout/ep_return", ep_return, global_step)
                            writer.add_scalar("rollout/ep_length", ep_length, global_step)

            # ============================================================
            # PHASE 2: Compute advantages (GAE)
            # ============================================================
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).flatten()
                advantages = torch.zeros_like(reward_buf).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = value_buf[t + 1]

                    delta = (
                        reward_buf[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - value_buf[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + value_buf

            # ============================================================
            # PHASE 3: Optimize policy and value function
            # ============================================================
            # Flatten batch
            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = act_buf.reshape(-1, act_dim)
            b_logprobs = logprob_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)

            # Optimization loop
            clipfracs = []
            for epoch in range(self.update_epochs):
                # Random permutation for minibatches
                b_inds = torch.randperm(batch_size, device=self.device)

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    # Policy loss (clipped surrogate)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    if self.normalize_advantages:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    # Track clip fraction
                    with torch.no_grad():
                        clipfracs.append(
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        )

            # ============================================================
            # Logging
            # ============================================================
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)

            # Print progress
            if update % 10 == 0 or update == 1:
                avg_return = np.mean(episode_returns[-100:]) if episode_returns else 0
                print(
                    f"  Update {update}/{num_updates} | "
                    f"Steps: {global_step:,} | "
                    f"Return: {avg_return:.2f} | "
                    f"SPS: {sps} | "
                    f"PG Loss: {pg_loss.item():.4f} | "
                    f"VF Loss: {v_loss.item():.4f}"
                )

            # Save checkpoint
            if update % max(num_updates // 10, 1) == 0:
                self.save(os.path.join(
                    self.save_dir, f"{self.exp_name}_step{global_step}"
                ))

        # ---- Save final model ----
        final_path = os.path.join(self.save_dir, f"{self.exp_name}_final")
        self.save(final_path)
        print(f"\nTraining complete! Final model: {final_path}")

        envs.close()
        writer.close()

    def save(self, path: str):
        """Save the actor-critic network."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path + ".pt")

    def load(self, path: str, obs_dim: int, act_dim: int):
        """Load a trained model."""
        checkpoint = torch.load(path + ".pt", map_location=self.device)
        self.agent = ActorCritic(
            obs_dim, act_dim, self.config["network"]
        ).to(self.device)
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        self.agent.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from trained policy."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            if deterministic:
                features = self.agent.feature_extractor(obs_t)
                action = self.agent.actor_mean(features)
            else:
                action, _, _, _ = self.agent.get_action_and_value(obs_t)
            return action.cpu().numpy().squeeze(0)
