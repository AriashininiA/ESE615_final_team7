"""
Imitation Learning (Behavioral Cloning)
=======================================
Instead of learning from scratch through trial and error, the agent
can first learn by watching an expert drive. This is called "behavioral
cloning" (BC) — the neural network learns to predict what action the
expert would take, given the same observation.

Think of it like a student driver watching an instructor first, then
practicing on their own. The BC phase gives the agent a reasonable
starting policy, and then RL fine-tuning makes it even better.

Workflow:
    1. Collect demonstrations: an expert (pure pursuit) drives 100 laps
       while we record every (observation, action) pair.

    2. Train behavioral cloning: we train a neural network using
       supervised learning (MSE loss between predicted and expert actions).
       This takes a few minutes and produces a policy that can already
       drive, though usually not as well as the expert.

    3. Fine-tune with RL (optional): initialize PPO with the BC weights
       and continue training with RL. The agent starts from a competent
       policy instead of random, so RL converges much faster.

Usage:
    # Step 1: Collect demos
    python scripts/collect_demos.py --episodes 100

    # Step 2a: BC only (no RL)
    python scripts/train.py --bc-pretrain demos/expert_demos.npz --bc-only

    # Step 2b: BC → PPO (recommended)
    python scripts/train.py --bc-pretrain demos/expert_demos.npz
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from f1tenth_rl.agents.networks import RacingMLP, LidarCNN, ActorCritic


class ImitationTrainer:
    """
    Behavioral cloning trainer.

    Trains a policy network to imitate expert demonstrations via
    supervised learning. The trained weights can then initialize
    an RL algorithm for fine-tuning.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Example
    -------
    >>> trainer = ImitationTrainer(config)
    >>> trainer.load_demonstrations("demos/expert_demos.npz")
    >>> trainer.train_bc()
    >>> trainer.save("checkpoints/bc_pretrained")
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        bc_cfg = config.get("imitation", {}).get("bc", {})

        self.epochs = bc_cfg.get("epochs", 100)
        self.learning_rate = bc_cfg.get("learning_rate", 1e-3)
        self.batch_size = bc_cfg.get("batch_size", 256)
        self.val_split = bc_cfg.get("validation_split", 0.1)

        # Device
        device_str = config["experiment"].get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = None
        self.obs_data = None
        self.act_data = None

    def load_demonstrations(self, path: str):
        """
        Load expert demonstrations from NPZ file.

        Expected format:
            observations: np.ndarray, shape (N, obs_dim)
            actions: np.ndarray, shape (N, act_dim)

        Parameters
        ----------
        path : str
            Path to .npz demonstration file.
        """
        data = np.load(path)
        self.obs_data = data["observations"].astype(np.float32)
        self.act_data = data["actions"].astype(np.float32)

        print(f"Loaded {len(self.obs_data)} demonstration transitions")
        print(f"  Observation shape: {self.obs_data.shape}")
        print(f"  Action shape: {self.act_data.shape}")

    def train_bc(self, obs_data=None, act_data=None) -> nn.Module:
        """
        Train behavioral cloning policy.

        Parameters
        ----------
        obs_data : np.ndarray, optional
            Override observations (if not loaded via load_demonstrations).
        act_data : np.ndarray, optional
            Override actions.

        Returns
        -------
        nn.Module
            Trained policy network.
        """
        if obs_data is not None:
            self.obs_data = obs_data.astype(np.float32)
            self.act_data = act_data.astype(np.float32)

        if self.obs_data is None:
            raise ValueError("No demonstrations loaded. Call load_demonstrations() first.")

        obs_dim = self.obs_data.shape[1]
        act_dim = self.act_data.shape[1]

        # ---- Create model ----
        net_cfg = self.config.get("network", {})
        net_type = net_cfg.get("type", "mlp")
        hidden = net_cfg.get("mlp", {}).get("hidden_sizes", [256, 256])

        if net_type == "cnn1d":
            n_lidar = obs_dim - 4  # Guess extra features
            cnn_cfg = net_cfg.get("cnn1d", {})
            self.model = nn.Sequential(
                LidarCNN(
                    n_lidar_beams=n_lidar,
                    n_extra_features=4,
                    output_dim=cnn_cfg.get("fc_size", 128),
                ),
                nn.Linear(cnn_cfg.get("fc_size", 128), act_dim),
                nn.Tanh(),  # Output in [-1, 1]
            ).to(self.device)
        else:
            self.model = nn.Sequential(
                RacingMLP(obs_dim, act_dim, hidden_sizes=hidden),
                nn.Tanh(),  # Output in [-1, 1]
            ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # ---- Create datasets ----
        dataset = TensorDataset(
            torch.tensor(self.obs_data),
            torch.tensor(self.act_data),
        )
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # ---- Training loop ----
        print(f"\nBehavioral Cloning Training")
        print(f"  Model: {net_type}")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")
        print(f"  Epochs: {self.epochs}")

        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for obs_batch, act_batch in train_loader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)

                pred = self.model(obs_batch)
                loss = criterion(pred, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for obs_batch, act_batch in val_loader:
                    obs_batch = obs_batch.to(self.device)
                    act_batch = act_batch.to(self.device)
                    pred = self.model(obs_batch)
                    val_loss += criterion(pred, act_batch).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs} | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"Best: {best_val_loss:.6f}"
                )

        print(f"  BC training complete. Best val loss: {best_val_loss:.6f}")
        return self.model

    def save(self, path: str):
        """Save BC model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, path + ".pt")
        print(f"  BC model saved to {path}.pt")

    def load(self, path: str, obs_dim: int, act_dim: int):
        """Load BC model."""
        checkpoint = torch.load(path + ".pt", map_location=self.device)
        # Reconstruct model architecture
        net_cfg = self.config.get("network", {})
        hidden = net_cfg.get("mlp", {}).get("hidden_sizes", [256, 256])
        self.model = nn.Sequential(
            RacingMLP(obs_dim, act_dim, hidden_sizes=hidden),
            nn.Tanh(),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get action from BC policy."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.model(obs_t)
            return action.cpu().numpy().squeeze(0)

    def init_sb3_from_bc(self, sb3_model, bc_path: str):
        """
        Initialize an SB3 model's policy with BC-pretrained weights.

        This is the recommended workflow:
        1. Train BC on expert demos
        2. Create SB3 PPO/SAC model
        3. Copy BC weights into SB3 policy
        4. Fine-tune with RL

        Parameters
        ----------
        sb3_model : SB3 algorithm instance
            The SB3 model to initialize.
        bc_path : str
            Path to BC checkpoint.
        """
        checkpoint = torch.load(bc_path + ".pt", map_location=self.device)
        bc_state = checkpoint["model_state_dict"]

        # Copy matching weights from BC to SB3 policy
        sb3_policy = sb3_model.policy
        sb3_state = sb3_policy.state_dict()

        copied = 0
        for key in bc_state:
            # Try to find matching keys in SB3 policy
            for sb3_key in sb3_state:
                if bc_state[key].shape == sb3_state[sb3_key].shape:
                    sb3_state[sb3_key] = bc_state[key]
                    copied += 1
                    break

        sb3_policy.load_state_dict(sb3_state)
        print(f"  Copied {copied} parameter tensors from BC to SB3 policy")
