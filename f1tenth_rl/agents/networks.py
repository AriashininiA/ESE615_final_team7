"""
Neural Network Architectures
==============================
Custom feature extractors and policy networks for F1TENTH RL.

Provides:
    - RacingMLP: Standard MLP for flattened observations
    - LidarCNN: 1D CNN for processing raw lidar scans
    - get_policy_kwargs: Factory for SB3 policy configuration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces

# ---- SB3 Feature Extractors ----
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class LidarCNNExtractor(BaseFeaturesExtractor):
        """
        1D CNN feature extractor for lidar-based observations.

        Processes the lidar portion with 1D convolutions and
        concatenates with other state features.

        Based on TinyLidarNet (3rd place, F1TENTH Grand Prix 2023).

        Parameters
        ----------
        observation_space : spaces.Box
            The observation space.
        features_dim : int
            Output feature dimension.
        n_lidar_beams : int
            Number of lidar beams in the observation.
        channels : list of int
            Number of channels for each conv layer.
        kernel_sizes : list of int
            Kernel sizes for each conv layer.
        strides : list of int
            Strides for each conv layer.
        """

        def __init__(
            self,
            observation_space: spaces.Box,
            features_dim: int = 128,
            n_lidar_beams: int = 108,
            channels: List[int] = [32, 64],
            kernel_sizes: List[int] = [5, 3],
            strides: List[int] = [2, 2],
        ):
            super().__init__(observation_space, features_dim)

            self.n_lidar = n_lidar_beams
            obs_size = observation_space.shape[0]
            self.n_extra = obs_size - n_lidar_beams  # velocity, yaw, prev_action, etc.

            # Build 1D CNN for lidar
            cnn_layers = []
            in_channels = 1
            for ch, ks, st in zip(channels, kernel_sizes, strides):
                cnn_layers.extend([
                    nn.Conv1d(in_channels, ch, kernel_size=ks, stride=st),
                    nn.ReLU(),
                ])
                in_channels = ch
            cnn_layers.append(nn.Flatten())
            self.cnn = nn.Sequential(*cnn_layers)

            # Compute CNN output size
            with torch.no_grad():
                sample = torch.zeros(1, 1, n_lidar_beams)
                cnn_out_size = self.cnn(sample).shape[1]

            # Combine CNN output with extra features
            self.fc = nn.Sequential(
                nn.Linear(cnn_out_size + max(0, self.n_extra), features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            # Split lidar and extra features
            lidar = observations[:, :self.n_lidar].unsqueeze(1)  # (B, 1, N)
            cnn_out = self.cnn(lidar)

            if self.n_extra > 0:
                extra = observations[:, self.n_lidar:]
                combined = torch.cat([cnn_out, extra], dim=1)
            else:
                combined = cnn_out

            return self.fc(combined)

except ImportError:
    # SB3 not installed - skip SB3-specific classes
    LidarCNNExtractor = None


# ---- Standalone Networks (for custom PPO and imitation learning) ----

class RacingMLP(nn.Module):
    """
    MLP network for racing policies.

    Can serve as both actor (policy) and critic (value function).

    Parameters
    ----------
    input_dim : int
        Observation dimension.
    output_dim : int
        Action dimension (for actor) or 1 (for critic).
    hidden_sizes : list of int
        Hidden layer sizes.
    activation : str
        Activation function: "relu" or "tanh".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()

        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), act_fn()])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (orthogonal initialization as in CleanRL)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LidarCNN(nn.Module):
    """
    1D CNN for processing lidar scans.

    Standalone version (not tied to SB3) for use in custom PPO
    and imitation learning.

    Parameters
    ----------
    n_lidar_beams : int
        Number of lidar beams.
    n_extra_features : int
        Number of additional state features.
    output_dim : int
        Output dimension.
    channels : list of int
        Conv layer channels.
    kernel_sizes : list of int
        Conv kernel sizes.
    strides : list of int
        Conv strides.
    """

    def __init__(
        self,
        n_lidar_beams: int = 108,
        n_extra_features: int = 4,
        output_dim: int = 128,
        channels: List[int] = [32, 64],
        kernel_sizes: List[int] = [5, 3],
        strides: List[int] = [2, 2],
    ):
        super().__init__()
        self.n_lidar = n_lidar_beams
        self.n_extra = n_extra_features

        # 1D CNN for lidar
        cnn_layers = []
        in_ch = 1
        for ch, ks, st in zip(channels, kernel_sizes, strides):
            cnn_layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size=ks, stride=st),
                nn.ReLU(),
            ])
            in_ch = ch
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, 1, n_lidar_beams)
            cnn_out = self.cnn(sample).shape[1]

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + n_extra_features, output_dim),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lidar = x[:, :self.n_lidar].unsqueeze(1)
        cnn_out = self.cnn(lidar)
        if self.n_extra > 0:
            extra = x[:, self.n_lidar:]
            combined = torch.cat([cnn_out, extra], dim=1)
        else:
            combined = cnn_out
        return self.fc(combined)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for custom PPO.

    Uses separate networks for policy (actor) and value (critic)
    with an optional shared feature extractor.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    config : dict
        Network configuration section.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: Dict[str, Any]):
        super().__init__()

        net_type = config.get("type", "mlp")
        hidden = config.get("mlp", {}).get("hidden_sizes", [256, 256])
        activation = config.get("mlp", {}).get("activation", "relu")
        pi_layers = config.get("pi_layers", [64, 64])
        vf_layers = config.get("vf_layers", [64, 64])
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        if net_type == "cnn1d":
            cnn_cfg = config.get("cnn1d", {})
            # Guess lidar beams from obs_dim (obs_dim - 4 extra features)
            n_lidar = obs_dim - 4
            self.feature_extractor = LidarCNN(
                n_lidar_beams=n_lidar,
                n_extra_features=4,
                output_dim=cnn_cfg.get("fc_size", 128),
                channels=cnn_cfg.get("channels", [32, 64]),
                kernel_sizes=cnn_cfg.get("kernel_sizes", [5, 3]),
                strides=cnn_cfg.get("strides", [2, 2]),
            )
            feature_dim = cnn_cfg.get("fc_size", 128)
        else:
            # MLP feature extractor
            layers = []
            prev = obs_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), act_fn()])
                prev = h
            self.feature_extractor = nn.Sequential(*layers)
            feature_dim = hidden[-1] if hidden else obs_dim

        # Policy head (actor)
        pi_net = []
        prev = feature_dim
        for h in pi_layers:
            pi_net.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        self.actor_mean = nn.Sequential(*pi_net, nn.Linear(prev, act_dim))

        # Learned log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        # Value head (critic)
        vf_net = []
        prev = feature_dim
        for h in vf_layers:
            vf_net.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        self.critic = nn.Sequential(*vf_net, nn.Linear(prev, 1))

        # Initialize
        self.apply(self._init_weights)
        # Smaller initialization for policy output
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(obs)
        return self.critic(features)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        features = self.feature_extractor(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd.expand_as(mean))

        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features)

        return action, log_prob, entropy, value


# ============================================================
# Factory for SB3 policy_kwargs
# ============================================================

def get_policy_kwargs(network_config: Dict[str, Any], algo_type: str = "ppo") -> Dict[str, Any]:
    """
    Build SB3 policy_kwargs from network configuration.

    Parameters
    ----------
    network_config : dict
        Network section of the config.
    algo_type : str
        Algorithm type ("ppo", "sac", "td3"). SAC/TD3 use "qf" for
        critic networks instead of PPO's "vf".

    Returns
    -------
    dict
        policy_kwargs for SB3 algorithm constructors.
    """
    net_type = network_config.get("type", "mlp")
    pi_layers = network_config.get("pi_layers", [64, 64])
    vf_layers = network_config.get("vf_layers", [64, 64])
    activation = network_config.get("mlp", {}).get("activation", "relu")

    act_fn = nn.ReLU if activation == "relu" else nn.Tanh

    # PPO uses "vf" (value function), SAC/TD3 use "qf" (Q function)
    if algo_type in ("sac", "td3"):
        net_arch = dict(pi=pi_layers, qf=vf_layers)
    else:
        net_arch = dict(pi=pi_layers, vf=vf_layers)

    kwargs = {
        "net_arch": net_arch,
        "activation_fn": act_fn,
    }

    if net_type == "cnn1d" and LidarCNNExtractor is not None:
        cnn_cfg = network_config.get("cnn1d", {})
        kwargs["features_extractor_class"] = LidarCNNExtractor
        kwargs["features_extractor_kwargs"] = {
            "features_dim": cnn_cfg.get("fc_size", 128),
            "channels": cnn_cfg.get("channels", [32, 64]),
            "kernel_sizes": cnn_cfg.get("kernel_sizes", [5, 3]),
            "strides": cnn_cfg.get("strides", [2, 2]),
        }

    return kwargs
