"""
Observation Space Builder
=========================
Constructs the input vector that the neural network receives at each step.

Think of this as the agent's "eyes" — it determines what information
the policy has access to when deciding what to do. The observation
is a flat numpy array that gets fed into the neural network.

Available observation types (set in config: observation.type):

    lidar_only (simplest):
        Just the downsampled lidar scan. The agent sees distances to
        walls in all directions but doesn't know its own speed.
        Example: [108 beams] = 108 dimensions

    lidar_state (recommended):
        Lidar + velocity + yaw rate + previous action. The agent can
        see walls AND knows how fast it's going and what it did last step.
        Example: [108 beams, vel, yaw, prev_steer, prev_speed] = 112 dims

    lidar_waypoint (localization required):
        Everything in lidar_state plus waypoint-relative features.
        The agent knows WHERE on the track it is and can see upcoming
        waypoints. Requires a particle filter on the real car.
        Example: [108 beams, vel, yaw, prev_act, 5×(dist,heading)] = 122 dims

    waypoint_only (simulation only):
        No lidar at all, just waypoint features and state. Useful for
        quick experiments but not deployable on a real car.

All values are normalized to neural-network-friendly ranges:
    - Lidar: divided by clip distance → [0, 1]
    - Velocity: divided by 10 → roughly [0, 1] for typical speeds
    - Yaw rate: divided by π → roughly [-1, 1]
    - Previous actions: already in [-1, 1]
    - Waypoint distances: divided by spacing → roughly [0, 1]
    - Waypoint headings: divided by π → [-1, 1]
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional
from collections import deque


class ObservationBuilder:
    """
    Builds flat observation vectors from raw F1TENTH Gym observations.

    The F1TENTH Gym returns dict observations with keys like 'scans',
    'poses_x', 'linear_vels_x', etc. This class converts them into
    flat numpy arrays suitable for RL policies.

    Parameters
    ----------
    config : dict
        Observation configuration section from YAML.
    num_agents : int
        Number of agents in the environment.

    Example
    -------
    >>> builder = ObservationBuilder(config["observation"], num_agents=1)
    >>> obs_space = builder.get_observation_space()
    >>> obs = builder.build(raw_obs_dict, ego_idx=0, prev_action=np.zeros(2))
    """

    def __init__(self, config: Dict[str, Any], num_agents: int = 1):
        self.config = config
        self.num_agents = num_agents
        self.obs_type = config.get("type", "lidar_state")

        # Lidar settings
        self.num_beams = config.get("lidar_beams", 108)
        self.lidar_clip = config.get("lidar_clip", 10.0)
        self.lidar_normalize = config.get("lidar_normalize", True)
        # dev-humble may have different beam counts depending on LiDARConfig
        self.raw_beams = config.get("_actual_raw_beams", 1080)

        # Compute downsample stride
        self.downsample_stride = max(1, self.raw_beams // self.num_beams)
        # Actual number of beams after downsampling
        self.actual_beams = len(range(0, self.raw_beams, self.downsample_stride))

        # State features
        self.include_velocity = config.get("include_velocity", True)
        self.include_yaw_rate = config.get("include_yaw_rate", True)
        self.include_steering = config.get("include_steering", False)
        self.include_prev_action = config.get("include_prev_action", True)

        # Waypoint features
        self.num_waypoints = config.get("num_waypoints", 5)
        self.waypoint_spacing = config.get("waypoint_spacing", 0.5)
        self.waypoints = None  # Set externally via set_waypoints()

        # Frame stacking
        self.frame_stack = config.get("frame_stack", 1)
        self.frame_buffer = None

        # Compute observation dimension
        self._obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        """Compute the total observation dimension."""
        dim = 0

        # Lidar component
        if self.obs_type in ["lidar_only", "lidar_state", "lidar_waypoint"]:
            dim += self.actual_beams

        # State component
        if self.obs_type in ["lidar_state", "lidar_waypoint", "waypoint_only"]:
            if self.include_velocity:
                dim += 1
            if self.include_yaw_rate:
                dim += 1
            if self.include_steering:
                dim += 1

        # Previous action
        if self.include_prev_action:
            dim += 2

        # Waypoint features (relative distance, heading error per waypoint)
        if self.obs_type in ["lidar_waypoint", "waypoint_only"]:
            dim += self.num_waypoints * 2  # (distance, heading_error) per waypoint

        # Frame stacking multiplies the dimension
        dim *= self.frame_stack

        return dim

    def get_observation_space(self) -> spaces.Box:
        """
        Return the gymnasium observation space.

        Returns
        -------
        spaces.Box
            Observation space with shape (obs_dim,).
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    def set_waypoints(self, waypoints: np.ndarray):
        """
        Set waypoints for waypoint-based observations.

        Parameters
        ----------
        waypoints : np.ndarray, shape (N, 2) or (N, 3+)
            Waypoint positions. Columns: [x, y, ...].
        """
        self.waypoints = waypoints[:, :2]  # Only need x, y

    def reset(self):
        """Reset observation builder state (call on env reset)."""
        if self.frame_stack > 1:
            self.frame_buffer = deque(maxlen=self.frame_stack)

    def build(
        self,
        obs_dict: Dict[str, Any],
        ego_idx: int = 0,
        prev_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a flat observation vector from raw F1TENTH observations.

        Parameters
        ----------
        obs_dict : dict
            Raw observation dictionary from F1TENTH Gym.
        ego_idx : int
            Index of the ego agent.
        prev_action : np.ndarray, shape (2,), optional
            Previous action [steer, speed].

        Returns
        -------
        np.ndarray, shape (obs_dim,)
            Flat observation vector.
        """
        components = []

        # ---- Lidar ----
        if self.obs_type in ["lidar_only", "lidar_state", "lidar_waypoint"]:
            scan = np.array(obs_dict["scans"][ego_idx], dtype=np.float32)
            # Downsample
            scan = scan[::self.downsample_stride]
            # Clip and normalize
            scan = np.clip(scan, 0.0, self.lidar_clip)
            if self.lidar_normalize:
                scan = scan / self.lidar_clip
            components.append(scan)

        # ---- State features ----
        if self.obs_type in ["lidar_state", "lidar_waypoint", "waypoint_only"]:
            if self.include_velocity:
                vel = float(obs_dict["linear_vels_x"][ego_idx])
                # Normalize velocity to roughly [-1, 1]
                components.append(np.array([vel / 10.0], dtype=np.float32))

            if self.include_yaw_rate:
                yaw_rate = float(obs_dict["ang_vels_z"][ego_idx])
                components.append(np.array([yaw_rate / 3.14], dtype=np.float32))

            if self.include_steering:
                # Steering angle from the internal state if available
                steer = float(obs_dict.get("steering_angles", [0.0])[ego_idx])
                components.append(np.array([steer / 0.4189], dtype=np.float32))

        # ---- Previous action ----
        if self.include_prev_action:
            if prev_action is not None:
                # Already in normalized scale
                components.append(prev_action.astype(np.float32))
            else:
                components.append(np.zeros(2, dtype=np.float32))

        # ---- Waypoint features ----
        if self.obs_type in ["lidar_waypoint", "waypoint_only"]:
            wp_features = self._compute_waypoint_features(obs_dict, ego_idx)
            components.append(wp_features)

        # Concatenate all components
        obs = np.concatenate(components).astype(np.float32)

        # Frame stacking
        if self.frame_stack > 1:
            if self.frame_buffer is None or len(self.frame_buffer) == 0:
                # Fill buffer with copies of the first frame
                for _ in range(self.frame_stack):
                    self.frame_buffer.append(obs.copy())
            else:
                self.frame_buffer.append(obs.copy())
            obs = np.concatenate(list(self.frame_buffer))

        return obs

    def _compute_waypoint_features(
        self, obs_dict: Dict, ego_idx: int
    ) -> np.ndarray:
        """
        Compute waypoint-relative features (distance, heading error).

        Returns
        -------
        np.ndarray, shape (num_waypoints * 2,)
            [dist_1, heading_err_1, dist_2, heading_err_2, ...] for each
            lookahead waypoint.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return np.zeros(self.num_waypoints * 2, dtype=np.float32)

        ego_x = float(obs_dict["poses_x"][ego_idx])
        ego_y = float(obs_dict["poses_y"][ego_idx])
        ego_theta = float(obs_dict["poses_theta"][ego_idx])

        # Find closest waypoint
        dists = np.sqrt(
            (self.waypoints[:, 0] - ego_x) ** 2
            + (self.waypoints[:, 1] - ego_y) ** 2
        )
        closest_idx = np.argmin(dists)

        # Get lookahead waypoints
        features = []
        n_wp = len(self.waypoints)
        for i in range(self.num_waypoints):
            wp_idx = (closest_idx + i + 1) % n_wp
            wp = self.waypoints[wp_idx]

            # Distance to waypoint
            dx = wp[0] - ego_x
            dy = wp[1] - ego_y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # Heading error: angle between car heading and direction to waypoint
            wp_angle = np.arctan2(dy, dx)
            heading_err = self._normalize_angle(wp_angle - ego_theta)

            features.extend([
                dist / 10.0,            # Normalize distance
                heading_err / np.pi,    # Normalize to [-1, 1]
            ])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
