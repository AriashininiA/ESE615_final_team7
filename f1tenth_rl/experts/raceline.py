"""
Raceline Computation
=====================
Extract centerlines from track maps and compute velocity profiles.

For simple tracks (like Levine), uses image processing to find the
centerline. For complex tracks, use TUM's optimizer:
    https://github.com/TUMFTM/global_racetrajectory_optimization

Provides:
    - Centerline extraction from occupancy grid maps
    - Speed profile computation (forward-backward solver)
    - Waypoint file I/O
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import os


class RacelineComputer:
    """
    Compute racing lines from F1TENTH track maps.

    The pipeline:
    1. Load the occupancy grid map image
    2. Extract the centerline via skeletonization
    3. Order and smooth the waypoints
    4. Compute a velocity profile based on curvature
    5. Save as CSV for use in training

    Parameters
    ----------
    map_path : str
        Path to the map YAML file (without extension).
    map_ext : str
        Image file extension.
    """

    def __init__(self, map_path: str, map_ext: str = ".png"):
        self.map_path = map_path
        self.map_ext = map_ext
        self.yaml_path = map_path + ".yaml"
        self.image_path = map_path + map_ext

        # Load map metadata
        self.resolution = 0.05  # m/pixel
        self.origin = [0.0, 0.0, 0.0]
        self._load_yaml()

    def _load_yaml(self):
        """Load map metadata from YAML."""
        import yaml
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path) as f:
                data = yaml.safe_load(f)
            self.resolution = data.get("resolution", 0.05)
            self.origin = data.get("origin", [0.0, 0.0, 0.0])

    def extract_centerline(self, smooth: bool = True) -> np.ndarray:
        """
        Extract the centerline from the map image.

        Uses morphological skeletonization to find the center of
        the free space (track).

        Parameters
        ----------
        smooth : bool
            Whether to smooth the centerline.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Centerline waypoints in world coordinates [x, y].
        """
        from PIL import Image
        from skimage.morphology import skeletonize
        from scipy import ndimage

        # Load and binarize the map
        img = np.array(Image.open(self.image_path).convert("L"))
        # Free space = white (> 200), occupied = black (< 50)
        free_space = (img > 200).astype(np.uint8)

        # Erode slightly to avoid edge artifacts
        from scipy.ndimage import binary_erosion
        free_eroded = binary_erosion(free_space, iterations=3).astype(np.uint8)

        # Skeletonize to get centerline
        skeleton = skeletonize(free_eroded > 0)

        # Get skeleton pixel coordinates
        points = np.argwhere(skeleton)  # (row, col) = (y_pixel, x_pixel)

        if len(points) < 10:
            print("[WARNING] Skeleton extraction found very few points.")
            print("  This may happen if the map has complex geometry.")
            print("  Consider providing waypoints manually.")
            return np.array([[0, 0]], dtype=np.float64)

        # Convert pixel to world coordinates
        world_points = np.zeros((len(points), 2))
        world_points[:, 0] = points[:, 1] * self.resolution + self.origin[0]  # x
        world_points[:, 1] = (img.shape[0] - points[:, 0]) * self.resolution + self.origin[1]  # y

        # Order points along the track
        ordered = self._order_points(world_points)

        # Smooth
        if smooth and len(ordered) > 20:
            ordered = self._smooth_waypoints(ordered, window=11)

        # Subsample to reasonable density (every ~10cm)
        ordered = self._subsample(ordered, spacing=0.1)

        return ordered

    def compute_speed_profile(
        self,
        waypoints: np.ndarray,
        max_speed: float = 8.0,
        max_accel: float = 5.0,
        max_decel: float = 8.0,
        max_lateral_accel: float = 5.0,
    ) -> np.ndarray:
        """
        Compute velocity profile using forward-backward integration.

        At each waypoint, maximum speed is limited by curvature:
            v_max = sqrt(a_lat_max / |kappa|)

        Then a forward pass (acceleration limited) and backward pass
        (braking limited) smooth the profile.

        Parameters
        ----------
        waypoints : np.ndarray, shape (N, 2)
            Ordered waypoints [x, y].
        max_speed : float
            Maximum allowed speed (m/s).
        max_accel : float
            Maximum longitudinal acceleration (m/s^2).
        max_decel : float
            Maximum braking deceleration (m/s^2).
        max_lateral_accel : float
            Maximum lateral acceleration (m/s^2).

        Returns
        -------
        np.ndarray, shape (N, 3)
            Waypoints with velocity: [x, y, velocity].
        """
        n = len(waypoints)
        if n < 3:
            return np.column_stack([waypoints, np.full(n, max_speed * 0.5)])

        # Compute curvature at each point
        kappa = self._compute_curvature(waypoints)

        # Maximum speed from curvature
        v_max = np.full(n, max_speed)
        for i in range(n):
            if abs(kappa[i]) > 1e-6:
                v_curvature = np.sqrt(max_lateral_accel / abs(kappa[i]))
                v_max[i] = min(max_speed, v_curvature)

        # Compute segment lengths
        diffs = np.diff(waypoints, axis=0)
        ds = np.sqrt((diffs ** 2).sum(axis=1))
        # Add wrap-around segment
        ds = np.append(ds, np.sqrt(
            (waypoints[0, 0] - waypoints[-1, 0]) ** 2
            + (waypoints[0, 1] - waypoints[-1, 1]) ** 2
        ))

        # Forward pass: accelerate from previous speed
        v_forward = np.full(n, max_speed)
        v_forward[0] = v_max[0]
        for i in range(1, n):
            v_forward[i] = min(
                v_max[i],
                np.sqrt(v_forward[i - 1] ** 2 + 2 * max_accel * ds[i - 1]),
            )

        # Backward pass: brake from next speed
        v_backward = np.full(n, max_speed)
        v_backward[-1] = v_max[-1]
        for i in range(n - 2, -1, -1):
            v_backward[i] = min(
                v_max[i],
                np.sqrt(v_backward[i + 1] ** 2 + 2 * max_decel * ds[i]),
            )

        # Final profile = minimum of all constraints
        velocity = np.minimum(np.minimum(v_forward, v_backward), v_max)

        return np.column_stack([waypoints, velocity])

    def save_waypoints(self, waypoints: np.ndarray, path: str):
        """
        Save waypoints to CSV file.

        Parameters
        ----------
        waypoints : np.ndarray
            Waypoints array (N, 2) or (N, 3+).
        path : str
            Output CSV path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        if waypoints.shape[1] == 2:
            header = "x,y"
        elif waypoints.shape[1] == 3:
            header = "x,y,velocity"
        else:
            header = ",".join([f"col{i}" for i in range(waypoints.shape[1])])

        np.savetxt(path, waypoints, delimiter=",", header=header, comments="")
        print(f"  Saved {len(waypoints)} waypoints to {path}")

    def compute_and_save(
        self, output_suffix: str = "_centerline.csv", **speed_kwargs
    ) -> np.ndarray:
        """
        Full pipeline: extract centerline, compute speed, save.

        Parameters
        ----------
        output_suffix : str
            Suffix for the output file.
        **speed_kwargs
            Arguments for compute_speed_profile().

        Returns
        -------
        np.ndarray, shape (N, 3)
            Waypoints with velocity.
        """
        print(f"Extracting centerline from {self.image_path}...")
        centerline = self.extract_centerline()
        print(f"  Found {len(centerline)} centerline points")

        print("Computing speed profile...")
        waypoints_with_speed = self.compute_speed_profile(centerline, **speed_kwargs)

        output_path = self.map_path + output_suffix
        self.save_waypoints(waypoints_with_speed, output_path)

        return waypoints_with_speed

    # ---- Internal utilities ----

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order scattered points along the track using nearest-neighbor."""
        n = len(points)
        if n <= 1:
            return points

        ordered = [points[0]]
        remaining = set(range(1, n))

        for _ in range(n - 1):
            if not remaining:
                break
            current = ordered[-1]
            # Find nearest unvisited point
            min_dist = float("inf")
            min_idx = -1
            for idx in remaining:
                d = np.sqrt(
                    (points[idx, 0] - current[0]) ** 2
                    + (points[idx, 1] - current[1]) ** 2
                )
                if d < min_dist:
                    min_dist = d
                    min_idx = idx
            ordered.append(points[min_idx])
            remaining.remove(min_idx)

        return np.array(ordered)

    @staticmethod
    def _smooth_waypoints(
        waypoints: np.ndarray, window: int = 11
    ) -> np.ndarray:
        """Smooth waypoints with a moving average filter."""
        if len(waypoints) < window:
            return waypoints

        # Pad for circular track
        padded = np.concatenate([
            waypoints[-window:],
            waypoints,
            waypoints[:window],
        ])

        from scipy.signal import savgol_filter
        try:
            smoothed_x = savgol_filter(padded[:, 0], window, 3)
            smoothed_y = savgol_filter(padded[:, 1], window, 3)
        except Exception:
            # Fallback to simple moving average
            kernel = np.ones(window) / window
            smoothed_x = np.convolve(padded[:, 0], kernel, mode="same")
            smoothed_y = np.convolve(padded[:, 1], kernel, mode="same")

        result = np.column_stack([
            smoothed_x[window:-window],
            smoothed_y[window:-window],
        ])
        return result

    @staticmethod
    def _subsample(waypoints: np.ndarray, spacing: float = 0.1) -> np.ndarray:
        """Subsample waypoints to approximately uniform spacing."""
        if len(waypoints) < 2:
            return waypoints

        result = [waypoints[0]]
        accumulated = 0.0

        for i in range(1, len(waypoints)):
            d = np.sqrt(
                (waypoints[i, 0] - waypoints[i - 1, 0]) ** 2
                + (waypoints[i, 1] - waypoints[i - 1, 1]) ** 2
            )
            accumulated += d
            if accumulated >= spacing:
                result.append(waypoints[i])
                accumulated = 0.0

        return np.array(result)

    @staticmethod
    def _compute_curvature(waypoints: np.ndarray) -> np.ndarray:
        """Compute curvature at each waypoint using finite differences."""
        n = len(waypoints)
        kappa = np.zeros(n)

        for i in range(n):
            im1 = (i - 1) % n
            ip1 = (i + 1) % n

            x1, y1 = waypoints[im1]
            x2, y2 = waypoints[i]
            x3, y3 = waypoints[ip1]

            # Menger curvature using triangle area
            dx1 = x2 - x1
            dy1 = y2 - y1
            dx2 = x3 - x2
            dy2 = y3 - y2

            cross = abs(dx1 * dy2 - dy1 * dx2)
            d1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
            d2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
            d3 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

            denom = d1 * d2 * d3
            if denom > 1e-10:
                kappa[i] = 2 * cross / denom
            else:
                kappa[i] = 0.0

        return kappa
