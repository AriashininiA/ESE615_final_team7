"""
Waypoint Utilities
===================
Loading, processing, and analyzing waypoint files.
"""

import numpy as np
from typing import Optional, Tuple
import os


def load_waypoints_from_file(path: str) -> np.ndarray:
    """
    Load waypoints from a CSV file.

    Handles various formats:
    - Simple [x, y] columns
    - TUM format [s, x, y, psi, kappa, vx, ax]
    - F1TENTH racetracks format

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    np.ndarray
        Waypoints with all available columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Waypoint file not found: {path}")

    # Try loading with header
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except (ValueError, IndexError):
        pass

    # Try without header
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception:
        pass

    # Try semicolon delimiter
    data = np.loadtxt(path, delimiter=";", skiprows=1)
    return data


def compute_progress(
    position: np.ndarray,
    waypoints: np.ndarray,
) -> Tuple[float, int, float]:
    """
    Compute track progress from position and waypoints.

    Parameters
    ----------
    position : np.ndarray, shape (2,)
        Current [x, y] position.
    waypoints : np.ndarray, shape (N, 2+)
        Waypoints.

    Returns
    -------
    progress : float
        Fractional progress [0, 1].
    closest_idx : int
        Index of closest waypoint.
    crosstrack : float
        Distance to closest waypoint.
    """
    wp = waypoints[:, :2]
    dists = np.sqrt(
        (wp[:, 0] - position[0]) ** 2
        + (wp[:, 1] - position[1]) ** 2
    )
    closest_idx = int(np.argmin(dists))
    crosstrack = float(dists[closest_idx])
    progress = closest_idx / len(wp)

    return progress, closest_idx, crosstrack


def interpolate_waypoints(
    waypoints: np.ndarray,
    num_points: int = 1000,
) -> np.ndarray:
    """
    Interpolate waypoints to a finer resolution.

    Parameters
    ----------
    waypoints : np.ndarray, shape (N, 2+)
        Input waypoints.
    num_points : int
        Desired number of output points.

    Returns
    -------
    np.ndarray, shape (num_points, 2+)
        Interpolated waypoints.
    """
    from scipy.interpolate import interp1d

    # Compute cumulative arc length
    diffs = np.diff(waypoints[:, :2], axis=0)
    ds = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])

    # Interpolate each column
    s_new = np.linspace(0, s[-1], num_points)
    result = np.zeros((num_points, waypoints.shape[1]))

    for col in range(waypoints.shape[1]):
        f = interp1d(s, waypoints[:, col], kind="cubic")
        result[:, col] = f(s_new)

    return result
