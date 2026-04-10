"""
Expert Controllers for F1TENTH.

Provides:
    - PurePursuitController: Waypoint-following expert for demo collection
    - RacelineComputer: Compute racing lines from track maps
    - DemoCollector: Collect expert demonstrations for imitation learning
"""

from f1tenth_rl.experts.pure_pursuit import PurePursuitController
from f1tenth_rl.experts.raceline import RacelineComputer
from f1tenth_rl.experts.demo_collector import DemoCollector

__all__ = [
    "PurePursuitController",
    "RacelineComputer",
    "DemoCollector",
]
