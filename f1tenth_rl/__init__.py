"""
F1TENTH RL Training Framework
==============================
A modular framework for training reinforcement learning policies
in the F1TENTH Gym simulator and deploying them on real cars.

Modules:
    envs    - Gymnasium wrappers, observation/action/reward design
    agents  - RL algorithms (SB3, custom PPO, imitation learning)
    experts - Expert controllers (pure pursuit, raceline computation)
    utils   - Waypoints, callbacks, logging
    ros2    - ROS2 inference nodes for sim2real deployment
"""

__version__ = "1.0.0"
