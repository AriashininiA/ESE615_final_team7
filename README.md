# ESE615 Final Project - Team 7

This repository was originally forked from `f1tenth_rl_humble`.

The project investigates sim-to-real-oriented reinforcement learning for F1TENTH autonomous racing, using behavioral cloning (BC), PPO fine-tuning, and reward shaping on the Levine 2nd floor map (`levine_blocked`).

## Baseline

We first ran the initial test setup on the Levine 2nd floor map and found that **BC + CTH reward + PPO fine-tuning** was the current strongest baseline.

The baseline model was evaluated over 20 deterministic episodes.

| **Metric** | **Result** |
| --- | --- |
| Lap completion | **100%** / 20 of 20 laps completed |
| Crash rate | **0.0%** / 0 of 20 episodes |
| Average speed | **2.967 m/s** |
| Average lap time | **20.85 s** |
| Best lap time | **20.73 s** |
| Mean progress | **100.00%** |
| Mean return | **1223.22 +/- 5.70** |

## Problem Observed

The BC + CTH + PPO baseline performs well on straight sections and completes laps reliably. However, the centerline-based CTH behavior can become too sharp at turns. This suggests a weakness on higher-curvature tracks and creates a disadvantage for real-car deployment, where sharp steering at speed can produce unstable or unsafe behavior.

The key issue is that the baseline reward strongly encourages staying close to the centerline and aligned with the local track heading, but it does not explicitly teach curvature-aware racing behavior such as slowing before turns and accelerating after turns.

## Curriculum Reward Shaping

To address this, we added a new reward type: `curriculum`.

The curriculum reward starts with CTH as the dominant safety/stability signal, then gradually shifts toward progress and speed while keeping CTH as a soft constraint.

The reward is:

```text
R = cth_mix      * R_cth
  + progress_mix * progress_weight * forward_progress_m_per_s
  + speed_mix    * speed_weight * speed
  - turn_mix     * turn_weight * abs(steering) * speed
```

The schedule is:

```text
0% - 10% training:
  CTH dominates so the agent learns stable lane-following behavior.

10% - 80% training:
  CTH gradually decreases.
  Progress, speed, and curvature-aware turn shaping gradually increase.

80% - 100% training:
  Final reward mix is active.
```

Final target mix:

```text
cth_mix       = 0.25
progress_mix  = 1.0
speed_mix     = 1.0
turn_mix      = 1.0
```

This makes CTH a soft constraint instead of the full objective, adds progress dominance, and introduces curvature-aware behavior through:

```text
R_turn = -abs(steering) * speed
```

This discourages high speed while steering hard and encourages more natural racing behavior: slower before turns, smoother through turns, and faster after turns.

## Curriculum Result

Run:

```text
bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43
```

Evaluation results over 20 deterministic episodes:

```text
return              : 1229.565 +/- 5.130
steps               : 2043.850 +/- 9.035
avg_speed           : 2.966 +/- 0.000
max_speed           : 3.000 +/- 0.000
progress            : 99.57% +/- 0.42%  (max: 100.00%)
collision           : 0.0% crash rate (0/20 episodes)
lap_time            : 20.43s  (best: 20.24s)
steer_smoothness    : 0.004 +/- 0.000
```

Compared with the baseline, the curriculum reward keeps the same 0% crash rate while improving average lap time from **20.85 s** to **20.43 s** and improving steering smoothness from roughly **0.010** to **0.004**.

## ROS/Foxglove Observation

When testing the curriculum policy in ROS/Foxglove, the model could make turns at the original `3.0 m/s` speed cap, but the resulting line was too close to the outer wall after corner exit. This is consistent with the eval policy optimizing lap completion and speed with a low clearance margin: the behavior is valid in the training/eval simulator, but sensitive to vehicle footprint, LiDAR preprocessing, and ROS simulation mismatch.

To improve real-car and ROS robustness, we increased the clearance penalty for the next retraining run:

```yaml
wall_proximity_penalty: 2.0
wall_proximity_mean_penalty: 1.0
wall_proximity_threshold: 0.7
```

The new `wall_proximity_mean_penalty` punishes sustained wall-hugging across many LiDAR beams, not only the single closest ray. The goal is to learn a wider safety buffer while preserving the curriculum reward's speed and progress benefits.

Retraining command:

```bash
python scripts/train.py \
  --config configs/sim2real_e2e.yaml \
  --bc-pretrain demos/levine_blocked_pp_3ms.npz \
  --algo ppo \
  --reward-type curriculum \
  --name bc_ppo_curriculum_clearance_levine_safe \
  --total-steps 5000000 \
  --wandb
```

## Training Command

The existing behavioral cloning demonstrations can be reused:

```bash
python scripts/train.py \
  --config configs/sim2real_e2e.yaml \
  --bc-pretrain demos/levine_blocked_pp_3ms.npz \
  --algo ppo \
  --reward-type curriculum \
  --name bc_ppo_curriculum_levine_safe \
  --total-steps 5000000 \
  --wandb
```

## Evaluation Command

```bash
python scripts/evaluate.py \
  --run runs/bc_ppo_curriculum_levine_safe_* \
  --episodes 20
```

## Video Command

```bash
python scripts/record_video.py \
  --run 'runs/bc_ppo_curriculum_levine_safe_*' \
  --output videos/bc_ppo_curriculum_levine_safe.mp4
```

## ROS2 and Foxglove

Foxglove is used for visualization and topic inspection. The policy itself runs as a ROS2 node that subscribes to:

```text
/scan
/odom
```

and publishes:

```text
/drive
```

This repository already includes the inference node at:

```text
f1tenth_rl/ros2/inference_node.py
```

We added ROS2/Foxglove deployment helpers under:

```text
ros2_deploy/
scripts/run_ros2_foxglove.sh
```

### Quick Run

On a machine with ROS2 Humble and the F1TENTH stack sourced:

```bash
source /opt/ros/humble/setup.bash

bash scripts/run_ros2_foxglove.sh \
  runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43 \
  2.0
```

The second argument is the deployment speed cap in m/s. Start low on the real car, for example `1.0` or `1.5`, before trying `2.0`.

If `foxglove_bridge` is installed, the script also starts a Foxglove websocket bridge:

```text
ws://localhost:8765
```

For a remote car, replace `localhost` with the car computer IP address in Foxglove.

### Install Foxglove Bridge

```bash
sudo apt update
sudo apt install ros-humble-foxglove-bridge
```

### Proper ROS2 Package Launch

If you prefer a normal colcon workflow, copy or symlink `ros2_deploy` into a ROS2 workspace:

```bash
mkdir -p ~/f1tenth_team7_ws/src
ln -s /path/to/ESE615_final_team7/ros2_deploy ~/f1tenth_team7_ws/src/f1tenth_rl_deploy
cd ~/f1tenth_team7_ws
colcon build --symlink-install
source install/setup.bash
```

Make sure this Python project is installed in the same environment:

```bash
cd /path/to/ESE615_final_team7
python3 -m pip install -e .
```

Then launch:

```bash
ros2 launch f1tenth_rl_deploy rl_foxglove.launch.py \
  model_path:=/path/to/ESE615_final_team7/runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43/final_model.zip \
  config_path:=/path/to/ESE615_final_team7/runs/bc_ppo_curriculum_levine_safe_2026-04-27_11-28-43/config.yaml \
  max_speed:=2.0 \
  scan_topic:=/scan \
  odom_topic:=/odom \
  drive_topic:=/drive
```

### What to Check in Foxglove

Before enabling motion, confirm these topics are alive:

```bash
ros2 topic list
ros2 topic echo /scan --once
ros2 topic echo /odom --once
ros2 topic echo /drive --once
```

In Foxglove, watch:

```text
/scan       lidar input
/odom       velocity and yaw-rate input
/drive      policy output command
```

Safety checks:

- Start with `max_speed:=1.0` or `1.5` on the real car.
- Keep an emergency stop ready.
- Confirm `/drive` steering and speed look reasonable before putting the car on the ground.
- If steering is mirrored, rerun with `flip_scan:=true` or check the LiDAR frame orientation.
