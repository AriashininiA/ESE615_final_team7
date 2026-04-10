# F1TENTH RL Training Framework

Train autonomous racing policies using Reinforcement Learning in the [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) simulator, then deploy them on real F1TENTH cars with no real-world training data required.

This framework was built for the [ESE 615](https://f1tenth.org/) course at the University of Pennsylvania. It's designed to be easy to pick up even if you've never worked with RL before. Every parameter lives in a YAML config file, every training mode has a self-contained example, and the entire sim-to-real pipeline (train → export → deploy) works out of the box.

---
## Table of Contents

- [What Can This Framework Do?](#what-can-this-framework-do)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How to Train](#how-to-train)
- [Student Examples](#student-examples)
- [Configuration](#configuration)
- [Deploying on a Real F1TENTH Car](#deploying-on-a-real-f1tenth-car)
- [Adding Your Own Maps](#adding-your-own-maps)
- [Customizing the Framework](#customizing-the-framework)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)
- [Hardware](#hardware)
- [F1TENTH Gym Internals Reference](#f1tenth-gym-internals-reference)
- [References](#references)

---

## What Can This Framework Do?

- **Train RL policies** that drive an autonomous car around a racetrack
- **Multiple training approaches**: pure RL, imitation learning, behavioral cloning, or any combination
- **Multi-agent racing**: train your agent to race against opponents (pure pursuit bots or other RL agents)
- **Self-play**: your agent races against a copy of itself that keeps improving
- **Domain randomization**: make policies robust to real-world conditions (friction, sensor noise, delays)
- **Sim-to-real transfer**: export trained policies and deploy on physical F1TENTH cars with zero real-world training
- **Multiple RL algorithms**: PPO, SAC, TD3, and a from-scratch PPO implementation for learning
- **WandB integration**: track all your experiments automatically

---

## Project Structure

```
f1tenth_rl/
│
├── configs/                          # All training settings live here
│   ├── default.yaml                  # ← Start here. Every parameter is documented.
│   ├── sim2real_e2e.yaml             # End-to-end policy for real car (lidar only)
│   ├── sim2real_fast.yaml            # Fast racing policy (up to 6 m/s)
│   ├── sim2real_localized.yaml       # Localized policy (lidar + particle filter)
│   ├── multi_agent_vs_pp.yaml        # RL agent vs pure pursuit opponent
│   ├── multi_agent_self_play.yaml    # RL agent vs itself (self-play)
│   └── levine.yaml                   # Settings for Levine Hall at UPenn
│
├── examples/                         # Self-contained teaching examples
│   ├── 1_simple_ppo.py               # Bare minimum: train PPO in ~200 lines
│   ├── 2_race_against_opponent.py    # Two cars: RL vs pure pursuit
│   ├── 3_imitation_learning.py       # Learn from an expert, then improve with RL
│   ├── 4_race_fast.py                # Train for maximum speed
│   ├── 5_ppo_from_scratch.py         # PPO implemented from scratch (educational)
│   ├── 6_custom_reward.py            # Custom reward functions (TAL, Frenet-frame)
│   ├── 7_custom_observations.py      # Custom observations (opponent, dynamics, track)
│   └── README.md                     # Guide to all examples
│
├── f1tenth_rl/                       # The main Python package
│   ├── envs/                         # Environment code
│   │   ├── wrapper.py                # Connects F1TENTH Gym to RL algorithms
│   │   ├── observations.py           # What the agent sees (lidar, velocity, etc.)
│   │   ├── rewards.py                # What the agent optimizes (progress, speed, etc.)
│   │   ├── domain_randomization.py   # Randomize physics for robust sim-to-real
│   │   └── self_play.py              # Self-play wrapper for RL vs RL
│   │
│   ├── agents/                       # RL algorithm implementations
│   │   ├── sb3_trainer.py            # Stable Baselines 3 trainer (PPO, SAC, TD3)
│   │   ├── custom_ppo.py             # PPO implemented from scratch (no SB3)
│   │   ├── networks.py               # Neural network architectures (MLP, 1D CNN)
│   │   └── imitation.py              # Behavioral cloning (learn from demonstrations)
│   │
│   ├── experts/                      # Non-RL controllers
│   │   ├── pure_pursuit.py           # Classic waypoint-following controller
│   │   ├── raceline.py               # Compute racing lines from track maps
│   │   └── demo_collector.py         # Record expert demonstrations for imitation learning
│   │
│   ├── utils/                        # Helper code
│   │   ├── callbacks.py              # Training callbacks (metrics, WandB, DR scheduling)
│   │   └── waypoints.py              # Waypoint utilities
│   │
│   └── ros2/                         # Real car deployment
│       └── inference_node.py         # ROS2 node that runs the policy on the car
│
├── scripts/                          # Command-line tools
│   ├── train.py                      # ← Main training script
│   ├── evaluate.py                   # Test trained policies and see metrics
│   ├── export_model.py               # Convert to ONNX for fast inference on the car
│   ├── collect_demos.py              # Record expert driving for imitation learning
│   ├── generate_centerline.py        # Create waypoints from a SLAM map
│   ├── prepare_deploy.py             # Bundle everything for the car
│   ├── test_pure_pursuit.py          # Verify a map works before training
│   └── pose_relay.py                 # Fake particle filter for gym_ros testing
│
├── maps/                             # Track maps
│   ├── levine_slam/                  # Levine Hall (SLAM-generated from real building)
│   └── levine_blocked/               # Levine Hall (hand-drawn version)
│
├── setup.py                          # pip install -e .
├── requirements.txt                  # Python dependencies
└── README.md                         # You are here
```

---

## Getting Started

### Step 1: Install

```bash
# Clone this repository
git clone git@github.com:JeffersonKoumbaMoussadjiLu/f1tenth_rl_humble.git f1tenth_rl
cd f1tenth_rl

# Install F1TENTH Gym (you need the dev-humble branch)
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Install framework dependencies and the framework itself
cd ..
pip install -r requirements.txt
pip install -e .
```

### Step 2: Verify everything works

```bash
python3 -c "import f1tenth_gym; print('F1TENTH Gym OK')"
python3 -c "import stable_baselines3; print('Stable Baselines OK')"
python3 -c "import torch; print(f'PyTorch OK, GPU available: {torch.cuda.is_available()}')"

# Pre-cache the default track map (avoids a race condition with parallel envs)
python3 -c "
from f1tenth_gym.envs.track import Track
Track.from_track_name('Spielberg')
print('Spielberg map cached')
"
```

### Step 3: Train your first policy

```bash
# This trains a PPO agent on the Spielberg track for 500k steps (~5 minutes)
python3 scripts/train.py --name my_first_policy --total-steps 500000
```

### Step 4: See how it drives

```bash
# Opens a window showing the car driving
python3 scripts/evaluate.py --run runs/my_first_policy_* --episodes 5 --render
```

---

## How to Train

### Option A: Pure RL (train from scratch)

The agent starts with completely random actions and learns entirely from trial and error. This is the simplest approach but takes the most training time.

```bash
# Spielberg track (built-in, no map files needed)
python3 scripts/train.py --name ppo_spielberg --total-steps 1000000

# Levine Hall SLAM map
python3 scripts/train.py \
    --config configs/sim2real_e2e.yaml \
    --map maps/levine_slam/levine_slam \
    --name ppo_levine \
    --num-envs 12 --device cuda \
    --total-steps 10000000
```

### Option B: Imitation Learning (learn from an expert first)

A pure pursuit controller drives the car and we record what it does. Then a neural network learns to copy those actions. This gives the RL agent a "warm start". It already knows the basics of driving before RL training begins.

```bash
# Step 1: Expert drives 100 laps, we record everything
python3 scripts/collect_demos.py --episodes 100 --output demos/expert_demos.npz

# Step 2a: Train behavioral cloning only (supervised learning, no RL)
python3 scripts/train.py --bc-pretrain demos/expert_demos.npz --bc-only

# Step 2b: Or use BC as a starting point for PPO (recommended)
python3 scripts/train.py --bc-pretrain demos/expert_demos.npz \
    --name bc_then_ppo --total-steps 1000000
```

### Option C: Multi-Agent Racing

Train your agent to race against an opponent. The opponent can be a pure pursuit bot or another RL agent.

```bash
# Race against a pure pursuit opponent
python3 scripts/train.py \
    --config configs/multi_agent_vs_pp.yaml \
    --name race_vs_pp --num-envs 8 --device cuda

# Self-play: race against a copy of yourself
python3 scripts/train.py \
    --config configs/multi_agent_self_play.yaml \
    --name self_play --num-envs 8 --device cuda
```

### Option D: Different RL Algorithms

```bash
python3 scripts/train.py --algo ppo --name test_ppo            # PPO (default, most stable)
python3 scripts/train.py --algo sac --name test_sac            # SAC (off-policy, sample efficient)
python3 scripts/train.py --algo td3 --name test_td3            # TD3 (off-policy, deterministic)
python3 scripts/train.py --algo custom_ppo --name test_custom  # From-scratch PPO (educational)
```

---

## Student Examples

If you're new to RL or want to understand how to write your training script from scratch for the F1tenth gym, start with these. Each example is a single self-contained Python file. You can read it top to bottom and understand everything without looking at the rest of the framework.

| # | File | What It Teaches | Time |
|---|------|-----------------|------|
| 1 | `1_simple_ppo.py` | How to wrap an environment for RL, what observations/actions/rewards are, how to train PPO | ~5 min |
| 2 | `2_race_against_opponent.py` | Multi-agent environments, pure pursuit implementation, how RL handles dynamic obstacles | ~10 min |
| 3 | `3_imitation_learning.py` | Expert demonstration collection, behavioral cloning (supervised learning), BC→RL fine-tuning | ~10 min |
| 4 | `4_race_fast.py` | Reward shaping for speed, steering smoothness penalty, domain randomization, parallel training | ~10 min |
| 5 | `5_ppo_from_scratch.py` | Every component of PPO: actor-critic networks, rollout collection, GAE (advantage estimation), clipped objective, entropy | ~10 min |
| 6 | `6_custom_reward.py` | Writing custom reward functions: trajectory-aided learning, Frenet-frame rewards, expert matching | ~10 min |
| 7 | `7_custom_observations.py` | Adding custom observations: opponent distance/bearing, vehicle dynamics state, track curvature | ~10 min |

Run any example:
```bash
python3 examples/1_simple_ppo.py                    # Train
python3 examples/1_simple_ppo.py --eval --render     # Watch it drive
```

Example 5: You can watch the car learn in real-time:
```bash
python3 examples/5_ppo_from_scratch.py --render      # Watch it go from crashing to driving laps
```

---

## Configuration

Every training parameter lives in a YAML config file inside `configs/`. You never need to touch the Python code to change how training works, just edit the config or pass command-line flags.

### Available Configs

| Config | Purpose | Max Speed | Domain Randomization |
|--------|---------|-----------|---------------------|
| `default.yaml` | General training on any track | 8 m/s | Off |
| `sim2real_e2e.yaml` | Safe policy for real car | 3 m/s | Curriculum |
| `sim2real_fast.yaml` | Fast racing for real car | 6 m/s | Curriculum (aggressive) |
| `sim2real_localized.yaml` | Localized policy (needs particle filter) | 3 m/s | Curriculum |
| `multi_agent_vs_pp.yaml` | RL vs pure pursuit opponent | 4 m/s | Curriculum |
| `multi_agent_self_play.yaml` | RL vs itself | 5 m/s | Curriculum |
| `levine.yaml` | Levine Hall specific settings | 3 m/s | Off |

### How to use configs

```bash
# Use a specific config
python3 scripts/train.py --config configs/sim2real_e2e.yaml

# Override specific values from the command line
python3 scripts/train.py --config configs/sim2real_e2e.yaml \
    --map maps/levine_slam/levine_slam \
    --total-steps 5000000 \
    --num-envs 12 \
    --device cuda

# Create your own config
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml, then:
python3 scripts/train.py --config configs/my_experiment.yaml
```

### Key parameters explained

| What | Config Key | What It Does |
|------|-----------|-------------|
| **Observation type** | `observation.type` | What the agent sees. `lidar_state` = lidar + velocity (recommended). `lidar_waypoint` = adds waypoint features (needs localization). |
| **Lidar beams** | `observation.lidar_beams` | How many lidar beams after downsampling. 108 works well. |
| **Max speed** | `action.max_speed` | The fastest the car can go. Start with 3.0, increase after it works. |
| **Reward type** | `reward.type` | What the agent optimizes. `progress` = go forward. `cth` = heading + cross-track (smoother). |
| **Steering penalty** | `reward.steering_change_penalty` | Penalizes jerky steering. Higher = smoother but slower. 0.5 for normal, 1.5 for fast racing. |
| **Domain randomization** | `domain_randomization.mode` | `off` = no randomization. `curriculum` = gradually increase (recommended for sim-to-real). |
| **Action delay** | `domain_randomization.action_delay.max_steps` | Simulates real-world latency. 3 for slow, 5 for fast racing. Critical for sim-to-real. |
| **Network size** | `network.mlp.hidden_sizes` | Neural network width. `[128, 128]` for normal, `[256, 256]` for fast racing. |

---

## Deploying on a Real F1TENTH Car

The entire deployment pipeline is: train in sim → export to ONNX → copy 3 files to car → run.

### Step 1: Train a sim-to-real policy

```bash
python3 scripts/train.py \
    --config configs/sim2real_e2e.yaml \
    --map maps/levine_slam/levine_slam \
    --name my_policy \
    --num-envs 12 --device cuda \
    --total-steps 15000000
```

### Step 2: Export and bundle for the car

```bash
python3 scripts/export_model.py --run runs/my_policy_* --benchmark
python3 scripts/prepare_deploy.py --run runs/my_policy_*
```

### Step 3: Copy to the car

```bash
scp -r deploy/my_policy_* jetson@<CAR_IP>:~/f1tenth_rl/
```

### Step 4: Install dependencies on the car

The inference node needs a few Python packages on the Jetson. Most are already installed if you have the F1TENTH ROS2 stack set up.

```bash
# These are required regardless of ONNX or PyTorch mode:
pip install numpy pyyaml

# Option A: ONNX Runtime (recommended: fast, no PyTorch needed)
pip install onnxruntime
# Note: on JetPack 6.x you may need the Jetson-specific wheel:
#   pip install onnxruntime-gpu   (from NVIDIA's pip index)

# Option B: PyTorch + SB3 fallback (if ONNX won't install)
pip install torch stable-baselines3
# On Jetson, install PyTorch from NVIDIA's wheel:
#   See https://forums.developer.nvidia.com/t/pytorch-for-jetson/
```

The ROS2 packages (`rclpy`, `sensor_msgs`, `nav_msgs`, `ackermann_msgs`, `geometry_msgs`) come with the standard F1TENTH ROS2 workspace. You don't need to install them separately.

### Step 5: Run on the car

The end-to-end model only needs 3 files on the Jetson. No normalization files, no SB3, no particle filter:

```bash
python3 inference_node.py --ros-args \
    -p model_path:=final_model.onnx \
    -p config_path:=config.yaml \
    -p use_onnx:=true \
    -p max_speed:=2.0 \
    -p scan_topic:=/scan \
    -p odom_topic:=/odom \
    -p drive_topic:=/drive
```

**Start slow and increase speed gradually:**
```
-p max_speed:=2.0    ← start here
-p max_speed:=3.0    ← if 2.0 is smooth
-p max_speed:=4.0    ← if 3.0 works
-p max_speed:=5.0    ← if 4.0 is stable
```

If the car steers the wrong way, the scan direction might be flipped:
```bash
-p flip_scan:=true
```

### Testing in f1tenth_gym_ros first (recommended)

Before putting the policy on the real car, test it in the gym_ros simulator. Same physics engine, but with ROS2 topics in the middle which catches most integration issues.

```bash
# Terminal 1: Launch the simulator
ros2 launch f1tenth_gym_ros gym_bridge_launch.py

# Terminal 2 (only for localized models): Relay ground-truth pose as fake particle filter
python3 scripts/pose_relay.py

# Terminal 3: Run the policy
python3 f1tenth_rl/ros2/inference_node.py --ros-args \
    -p model_path:=final_model.onnx \
    -p config_path:=config.yaml \
    -p use_onnx:=true \
    -p max_speed:=2.0 \
    -p scan_topic:=/scan \
    -p odom_topic:=/ego_racecar/odom \
    -p drive_topic:=/drive
```

---

## Adding Your Own Maps

1. **SLAM your environment** and save the map as `my_map.pgm` + `my_map.png`

2. **Create the map YAML** (important: do NOT include `mode: trinary`):
   ```yaml
   image: my_map.png
   resolution: 0.05
   origin: [-21.4, -4.87, 0]
   negate: 0
   occupied_thresh: 0.65
   free_thresh: 0.25
   ```

3. **Generate a centerline** (the framework does this automatically, but you can also run it manually):
   ```bash
   python3 scripts/generate_centerline.py --map maps/my_map/my_map --visualize
   ```

4. **Test with pure pursuit** to make sure the map and centerline work:
   ```bash
   python3 scripts/test_pure_pursuit.py --map maps/my_map/my_map --speed 3.0
   ```

5. **Train:**
   ```bash
   python3 scripts/train.py --config configs/sim2real_e2e.yaml \
       --map maps/my_map/my_map --name my_map_policy
   ```

---

## Customizing the Framework

The config files let you change parameters (how many lidar beams, what speed, which reward type). But sometimes you need to go deeper and add a completely new reward function, change what the agent observes, or wire up a new RL algorithm. Here's a guide for each.

### Adding a New Reward Function

All reward functions live in `f1tenth_rl/envs/rewards.py`. Each one inherits from `RewardFunction` and implements two methods:

```python
# In f1tenth_rl/envs/rewards.py - add your new class:

class GapFollowReward(RewardFunction):
    """
    Custom reward that encourages driving toward the largest gap.
    """

    def __init__(self, config, waypoints):
        super().__init__(config)
        self.gap_weight = config.get("gap_weight", 0.5)

    def _reset_impl(self, obs_dict, ego_idx):
        # Called at the start of each episode.
        # Use this to initialize any per-episode state.
        pass

    def _compute_impl(self, obs_dict, ego_idx, action):
        # Called every step. Return a float reward.
        # obs_dict contains: scans, poses_x, poses_y, poses_theta,
        #                    linear_vels_x, ang_vels_z, etc.
        scan = obs_dict["scans"][ego_idx]
        speed = obs_dict["linear_vels_x"][ego_idx]

        # Find the largest gap in the lidar scan
        largest_gap_idx = np.argmax(scan)
        center_idx = len(scan) // 2

        # Reward steering toward the gap
        gap_alignment = 1.0 - abs(largest_gap_idx - center_idx) / center_idx
        reward = self.gap_weight * gap_alignment + 0.1 * speed

        return reward
```

Then register it in `f1tenth_rl/envs/wrapper.py` inside `_make_reward()`:

```python
# In wrapper.py, find the _make_reward method and add your type:

def _make_reward(self, rew_cfg):
    reward_type = rew_cfg.get("type", "progress")
    ...
    elif reward_type == "gap_follow":
        return GapFollowReward(rew_cfg, wp)
```

Now use it in your config: `reward.type: "gap_follow"`

The base class automatically handles collision penalties, lap bonuses, steering smoothness penalties, and wall proximity penalties. You only need to implement the core reward logic in `_compute_impl()`.

### Modifying the Observation Space

Observations are built in `f1tenth_rl/envs/observations.py`. The class `ObservationBuilder` has two key methods:

- `_compute_obs_dim()` → returns the total size of the observation vector
- `build()` → constructs the actual observation from raw simulator data

To add a new feature (for example, distance to the nearest opponent):

```python
# In observations.py:

# Step 1: Add a config flag
self.include_opponent_dist = config.get("include_opponent_dist", False)

# Step 2: Update _compute_obs_dim() to account for the new dimension
if self.include_opponent_dist:
    dim += 1

# Step 3: In build(), add the feature to the components list
if self.include_opponent_dist:
    # Compute distance to nearest opponent (from pose data)
    ego_x = float(obs_dict["poses_x"][ego_idx])
    ego_y = float(obs_dict["poses_y"][ego_idx])
    min_opp_dist = float('inf')
    for i in range(len(obs_dict["poses_x"])):
        if i != ego_idx:
            dx = float(obs_dict["poses_x"][i]) - ego_x
            dy = float(obs_dict["poses_y"][i]) - ego_y
            min_opp_dist = min(min_opp_dist, np.sqrt(dx**2 + dy**2))
    # Normalize to [0, 1] range (assuming max 10m detection)
    components.append(np.array([min(min_opp_dist / 10.0, 1.0)], dtype=np.float32))
```

**Important:** Always normalize new features to a reasonable range (roughly 0 to 1 or -1 to 1). Neural networks learn much faster when inputs are on similar scales.

Then in your config:
```yaml
observation:
  include_opponent_dist: true
```

### Changing the Action Space

The action space is defined in `f1tenth_rl/envs/wrapper.py`. Look for:

- `self.action_space = ...` in `__init__()` defines the shape and range
- `_scale_action()` converts the neural network's [-1, 1] output to physical steering and speed

For example, to add a new "speed-only" action type where steering comes from pure pursuit and the agent only controls speed:

```python
# In wrapper.py __init__, after the existing action types:
elif self.action_type == "speed_only":
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

# In _scale_action():
elif self.action_type == "speed_only":
    # Steering comes from pure pursuit, agent controls speed
    if self.opponent_controller is not None and self.prev_obs_dict is not None:
        steer, _ = self.opponent_controller.get_action(self.prev_obs_dict)
    else:
        steer = 0.0
    speed = (float(action[0]) + 1.0) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
```

### Adding a New RL Algorithm

The easiest approach is to follow the pattern in `f1tenth_rl/agents/custom_ppo.py`:

1. Create a new file (e.g., `f1tenth_rl/agents/my_algorithm.py`)
2. Your class needs:
   - `__init__(self, config)` read hyperparameters from the config dict
   - `train()` the main training loop, using `make_env()` from `wrapper.py`
   - `save(path)` and `load(path)` for checkpointing

3. Register it in `scripts/train.py`:

```python
# In train.py, find the algorithm dispatch section and add:
elif algo_type == "my_algo":
    from f1tenth_rl.agents.my_algorithm import MyAlgorithm
    agent = MyAlgorithm(config)
    agent.train()
```

4. Add the CLI option in `train.py`'s argument parser:

```python
parser.add_argument("--algo", choices=["ppo", "sac", "td3", "custom_ppo", "my_algo"])
```

5. Add default hyperparameters in your config:

```yaml
algorithm:
  type: "my_algo"
  my_algo:
    learning_rate: 0.001
    # ... your hyperparameters
```

### Quick Reference: Which File to Edit

| I want to... | Edit this file |
|--------------|----------------|
| Change what the agent sees | `f1tenth_rl/envs/observations.py` |
| Change what the agent optimizes | `f1tenth_rl/envs/rewards.py` |
| Change what the agent controls | `f1tenth_rl/envs/wrapper.py` (action space) |
| Change the neural network shape | `f1tenth_rl/agents/networks.py` |
| Change how training works | `f1tenth_rl/agents/sb3_trainer.py` or `custom_ppo.py` |
| Change domain randomization | `f1tenth_rl/envs/domain_randomization.py` |
| Change how the expert drives | `f1tenth_rl/experts/pure_pursuit.py` |
| Change how demos are collected | `f1tenth_rl/experts/demo_collector.py` |
| Change what runs on the real car | `f1tenth_rl/ros2/inference_node.py` |

### Tips for Common Advanced Use Cases

**Using the Frenet Frame in reward functions.** The Track object gives you the car's position relative to the centerline. This is incredibly useful for reward shaping:

```python
# Inside your reward function's _compute_impl():
track = self.track  # Set this in __init__ from env.unwrapped.track
x = float(obs_dict["poses_x"][ego_idx])
y = float(obs_dict["poses_y"][ego_idx])
theta = float(obs_dict["poses_theta"][ego_idx])

s, ey, ephi = track.cartesian_to_frenet(x, y, theta)
# s    = arc-length progress along track (meters)
# ey   = lateral deviation from centerline (meters, + = left)
# ephi = heading error vs track direction (radians)
```

See `examples/6_custom_reward.py` for a complete working example.

**Enabling reverse driving (for parking).** Set `action.min_speed` to a negative value in your config:

```yaml
action:
  min_speed: -2.0   # Allow reversing at up to 2 m/s
  max_speed: 3.0
```

Note: the default reward functions assume forward progress. For parking, you'll need a custom reward based on distance to a goal pose, and a custom observation that includes the relative position to the target. See `examples/6_custom_reward.py` and `examples/7_custom_observations.py` for how to add custom rewards and observations.

**Using a custom neural network encoder (e.g., for JEPA, VAE).** SB3 supports custom feature extractors via `policy_kwargs`. To use a pre-trained frozen encoder:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FrozenEncoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, encoder, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze weights

    def forward(self, observations):
        with torch.no_grad():
            return self.encoder(observations)

# Then pass it to PPO:
policy_kwargs = {
    "features_extractor_class": FrozenEncoderExtractor,
    "features_extractor_kwargs": {"encoder": my_pretrained_encoder, "features_dim": 64},
}
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

**Collecting diverse trajectory data (not just expert demos).** The `collect_demos.py` script uses pure pursuit by default. For pre-training a world model or VAE, you want diverse trajectories from different controllers and speeds:

```python
# Collect data at multiple speeds for diverse training data
for speed in [2.0, 3.0, 4.0, 5.0]:
    os.system(f"python scripts/collect_demos.py "
              f"--speed {speed} --episodes 25 "
              f"--output demos/demos_speed{speed}.npz")

# Or add random perturbations to the expert for even more diversity:
# In your collection loop, occasionally add noise to the expert's action
action = expert.get_normalized_action(obs, ego_idx=0)
action += np.random.normal(0, 0.1, size=2)  # Small random perturbation
action = np.clip(action, -1, 1)
```

**Adding a custom opponent type.** The framework supports `pure_pursuit` and `self_play` opponents. To add a new type (e.g., Follow-the-Gap, Wall-Following):

```python
# In f1tenth_rl/envs/wrapper.py, find where opponents are created:
if ma_cfg.get("opponent") == "pure_pursuit":
    self.opponent_controller = PurePursuitController(...)
elif ma_cfg.get("opponent") == "follow_the_gap":
    from f1tenth_rl.experts.follow_the_gap import FollowTheGapController
    self.opponent_controller = FollowTheGapController(config)

# Your controller needs a get_action(obs_dict, ego_idx) method that
# returns (steering_angle, speed) in physical units.
```

---

## Monitoring Training

```bash
# WandB (enabled by default - creates an account automatically if needed)
# View at: https://wandb.ai/<your-username>/f1tenth_rl

# Disable WandB
python3 scripts/train.py --no-wandb

# TensorBoard (always available)
tensorboard --logdir runs/
```

### Understanding the Training Output

During training, the framework prints metrics every few thousand steps. Here's what they mean and what to look for:

```
| rollout/ep_rew_mean   | 1200     |   ← Average total reward per episode (HIGHER = BETTER)
| rollout/ep_len_mean   | 2150     |   ← Average episode length in steps (LONGER = car survived longer)
| racing/avg_progress   | 1.0      |   ← Fraction of lap completed (1.0 = full lap!)
| racing/avg_speed      | 2.96     |   ← Average speed in m/s
| racing/collision_rate | 0.0      |   ← Fraction of episodes ending in crash (0.0 = perfect)
| racing/avg_lap_time   | 21.5     |   ← Seconds per lap (LOWER = FASTER)
| eval/mean_reward      | 1240     |   ← Reward during clean evaluation (no exploration noise)
| dr/strength           | 0.75     |   ← Domain randomization strength (0→1 for curriculum)
```

**How to tell if training is working:**
- `ep_rew_mean` should generally trend upward over time
- `collision_rate` should decrease (start near 1.0, drop toward 0.0)
- `avg_progress` should increase (start near 0.05, reach 1.0 when completing laps)
- If reward is flat for 500k+ steps, something is wrong (bad reward function, learning rate too high, etc.)

**When is training "done"?**
- For a safe policy: `avg_progress ≥ 1.0` and `collision_rate ≤ 0.05` and reward is stable
- For a fast policy: all the above, plus `avg_speed` is close to `max_speed` and `avg_lap_time` has plateaued
- More steps rarely hurts - if you have time, let it keep going

### What's Inside a Run Directory

After training completes, everything is saved in `runs/<your_run_name>/`:

```
runs/my_run_2026-04-09_18-30-00/
├── config.yaml                  # Exact config used (for reproducibility)
├── final_model.zip              # ← USE THIS for evaluation and deployment
├── final_vecnormalize.pkl       # Reward normalization stats
├── best_model/
│   └── best_model.zip           # Best model found during training (by eval reward)
├── checkpoints/                 # Periodic snapshots
│   ├── model_50000_steps.zip
│   └── model_100000_steps.zip
└── eval/                        # Evaluation logs
```

**Which model file should I use?**
- `final_model.zip` - the model at the end of training. Use this by default.
- `best_model/best_model.zip` - the model that scored highest during evaluation checkpoints. Use this if your training got worse toward the end (sometimes happens with domain randomization).

### Evaluating Trained Policies

The `evaluate.py` script has more capabilities than just `--run`:

```bash
# Basic evaluation with rendering
python3 scripts/evaluate.py --run runs/my_run_* --episodes 20 --render

# Evaluate the best model instead of the final model
python3 scripts/evaluate.py --run runs/my_run_* --use-best --episodes 20

# Evaluate a standalone BC model
python3 scripts/evaluate.py --bc-model runs/bc_pretrained/bc_model.pt \
    --config configs/sim2real_e2e.yaml --episodes 10

# Compare multiple runs side by side
python3 scripts/evaluate.py --run runs/run_a_* runs/run_b_* --episodes 20

# Export to ONNX during evaluation
python3 scripts/evaluate.py --run runs/my_run_* --export-onnx
```

### Parallel Environments (--num-envs)

RL training collects experience by running the environment step-by-step. With `--num-envs 1`, you collect one step at a time. With `--num-envs 12`, you collect 12 steps simultaneously: same wall-clock time, 12× more data.

```bash
# Slow (1 environment, ~200 steps/second)
python3 scripts/train.py --num-envs 1

# Fast (12 environments in parallel, ~2400 steps/second)
python3 scripts/train.py --num-envs 12 --device cuda
```

**How to choose:** More envs = faster training, but more RAM. On a GPU with 8+ GB VRAM, use 8-16. On CPU only, use 4-8. If you run out of memory, reduce `num_envs`.

### Which Config Should I Start With?

```
Are you training for the real car?
├── YES → Do you need to follow a specific raceline?
│   ├── YES → sim2real_localized.yaml (needs particle filter on car)
│   └── NO  → How fast do you want to go?
│       ├── Safe (≤3 m/s)  → sim2real_e2e.yaml ← START HERE
│       └── Fast (≤6 m/s)  → sim2real_fast.yaml
├── NO → Are you racing against opponents?
│   ├── YES → Against pure pursuit? → multi_agent_vs_pp.yaml
│   │         Against itself?       → multi_agent_self_play.yaml
│   └── NO  → Just experimenting?
│       ├── On Spielberg   → default.yaml (works out of the box)
│       └── On Levine Hall → levine.yaml
```

**If you're not sure, start with `default.yaml` and the Spielberg track.** It works out of the box with no map files. Once you're comfortable, switch to a sim2real config for your real map.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mode: trinary` error | Remove `mode: trinary` from your map's YAML file |
| `broken PNG file` or `EOFError: Compressed file ended` | Multiple parallel envs tried to extract the map simultaneously. Run `python3 -c "import f1tenth_gym; import gymnasium; gymnasium.make('f1tenth_gym:f1tenth-v0')"` once to pre-cache the track, then try again. |
| `ImportError: tqdm and rich` | Run `pip install stable-baselines3[extra]` or `pip install rich tqdm` |
| `wandb not installed` warning | Run `pip install wandb`. Or use `--no-wandb` to skip logging. |
| Training not converging | Try more steps (10M+), lower learning rate (`--lr 0.0001`), or BC pretraining |
| Policy crashes on real car | Reduce `max_speed`, check action delay in DR config, verify scan beam count |
| Localized model wobbles | Increase `waypoint_spacing` (1.5+) and `steering_change_penalty` (1.0+) |
| `No module named 'onnxscript'` during ONNX export | Run `pip install onnxscript onnx onnxruntime`. Required by PyTorch 2.11+. |
| ONNX not installed on Jetson | Use `--use_onnx:=false` with `final_model.zip` instead (needs SB3 + PyTorch) |
| Car steers into wrong wall | Try `-p flip_scan:=true` - the scan direction might be reversed |
| Pure pursuit too slow | Make sure config `max_speed` matches your `--speed` flag |

---

## Hardware

**Training PC:** Any machine with Python 3.8+ works. A GPU speeds up training significantly but is **not required**, CPU-only training works fine, just slower. We use an RTX 4070 (12GB) with 24 CPU cores. Training 15M steps takes ~2-6 hours on GPU, or ~6-18 hours on CPU depending on the config.

**F1TENTH Car:** Jetson Orin Nano + Hokuyo UST-10LX lidar (1080 beams) or SICK TIM571 (812 beams) + VESC motor controller. The inference node handles both lidar models automatically.

---

## F1TENTH Gym Internals Reference

This section documents what the F1TENTH Gym (dev-humble branch) exposes under the hood. You don't need any of this for basic training, the wrapper handles everything. But if you're an experienced student who wants to go deeper, build custom features, or debug unusual behavior, this is where to look.

### Vehicle Parameters

The simulated car's physics are controlled by `VehicleParameters`. The default values match a real F1TENTH car:

| Parameter | Default | What It Is |
|-----------|---------|-----------|
| `mu` | 1.0489 | Tire friction coefficient |
| `C_Sf` | 4.718 | Front cornering stiffness |
| `C_Sr` | 5.4562 | Rear cornering stiffness |
| `lf` | 0.15875 | Distance from CG to front axle (m) |
| `lr` | 0.17145 | Distance from CG to rear axle (m) |
| `h` | 0.074 | CG height (m) |
| `m` | 3.74 | Vehicle mass (kg) |
| `I` | 0.04712 | Yaw moment of inertia (kg⋅m²) |
| `s_min` / `s_max` | -0.4189 / 0.4189 | Steering angle limits (rad, ≈ ±24°) |
| `sv_min` / `sv_max` | -3.2 / 3.2 | Steering velocity limits (rad/s) |
| `v_switch` | 7.319 | Velocity where tire model switches (m/s) |
| `a_max` | 9.51 | Max acceleration (m/s²) |
| `v_min` / `v_max` | -5.0 / 20.0 | Velocity limits (m/s) |
| `width` / `length` | 0.31 / 0.58 | Vehicle dimensions (m) |

You can modify these in your wrapper by changing the `EnvConfig.params` field. For example, to simulate a heavier car:

```python
from f1tenth_gym.envs.dynamic_models import VehicleParameters, F1TENTH_VEHICLE_PARAMETERS
from dataclasses import replace

heavy_params = replace(F1TENTH_VEHICLE_PARAMETERS, m=5.0, mu=0.8)
```

### Dynamic Models

The gym supports two vehicle dynamics models:

| Model | Enum | Description |
|-------|------|-------------|
| **Kinematic Single Track (KS)** | `DynamicModel.KS` | Simple bicycle model. No tire slip. Fast and stable but less realistic at high speed. Good for basic training. |
| **Single Track (ST)** | `DynamicModel.ST` | Includes tire forces and slip angles. More realistic, especially for drifting and high-speed cornering. This is the default. |

Change it in the `EnvConfig`:
```python
from f1tenth_gym.envs.dynamic_models import DynamicModel
# Use kinematic model for faster simulation
config.simulation_config = config.simulation_config.with_updates(dynamics_model=DynamicModel.KS)
```

### State Vector

Each agent has a 7-element standard state vector (`std_state`):

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `x` | Position X (meters) |
| 1 | `y` | Position Y (meters) |
| 2 | `delta` | Steering angle (radians) |
| 3 | `velocity` | Speed magnitude (m/s) |
| 4 | `theta` | Heading angle / yaw (radians) |
| 5 | `omega` | Yaw rate / angular velocity (rad/s) |
| 6 | `beta` | Slip angle (radians) |

Access it in the observation dict: `obs["agent_0"]["std_state"]`

### Observation Fields

The gym can return many different observation fields per agent. Our wrapper extracts and normalizes these, but you can access them directly:

**Base fields** (always available):
| Field | Shape | Description |
|-------|-------|-------------|
| `scan` | `(num_beams,)` | LiDAR range measurements (meters) |
| `std_state` | `(7,)` | Standard state vector (see above) |
| `state` | `(state_dim,)` | Full dynamics state (model-dependent) |
| `collision` | `(1,)` | 1.0 if collided, 0.0 otherwise |
| `lap_time` | `(1,)` | Current lap time (seconds) |
| `lap_count` | `(1,)` | Number of completed laps |
| `sim_time` | `(1,)` | Total simulation time (seconds) |
| `frenet_pose` | `(3,)` | Frenet frame: `[s, ey, ephi]` (progress, lateral error, heading error) |

**Derived fields** (computed from std_state):
| Field | Description |
|-------|-------------|
| `pose_x`, `pose_y` | Position components |
| `pose_theta` | Heading angle |
| `linear_vel_x`, `linear_vel_y` | Velocity components (body frame) |
| `linear_vel_magnitude` | Speed scalar |
| `ang_vel_z` | Yaw rate |
| `delta` | Current steering angle |
| `beta` | Slip angle |

### Frenet Frame

The Frenet frame describes a car's position relative to a reference line (centerline or raceline):

| Coordinate | Name | Meaning |
|------------|------|---------|
| `s` | Arc length | How far along the track you are (meters from start) |
| `ey` | Lateral error | Distance from the centerline (positive = left, negative = right) |
| `ephi` | Heading error | Angle between car heading and track direction (radians) |

The Track object provides conversion functions:
```python
track = env.unwrapped.track

# Car position → Frenet coordinates
s, ey, ephi = track.cartesian_to_frenet(x, y, theta)

# Frenet coordinates → Car position
x, y, theta = track.frenet_to_cartesian(s, ey, ephi)

# Use the raceline instead of centerline
s, ey, ephi = track.cartesian_to_frenet(x, y, theta, use_raceline=True)
```

This is extremely useful for reward functions: `ey` tells you how far from the centerline the car is, and `ephi` tells you if it's pointing the right way.

### Track Object

The `Track` object is accessible at `env.unwrapped.track` and contains:

```python
track = env.unwrapped.track

# Centerline (auto-generated from map)
track.centerline.xs          # x coordinates (numpy array)
track.centerline.ys          # y coordinates
track.centerline.vxs         # velocity profile
track.centerline.yaws        # heading angles
track.centerline.ks          # curvature values
track.centerline.length      # total track length (meters)
track.centerline.spline      # cubic spline interpolation object

# Raceline (same as centerline unless a custom raceline is loaded)
track.raceline.xs            # optimized x coordinates
track.raceline.vxs           # optimized velocity profile

# Occupancy map (for collision checking)
track.occupancy_map          # 2D numpy array (0 = free, 1 = occupied)
track.spec.resolution        # meters per pixel
track.spec.origin            # [x, y, theta] of map origin
```

### Reset Strategies

How agents are placed at the start of each episode:

| Strategy | Description |
|----------|-------------|
| `RL_GRID_STATIC` | Grid formation near start line, fixed order (default) |
| `RL_RANDOM_STATIC` | Random position along track, fixed agent order |
| `RL_GRID_RANDOM` | Grid formation, shuffled agent order |
| `RL_RANDOM_RANDOM` | Random position, shuffled order |
| `MAP_RANDOM_STATIC` | Random position anywhere on free space |

`RL_RANDOM_*` strategies are useful for training because the agent sees different parts of the track at the start of each episode, which improves generalization.

### Action Types

The gym supports different ways to control the car:

**Longitudinal (speed/throttle):**
| Type | Description |
|------|-------------|
| `SPEED` | Target speed in m/s (internal PID controller handles acceleration). This is what our wrapper uses. |
| `ACCL` | Direct acceleration command in m/s². More realistic but harder to control. |

**Steering:**
| Type | Description |
|------|-------------|
| `STEERING_ANGLE` | Target steering angle in radians (internal PID). This is what our wrapper uses. |
| `STEERING_SPEED` | Direct steering velocity in rad/s. More realistic, models servo dynamics. |

You can also add steering delay: `control_config.steer_delay_steps = 3` simulates servo latency.

### LiDAR Configuration

All LiDAR parameters are configurable:

```python
from f1tenth_gym.envs.lidar import LiDARConfig

lidar = LiDARConfig(
    num_beams=1080,                      # Number of beams
    field_of_view=4.7124,                # FOV in radians (270°)
    range_min=0.0,                       # Min range (m)
    range_max=30.0,                      # Max range (m)
    noise_std=0.01,                      # Gaussian noise std (m)
    base_link_to_lidar_tf=(0.275, 0, 0), # Lidar mount offset from rear axle
)
```

### Collision Detection Modes

| Mode | Description |
|------|-------------|
| `LIDAR_SCAN` | Check if any lidar beam is shorter than the car body (default, fast) |
| `BOUNDING_BOX` | Full bounding box overlap check between agents (more accurate for multi-agent) |

### Integrators

| Type | Description |
|------|-------------|
| `EULER` | First-order Euler integration. Fast but less accurate. |
| `RK4` | Fourth-order Runge-Kutta. More accurate, especially at high speeds. Default. |

### Accessing the Simulator Directly

For advanced use, you can access the simulator internals:

```python
env = gym.make("f1tenth_gym:f1tenth-v0", config=env_config)

# After stepping:
obs, reward, terminated, truncated, info = env.step(action)

# Access internal state
sim = env.unwrapped.sim
sim.agent_states[0]          # Full state of agent 0
sim.agent_scans[0]           # Raw lidar scan of agent 0

# Access track
track = env.unwrapped.track
track.centerline             # Raceline object
track.raceline               # Raceline object (may differ if custom raceline loaded)

# Lap tracking
env.unwrapped.lap_counts[0]  # Laps completed by agent 0
env.unwrapped.lap_times[0]   # Current lap time
```

---

## References

- [F1TENTH Gym (dev-humble branch)](https://github.com/f1tenth/f1tenth_gym)
- [F1TENTH Course](https://f1tenth.org/)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [F1TENTH Racetracks](https://github.com/f1tenth/f1tenth_racetracks)
