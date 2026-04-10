# Student Examples

These are self-contained teaching examples. Each one is a single Python file that demonstrates a different RL concept applied to F1TENTH racing. You can read any example top-to-bottom without needing to understand the rest of the framework.

## How to Use

Every example follows the same pattern:
```bash
python3 examples/<example>.py                    # Train
python3 examples/<example>.py --eval --render    # Watch the trained policy drive
```

## The Examples

### Example 1: Simple PPO (`1_simple_ppo.py`)
**What you'll learn:** The absolute basics of RL for robotics.

This is where to start. It creates a minimal F1TENTH environment, trains a PPO agent, and evaluates it. Everything is in one file — environment wrapper, training loop, evaluation. The car starts by crashing into walls, and over 500k steps it gradually learns to drive laps.

```bash
python3 examples/1_simple_ppo.py --steps 500000
python3 examples/1_simple_ppo.py --eval --render
```

### Example 2: Race Against an Opponent (`2_race_against_opponent.py`)
**What you'll learn:** Multi-agent environments, how RL handles dynamic obstacles.

Two cars on the track: your RL agent vs a pure pursuit bot. The RL agent sees the opponent through its lidar as a moving obstacle and has to learn to drive fast without crashing into it. This teaches the agent defensive driving and overtaking.

```bash
python3 examples/2_race_against_opponent.py --steps 1000000
python3 examples/2_race_against_opponent.py --eval --render
```

### Example 3: Imitation Learning (`3_imitation_learning.py`)
**What you'll learn:** Learning from demonstrations, behavioral cloning, BC→RL fine-tuning.

Instead of learning from scratch, the agent first watches an expert (pure pursuit) drive and learns to imitate it. Then RL fine-tuning takes over to improve beyond the expert. This is the fastest way to get a competent policy.

```bash
python3 examples/3_imitation_learning.py --episodes 100 --steps 500000
python3 examples/3_imitation_learning.py --eval --render
```

### Example 4: Race Fast (`4_race_fast.py`)
**What you'll learn:** Reward shaping for speed, domain randomization, parallel training.

This example trains the agent to go FAST, not just survive. It uses a reward function that balances speed against safety (steering smoothness, wall proximity). Domain randomization is enabled to make the policy robust. Multiple environments run in parallel for faster training.

```bash
python3 examples/4_race_fast.py --steps 1000000 --num-envs 4
python3 examples/4_race_fast.py --eval --render
```

### Example 5: PPO From Scratch (`5_ppo_from_scratch.py`)
**What you'll learn:** Exactly how PPO works, line by line. No libraries hiding the algorithm.

This implements the entire PPO algorithm in ~300 lines of PyTorch. No Stable-Baselines3 abstractions — just the raw math. You can trace the code from observation → action → reward → advantage → policy update. Run with `--render` to watch the car learn in real-time.

```bash
python3 examples/5_ppo_from_scratch.py --render --steps 500000
python3 examples/5_ppo_from_scratch.py --eval --render
```

## Recommended Order

If you're new to RL: 1 → 5 → 2 → 3 → 4

- Start with Example 1 to see the end-to-end pipeline.
- Read Example 5 to understand what's happening inside PPO.
- Then try 2, 3, 4 for advanced concepts.

### Example 6: Custom Reward Functions (`6_custom_reward.py`)
**What you'll learn:** How to write your own reward function and plug it in.

Shows three reward types: Trajectory-Aided Learning (reward matching an expert's speed/steering), Frenet-frame reward (reward based on lateral error and heading error relative to the raceline), and simple progress. If you're designing a reward for a specific racing objective (go fast, follow a line, match an expert), start here.

```bash
python3 examples/6_custom_reward.py --reward tal          # Trajectory-aided learning
python3 examples/6_custom_reward.py --reward frenet       # Frenet-frame reward
python3 examples/6_custom_reward.py --eval --render       # Watch
```

### Example 7: Custom Observations (`7_custom_observations.py`)
**What you'll learn:** How to add new information to what the agent sees.

Shows three observation extensions: opponent-aware (add distance/bearing to another car), dynamics-aware (add steering angle, slip angle, and lateral velocity from the vehicle state), and track-aware (add Frenet coordinates and curvature). If your project needs the agent to see something beyond lidar + velocity, this is the template.

```bash
python3 examples/7_custom_observations.py --obs opponent  # See opponents
python3 examples/7_custom_observations.py --obs dynamics  # See vehicle dynamic state
python3 examples/7_custom_observations.py --obs track     # See track geometry
python3 examples/7_custom_observations.py --eval --render  # Watch
```
