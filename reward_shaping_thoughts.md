# Reward Shaping & Training Guide for F1Tenth RL Racing

## TL;DR — What to Do

Your codebase already has the building blocks. Here's the recommended path:

1. **Week 1**: Get Pure Pursuit baseline running → measure lap times on Levine
2. **Week 2**: Train with `progress` reward at low speed (3 m/s) to get a working policy
3. **Week 2-3**: Switch to your **TAL reward** (which matches your proposal), increase speed
4. **Week 3-4**: Add domain randomization, train for 10M+ steps, deploy

---

## 1. Reward Functions Already in Your Codebase

Your repo has **three reward types** in [rewards.py](file:///Users/yeongheonlee/Desktop/Projects/f1tenth_rl_humble/f1tenth_rl/envs/rewards.py) plus additional examples in [6_custom_reward.py](file:///Users/yeongheonlee/Desktop/Projects/f1tenth_rl_humble/examples/6_custom_reward.py):

| Reward Type | Config Key | What It Rewards | Best For |
|---|---|---|---|
| **Progress** | `progress` | Forward movement along centerline | Getting a working policy fast |
| **CTH** | `cth` | Heading alignment + staying near centerline | Smooth sim2real driving |
| **Speed** | `speed` | Raw velocity | Only with heavy penalties |
| **TAL** | *(example only)* | Matching expert actions | Racing faster than PP baseline |

### Progress Reward (Start Here)

```
r = progress_weight × (Δs / track_length) + speed_weight × max(0, v)
```

- **Δs** = how far forward you moved along the centerline this timestep
- Simple, robust, fast to converge
- Config: `reward.type: "progress"`, `progress_weight: 10.0`, `speed_weight: 0.1`

> [!TIP]
> Start with this. If the agent can't learn to complete laps with progress reward, nothing else will work either. It's your sanity check.

### CTH Reward (Cross-Track + Heading)

```
r = β_heading × v × cos(θ_error) − β_crosstrack × d_crosstrack
```

- Rewards pointing along the track AND going fast
- Naturally produces smooth driving (good for real car)
- Config: `reward.type: "cth"`, `heading_weight: 0.04`, `crosstrack_weight: 0.004`

> [!NOTE]
> The sim2real config (`configs/sim2real_e2e.yaml`) already uses CTH with `heading_weight: 0.2`, `crosstrack_weight: 0.1`. These higher weights enforce tighter centerline tracking.

---

## 2. Implementing Your Proposal's TAL Reward

Your proposal defines the TAL reward from Evans et al. [1]:

```
r_TAL = 1 − |v_agent − v_ref| − |δ_agent − δ_ref|
```

**Good news**: this is already implemented in [6_custom_reward.py](file:///Users/yeongheonlee/Desktop/Projects/f1tenth_rl_humble/examples/6_custom_reward.py#L152-L179). Here's how it works and how to integrate it properly:

### How TAL Works

```python
def _tal_reward(self, obs, action, collision):
    if collision:
        return -10.0
    
    # Get what pure pursuit would do at the current position
    expert_action = self.expert.get_normalized_action(obs, ego_idx=0)
    
    # Penalize deviation from expert
    steer_error = abs(action[0] - expert_action[0])   # both in [-1, 1]
    speed_error = abs(action[1] - expert_action[1])   # both in [-1, 1]
    
    # TAL component: 1.0 when perfectly matching expert, 0.0 when maximally wrong
    tal_reward = 1.0 - 0.5 * steer_error - 0.5 * speed_error
    
    # CRITICAL: also add progress reward so agent cares about forward motion
    progress_reward = 5.0 * (delta_progress / track_length)
    
    return tal_reward + progress_reward
```

### Why TAL Alone Is Not Enough

> [!IMPORTANT]
> If you use **only** `r_TAL = 1 − |v_agent − v_ref| − |δ_agent − δ_ref|` from your proposal (Equation 1), the agent will learn to **perfectly imitate pure pursuit** — but never exceed it. You need to combine TAL with a progress component so the agent is *biased toward* the expert but can *discover faster strategies*.

### Recommended Combined Reward for Your Project

Here's what I recommend for your actual training:

```
r = α × r_TAL + β × r_progress + γ × r_speed − λ × r_steer_change + r_collision + r_lap
```

With concrete weights:

| Component | Symbol | Weight | Purpose |
|---|---|---|---|
| TAL matching | α | 1.0 → 0.3 (anneal) | Bias toward expert early, free agent later |
| Progress | β | 5.0 → 10.0 (increase) | Reward forward motion |
| Speed bonus | γ | 0.1 | Encourage going fast |
| Steering change | λ | 0.3 | Prevent oscillation (critical for real car) |
| Collision | — | −10.0 | Hard penalty |
| Lap completion | — | +10.0 | Bonus |

> [!TIP]
> **Anneal TAL weight over training**: Start with high TAL weight so the agent learns basic track behavior from the expert.  Gradually reduce it so the agent can discover faster-than-expert strategies. This is the key insight from Evans et al.

### Concrete Implementation

To add a proper TAL reward to the framework, add a new class to [rewards.py](file:///Users/yeongheonlee/Desktop/Projects/f1tenth_rl_humble/f1tenth_rl/envs/rewards.py):

```python
class TALReward(RewardFunction):
    """
    Trajectory-Aided Learning reward (Evans et al. 2023).
    Combines expert action matching with progress reward.
    """
    
    def __init__(self, config, waypoints, expert_controller):
        super().__init__(config)
        self.waypoints = waypoints[:, :2]
        self.expert = expert_controller
        self.tal_weight = config.get("tal_weight", 1.0)
        self.progress_weight = config.get("progress_weight", 5.0)
        self.speed_weight = config.get("speed_weight", 0.1)
        
        # For progress computation
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        self.cumulative_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.cumulative_dist[-1]
        self.prev_progress_dist = 0.0
    
    def _reset_impl(self, obs_dict, ego_idx):
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        self.prev_progress_dist = self._get_progress_dist(x, y)
    
    def _compute_impl(self, obs_dict, ego_idx, action):
        # --- TAL component ---
        expert_action = self.expert.get_normalized_action(obs_dict, ego_idx)
        
        # Normalize the physical action back to [-1, 1] for comparison
        max_steer = 0.4189
        max_speed = self.config.get("max_speed", 8.0)
        min_speed = self.config.get("min_speed", 0.5)
        
        agent_steer_norm = action[0] / max_steer
        agent_speed_norm = 2.0 * (action[1] - min_speed) / (max_speed - min_speed) - 1.0
        
        steer_error = abs(agent_steer_norm - expert_action[0])
        speed_error = abs(agent_speed_norm - expert_action[1])
        
        tal_reward = self.tal_weight * (1.0 - 0.5 * steer_error - 0.5 * speed_error)
        
        # --- Progress component ---
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        vel = float(obs_dict["linear_vels_x"][ego_idx])
        
        current_dist = self._get_progress_dist(x, y)
        delta = current_dist - self.prev_progress_dist
        if delta < -self.total_length * 0.5:
            delta += self.total_length
        elif delta > self.total_length * 0.5:
            delta -= self.total_length
        self.prev_progress_dist = current_dist
        self._progress += max(0, delta) / self.total_length
        
        progress_reward = self.progress_weight * (delta / self.total_length)
        speed_bonus = self.speed_weight * max(0, vel)
        
        return tal_reward + progress_reward + speed_bonus
    
    def _get_progress_dist(self, x, y):
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        return self.cumulative_dist[np.argmin(dists)]
```

---

## 3. Recommended Training Strategy (Phased)

### Phase 1: Sanity Check (1-2 hours)

**Goal**: Verify the environment works, get PP baseline lap times.

```bash
# Test pure pursuit on your track
python scripts/test_pure_pursuit.py --map maps/levine_blocked/levine_blocked

# Quick RL training sanity check (should learn to not crash within 200k steps)
python scripts/train.py --config configs/levine.yaml \
    --total-steps 200000 --num-envs 4 --no-wandb
```

**Config**: Use `configs/levine.yaml` as-is. `reward.type: "progress"`, `max_speed: 3.0`.

### Phase 2: Low-Speed Policy (2-4 hours)

**Goal**: Get a policy that reliably completes laps at conservative speed.

```yaml
# Modify configs/levine.yaml:
action:
  max_speed: 3.0          # Keep it safe
  min_speed: 0.5
  
reward:
  type: "progress"        # Simple, reliable
  progress_weight: 10.0
  collision_penalty: -10.0
  steering_change_penalty: 0.3  # Add this for smooth steering

algorithm:
  total_timesteps: 1_000_000
```

```bash
python scripts/train.py --config configs/levine.yaml --total-steps 1000000
```

### Phase 3: TAL Training — Faster Than Baseline (4-8 hours)

**Goal**: Use TAL reward to learn expert-like behavior, then exceed it.

Create a new config `configs/levine_tal.yaml`:

```yaml
# Copy from levine.yaml, then change:
action:
  max_speed: 5.0          # Increase to allow faster driving
  min_speed: 0.5

reward:
  type: "tal"             # Your proposal's reward function
  tal_weight: 1.0         # Start high, matches expert
  progress_weight: 5.0    # Also reward forward motion
  speed_weight: 0.1       # Small speed bonus
  collision_penalty: -10.0
  lap_bonus: 10.0
  steering_change_penalty: 0.3

algorithm:
  total_timesteps: 5_000_000
  ppo:
    ent_coef: 0.005       # Less exploration — TAL provides guidance
```

> [!NOTE]
> To implement TAL weight annealing, you'd modify the reward function to reduce `tal_weight` based on `self._progress` (how many total laps the agent has experienced). Start at 1.0, anneal to 0.3 over training. This lets the agent eventually "outgrow" the expert.

### Phase 4: Sim2Real Preparation (Week 4)

**Goal**: Domain-randomized policy that transfers to real car.

```bash
python scripts/train.py --config configs/sim2real_e2e.yaml \
    --map maps/levine_slam/levine_slam \
    --total-steps 10000000 --num-envs 12 --device cuda
```

Key settings that matter for real-car transfer:
- `smoothing_alpha: 0.3` (prevents servo jitter)
- `domain_randomization.mode: "curriculum"` (critical!)
- `steering_change_penalty: 0.5` (smooth steering)
- `max_speed: 3.0` initially (increase once it works)

---

## 4. Your Proposal vs. Your Codebase — Gap Analysis

| Proposal Says | Codebase Has | Action Needed |
|---|---|---|
| TD3 algorithm | ✅ TD3 in `configs/*.yaml` and `sb3_trainer.py` | Just set `algorithm.type: "td3"` |
| Stacked LiDAR (prev + current) | ✅ `frame_stack` in observation config | Set `frame_stack: 2` |
| TAL reward with expert matching | ✅ Example in `6_custom_reward.py` | Add `TALReward` class to `rewards.py` |
| Pure Pursuit baseline | ✅ `pure_pursuit.py` + `test_pure_pursuit.py` | Run it, record lap times |
| Collision + lap completion terms | ✅ Built into `RewardFunction` base class | Already done |
| Bounded speed for safety | ✅ `max_speed`/`min_speed` in action config | Set conservative values |

### Algorithm: PPO vs TD3

Your proposal mentions TD3 (from Evans et al.), but **I recommend starting with PPO**:

| | PPO | TD3 |
|---|---|---|
| Stability | Very stable, hard to diverge | Can be unstable, sensitive to hyperparams |
| Sample efficiency | Less efficient | More efficient (replay buffer) |
| Exploration | Stochastic policy (natural exploration) | Deterministic + noise (needs tuning) |
| Debugging | Easier to understand what's going wrong | Harder to diagnose issues |

> [!TIP]
> Start with PPO for development. Switch to TD3 as an experiment for your paper. You can compare both in your evaluation section — that actually makes a stronger paper!

---

## 5. Common Pitfalls & How to Avoid Them

### Pitfall 1: Reward Magnitude Mismatch
If your collision penalty is −10 but your per-step reward is 0.001, the agent learns "never move" because the risk isn't worth it.

**Fix**: Keep reward components on similar scales. With `progress_weight: 10.0` and `collision_penalty: -10.0`, one collision costs roughly one full lap of progress. That's a good balance.

### Pitfall 2: Sparse Rewards
If the only signal is "complete the lap → +10" and "crash → −10", the agent has no gradient to learn from during the 3000 steps in between.

**Fix**: Your progress reward is **dense** (reward every step). TAL is also dense. You're fine.

### Pitfall 3: The Agent Learns to Drive in Circles
With pure progress reward, the agent sometimes finds a local optimum: drive in a tight circle near a waypoint to accumulate progress.

**Fix**: The `_get_progress_dist` function with wrap-around handling prevents this. Also add `survival_reward: 0.0` (don't reward doing nothing).

### Pitfall 4: Sim2Real Steering Oscillation
The most common failure when deploying to real car: the learned policy oscillates the steering rapidly because it never learned that jitter is bad.

**Fix**: `steering_change_penalty: 0.3-0.5` during training, `smoothing_alpha: 0.3` in action config.

### Pitfall 5: Training Too Fast Before Basic Skills
If you jump to `max_speed: 8.0` before the agent can drive at `3.0`, it just crashes constantly and never learns.

**Fix**: Phase the training as described above (Phase 1-4). Speed curriculum is important.

---

## 6. Evaluation Setup

For your paper's evaluation metrics (from the proposal), log these during training:

```python
# These are already tracked in info dict from wrapper.py:
info["ego_speed"]       # Current velocity
info["ego_collision"]   # Whether crashed this step
info["ego_lap_count"]   # Laps completed
info["ego_lap_time"]    # Time per lap
info["progress"]        # Track progress [0, 1]
info["physical_action"] # Actual steer/speed sent to car
```

### Metrics to Report

| Metric | How to Compute |
|---|---|
| Average lap time | `ego_lap_time` when `ego_lap_count` increments |
| Min lap time | Minimum of above |
| Lap completion rate | Episodes where `ego_lap_count ≥ 1` / total episodes |
| Consecutive collision-free laps | Count sequential laps without `ego_collision` |
| Average progress before failure | `progress` at episode end when `ego_collision` is True |
| Crash frequency | `ego_collision` count / total steps |
| Steering smoothness | `std(diff(physical_action[:, 0]))` over an episode |

---

## 7. Quick-Start Checklist

- [ ] Run Pure Pursuit baseline: `python scripts/test_pure_pursuit.py`
- [ ] Record PP lap times on Levine (this is your "bar to beat")
- [ ] Train progress reward, 1M steps, 3 m/s max: verify laps complete
- [ ] Add TAL reward class to `rewards.py` (code above)
- [ ] Train TAL reward, 5M steps, 5 m/s max: compare lap times to PP
- [ ] Enable domain randomization, train 10M steps
- [ ] Export model for real car: `python scripts/prepare_deploy.py`
- [ ] Test on real car at 2 m/s first, gradually increase
