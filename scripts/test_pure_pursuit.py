#!/usr/bin/env python3
"""
Test Pure Pursuit Controller
===============================
Visualize the pure pursuit controller driving a lap before using it
for demo collection or as a multi-agent opponent.

ALWAYS test this before collecting demos — if pure pursuit crashes,
the demos will be garbage and imitation learning will fail.

Usage:
    # Test on your map
    python3 scripts/test_pure_pursuit.py --map maps/levine_slam/levine_slam

    # Adjust speed
    python3 scripts/test_pure_pursuit.py --map maps/levine_slam/levine_slam --speed 2.0

    # Run multiple episodes and print stats
    python3 scripts/test_pure_pursuit.py --map maps/levine_slam/levine_slam --episodes 10

    # Without rendering (just stats)
    python3 scripts/test_pure_pursuit.py --map maps/levine_slam/levine_slam --episodes 20 --no-render
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test pure pursuit controller")
    parser.add_argument("--map", type=str, required=True, help="Map path (e.g., maps/levine_slam/levine_slam)")
    parser.add_argument("--config", type=str, default="configs/sim2real_e2e.yaml")
    parser.add_argument("--speed", type=float, default=3.0, help="Target speed (m/s)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering")
    parser.add_argument("--lookahead", type=float, default=None, help="Override lookahead distance")
    parser.add_argument("--raceline", type=str, default=None,
                        help="Custom raceline CSV (default: uses auto-generated centerline)")
    args = parser.parse_args()

    # Load config
    config_path = Path(project_root) / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["env"]["map_path"] = args.map
    if "expert" not in config:
        config["expert"] = {"pure_pursuit": {}}
    config["expert"]["pure_pursuit"]["target_speed"] = args.speed
    if args.lookahead:
        config["expert"]["pure_pursuit"]["lookahead_distance"] = args.lookahead
        config["expert"]["pure_pursuit"]["adaptive_lookahead"] = False

    # Create env
    from f1tenth_rl.envs.wrapper import F1TenthWrapper
    from f1tenth_rl.experts.pure_pursuit import PurePursuitController

    render_mode = None if args.no_render else "human"
    env = F1TenthWrapper(config, render_mode=render_mode)

    if env.waypoints is None:
        print("ERROR: No waypoints found! Generate centerline first:")
        print(f"  python3 scripts/generate_centerline.py --map {args.map}")
        return

    # Use custom raceline if provided, otherwise use centerline
    if args.raceline:
        data = np.loadtxt(args.raceline, delimiter=",", skiprows=1)
        if data.shape[1] >= 3:
            expert_wp = data[:, :3]
        else:
            expert_wp = np.column_stack([data[:, :2], np.ones(len(data)) * args.speed])
        print(f"  Using custom raceline: {args.raceline} ({len(expert_wp)} waypoints)")
    else:
        expert_wp = env.waypoints

    expert_cfg = dict(config.get("expert", {}))
    expert_cfg["_action_config"] = config.get("action", {})
    pp = PurePursuitController(expert_wp, expert_cfg)

    print(f"\n{'='*50}")
    print(f"  Pure Pursuit Test")
    print(f"{'='*50}")
    print(f"  Map:        {args.map}")
    print(f"  Speed:      {args.speed} m/s")
    print(f"  Waypoints:  {len(env.waypoints)}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Render:     {not args.no_render}")
    print(f"{'='*50}\n")

    # Run episodes
    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        flat_obs = info["raw_obs"]
        speeds = []
        steers = []
        done = False
        step = 0

        while not done and step < args.max_steps:
            action = pp.get_normalized_action(flat_obs, ego_idx=0)
            obs, reward, terminated, truncated, info = env.step(action)
            flat_obs = info["raw_obs"]
            step += 1
            done = terminated or truncated

            speeds.append(info.get("ego_speed", 0))
            steers.append(abs(info.get("physical_action", [0, 0])[0]))

            if render_mode:
                env.render()

            if step % 500 == 0 and not args.no_render:
                print(f"  Step {step}: speed={info['ego_speed']:.1f} m/s, "
                      f"progress={info['progress']:.1%}")

        collision = info.get("ego_collision", False)
        progress = info.get("progress", 0)
        lap_time = info.get("ego_lap_time", 0)

        result = {
            "episode": ep + 1,
            "progress": progress,
            "collision": collision,
            "steps": step,
            "avg_speed": np.mean(speeds),
            "max_speed": np.max(speeds),
            "lap_time": lap_time,
            "steer_smoothness": np.std(np.diff(steers)) if len(steers) > 1 else 0,
        }
        results.append(result)

        status = "CRASH" if collision else f"OK ({lap_time:.1f}s)" if lap_time > 0 else "OK"
        print(f"  Episode {ep+1}/{args.episodes}: "
              f"progress={progress:.1%} | speed={np.mean(speeds):.1f} m/s | {status}")

    env.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"  Results")
    print(f"{'='*50}")
    progresses = [r["progress"] for r in results]
    crashes = [r["collision"] for r in results]
    speeds = [r["avg_speed"] for r in results]
    lap_times = [r["lap_time"] for r in results if r["lap_time"] > 0]

    print(f"  Avg progress:  {np.mean(progresses):.1%}")
    print(f"  Crash rate:    {np.mean(crashes):.0%} ({sum(crashes)}/{len(crashes)})")
    print(f"  Avg speed:     {np.mean(speeds):.2f} m/s")
    if lap_times:
        print(f"  Avg lap time:  {np.mean(lap_times):.2f}s (best: {min(lap_times):.2f}s)")
    print(f"{'='*50}")

    # Verdict
    crash_rate = np.mean(crashes)
    avg_progress = np.mean(progresses)
    if crash_rate == 0 and avg_progress > 0.95:
        print(f"\n  ✓ GOOD — Safe to collect demos at {args.speed} m/s")
        print(f"  Next: python3 scripts/collect_demos.py --config {args.config} "
              f"--episodes 100 --speed {args.speed}")
    elif crash_rate < 0.2 and avg_progress > 0.8:
        print(f"\n  ~ OK — Mostly works but some crashes. Try lower speed:")
        print(f"  python3 scripts/test_pure_pursuit.py --map {args.map} --speed {args.speed * 0.7:.1f}")
    else:
        print(f"\n  ✗ BAD — Pure pursuit is crashing too much.")
        print(f"  Try: python3 scripts/test_pure_pursuit.py --map {args.map} --speed {args.speed * 0.5:.1f}")
        print(f"  Or regenerate centerline: python3 scripts/generate_centerline.py --map {args.map}")


if __name__ == "__main__":
    main()
