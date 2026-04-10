#!/usr/bin/env python3
"""
Expert Demonstration Collection
=================================
Collect demonstrations from the pure pursuit controller for
behavioral cloning and imitation learning.

Usage:
    # Collect 100 episodes of expert demonstrations
    python scripts/collect_demos.py --episodes 100

    # Collect with visualization
    python scripts/collect_demos.py --episodes 50 --render

    # Custom output path
    python scripts/collect_demos.py --output demos/my_demos.npz

    # Different expert speed
    python scripts/collect_demos.py --speed 6.0
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations for imitation learning"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="demos/expert_demos.npz")
    parser.add_argument("--speed", type=float, default=None, help="Expert target speed")
    parser.add_argument("--raceline", type=str, default=None,
                        help="Custom raceline CSV for expert to follow")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    # Load config
    config_path = Path(project_root) / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override speed if specified
    if args.speed:
        if "expert" not in config:
            config["expert"] = {}
        if "pure_pursuit" not in config["expert"]:
            config["expert"]["pure_pursuit"] = {}
        config["expert"]["pure_pursuit"]["target_speed"] = args.speed

    # Custom raceline
    if args.raceline:
        if "expert" not in config:
            config["expert"] = {}
        config["expert"]["waypoint_path"] = args.raceline

    # Collect demonstrations
    from f1tenth_rl.experts.demo_collector import DemoCollector

    collector = DemoCollector(config)
    collector.collect(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
    )

    # Save
    output_path = Path(project_root) / args.output
    os.makedirs(output_path.parent, exist_ok=True)
    collector.save(str(output_path))
    collector.close()

    print(f"\nDemonstrations saved to {output_path}")
    print("\nNext steps:")
    print(f"  1. Train BC:  python scripts/train.py --bc-pretrain {args.output}")
    print(f"  2. Or use standalone BC: see f1tenth_rl.agents.imitation.ImitationTrainer")


if __name__ == "__main__":
    main()
