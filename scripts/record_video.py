#!/usr/bin/env python3
"""
Headless trajectory video recorder for trained F1TENTH RL policies.

This avoids the simulator GUI/Qt renderer by drawing the evaluated policy's
pose trace directly onto the map image and writing an MP4.
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def resolve_run(run_patterns):
    """Resolve one or more run paths/globs to the newest matching directory."""
    matches = []
    for pattern in run_patterns:
        expanded = glob.glob(pattern)
        matches.extend(expanded if expanded else [pattern])

    dirs = [Path(p) for p in matches if Path(p).is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directory found for: {run_patterns}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def find_model_and_config(run_dir, use_best=False):
    run_dir = Path(run_dir)
    model_path = run_dir / ("best_model/best_model" if use_best else "final_model")
    config_path = run_dir / "config.yaml"
    norm_path = run_dir / "final_vecnormalize.pkl"
    return str(model_path), str(config_path), str(norm_path)


def load_map(config):
    map_path = Path(project_root) / config["env"]["map_path"]
    yaml_candidates = [map_path.with_suffix(".yaml"), Path(str(map_path) + "_map.yaml")]
    yaml_path = next((p for p in yaml_candidates if p.exists()), None)
    if yaml_path is None:
        raise FileNotFoundError(f"Map YAML not found for {map_path}")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    image_path = yaml_path.parent / meta.get("image", map_path.name + ".png")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Map image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, float(meta["resolution"]), np.array(meta["origin"][:2], dtype=np.float32)


def world_to_pixel(x, y, origin, resolution, image_height):
    px = int(round((x - origin[0]) / resolution))
    py = int(round(image_height - (y - origin[1]) / resolution))
    return px, py


def draw_centerline(frame, config, origin, resolution):
    centerline_path = Path(project_root) / (config["env"]["map_path"] + "_centerline.csv")
    if not centerline_path.exists():
        return

    try:
        data = np.loadtxt(centerline_path, delimiter=",", skiprows=1)
    except Exception:
        return

    if data.ndim == 1:
        data = data.reshape(1, -1)

    points = [
        world_to_pixel(float(x), float(y), origin, resolution, frame.shape[0])
        for x, y in data[:, :2]
    ]
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(frame, p0, p1, (120, 120, 120), 2, lineType=cv2.LINE_AA)


def draw_car(frame, x, y, theta, origin, resolution):
    center = world_to_pixel(x, y, origin, resolution, frame.shape[0])
    heading = (
        int(round(center[0] + 18 * np.cos(theta))),
        int(round(center[1] - 18 * np.sin(theta))),
    )
    cv2.circle(frame, center, 8, (0, 80, 255), -1, lineType=cv2.LINE_AA)
    cv2.line(frame, center, heading, (0, 0, 255), 3, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Record a headless F1TENTH policy video")
    parser.add_argument("--run", nargs="+", required=True, help="Run directory or glob")
    parser.add_argument("--output", type=str, default=None, help="Output .mp4 path")
    parser.add_argument("--use-best", action="store_true", help="Use best_model instead of final_model")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5, help="Record every N simulator steps")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--scale", type=float, default=0.5, help="Output image scale")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    model_path, config_path, norm_path = find_model_and_config(run_dir, args.use_best)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if args.max_steps is not None:
        config["env"]["max_steps"] = args.max_steps

    from stable_baselines3 import PPO, SAC, TD3
    from f1tenth_rl.envs.wrapper import F1TenthWrapper

    algo_type = config["algorithm"]["type"]
    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo_type]
    model = algo_cls.load(model_path, device="cpu")

    obs_rms = None
    if os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            obs_rms = pickle.load(f).obs_rms

    map_image, resolution, origin = load_map(config)
    base_frame = map_image.copy()
    draw_centerline(base_frame, config, origin, resolution)

    output_path = Path(args.output) if args.output else run_dir / "policy_video.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_size = (
        int(base_frame.shape[1] * args.scale),
        int(base_frame.shape[0] * args.scale),
    )
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    env = F1TenthWrapper(config, render_mode=None)
    obs, info = env.reset()
    done = False
    step = 0
    episode_return = 0.0
    trail = []

    while not done:
        obs_input = obs
        if obs_rms is not None:
            obs_input = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            obs_input = np.clip(obs_input, -10.0, 10.0).astype(np.float32)

        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        step += 1

        raw = info["raw_obs"]
        x = float(raw["poses_x"][0])
        y = float(raw["poses_y"][0])
        theta = float(raw["poses_theta"][0])
        trail.append((x, y))

        if step % args.stride == 0 or done:
            frame = base_frame.copy()
            pixels = [
                world_to_pixel(px, py, origin, resolution, frame.shape[0])
                for px, py in trail
            ]
            for p0, p1 in zip(pixels[:-1], pixels[1:]):
                cv2.line(frame, p0, p1, (0, 200, 0), 3, lineType=cv2.LINE_AA)
            draw_car(frame, x, y, theta, origin, resolution)

            label = (
                f"step={step}  progress={info.get('progress', 0):.1%}  "
                f"speed={info.get('ego_speed', 0):.2f} m/s"
            )
            cv2.putText(frame, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (255, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)

    env.close()
    writer.release()

    print(f"Saved video: {output_path}")
    print(f"Steps: {step}")
    print(f"Return: {episode_return:.2f}")
    print(f"Progress: {info.get('progress', 0):.2%}")
    print(f"Collision: {info.get('ego_collision', False)}")


if __name__ == "__main__":
    main()
