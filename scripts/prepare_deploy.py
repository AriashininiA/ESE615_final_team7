#!/usr/bin/env python3
"""
Prepare Deployment Package for F1TENTH Car
=============================================
Bundles all files needed to run a trained policy on the Jetson.

Creates a deploy folder with:
    - ONNX model (lightweight, no PyTorch needed on Jetson)
    - Config file
    - Inference node
    - Centerline CSV (if localized model)
    - Launch script (run.sh)

Note: With norm_obs=False (default since v28), no normalization
stats are needed. The policy sees the same raw observations
on the car as it did during training.

Usage:
    python3 scripts/prepare_deploy.py --run runs/levine_slam_e2e_raw_*
    python3 scripts/prepare_deploy.py --run runs/levine_slam_localized_*

    # Then copy to car:
    scp -r deploy/levine_slam_e2e_raw_* jetson@<CAR_IP>:~/f1tenth_rl/
"""

import argparse
import os
import sys
import shutil
import pickle
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Prepare deployment package for F1TENTH car")
    parser.add_argument("--run", type=str, required=True, help="Run directory")
    parser.add_argument("--output", type=str, default="deploy", help="Output directory")
    parser.add_argument("--use-best", action="store_true", help="Use best model instead of final")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"ERROR: Run directory not found: {run_dir}")
        return

    run_name = run_dir.name
    deploy_dir = Path(args.output) / run_name
    os.makedirs(deploy_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Preparing deployment: {run_name}")
    print(f"  Output: {deploy_dir}")
    print(f"{'='*50}\n")

    # ---- 1. Export ONNX model ----
    if args.use_best:
        model_path = run_dir / "best_model" / "best_model"
    else:
        model_path = run_dir / "final_model"

    zip_path = str(model_path) + ".zip"
    onnx_path = deploy_dir / "final_model.onnx"

    if not os.path.exists(zip_path):
        print(f"ERROR: Model not found: {zip_path}")
        return

    if not onnx_path.exists():
        print("1. Exporting ONNX model...")
        try:
            import torch
            import torch.nn as nn
            from stable_baselines3 import PPO, SAC, TD3
            import yaml

            with open(run_dir / "config.yaml") as f:
                config = yaml.safe_load(f)

            algo_type = config.get("algorithm", {}).get("type", "ppo")
            AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo_type]
            model = AlgoClass.load(zip_path, device="cpu")
            obs_dim = model.observation_space.shape[0]

            class PolicyWrapper(nn.Module):
                def __init__(self, sb3_policy):
                    super().__init__()
                    self.features_extractor = sb3_policy.features_extractor
                    self.mlp_extractor = sb3_policy.mlp_extractor
                    self.action_net = sb3_policy.action_net

                def forward(self, obs):
                    features = self.features_extractor(obs)
                    latent_pi, _ = self.mlp_extractor(features)
                    return self.action_net(latent_pi)

            wrapper = PolicyWrapper(model.policy)
            wrapper.eval()

            dummy = torch.randn(1, obs_dim, dtype=torch.float32)
            torch.onnx.export(
                wrapper, dummy, str(onnx_path), opset_version=11,
                input_names=["observation"], output_names=["action"],
                dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            )
            size_kb = os.path.getsize(onnx_path) / 1024
            print(f"   ✓ {onnx_path.name} ({size_kb:.0f} KB, obs_dim={obs_dim})")
        except Exception as e:
            print(f"   ✗ ONNX export failed: {e}")
            return
    else:
        print(f"1. ONNX model already exists: {onnx_path.name}")

    # ---- 2. Normalization stats (only if model uses VecNormalize obs) ----
    npz_path = deploy_dir / "obs_norm_stats.npz"
    pkl_path = run_dir / "final_vecnormalize.pkl"

    if pkl_path.exists():
        print("2. Converting normalization stats...")
        try:
            with open(pkl_path, "rb") as f:
                vec_norm = pickle.load(f)
            if hasattr(vec_norm, 'obs_rms') and vec_norm.norm_obs:
                np.savez(
                    str(npz_path),
                    mean=vec_norm.obs_rms.mean,
                    var=vec_norm.obs_rms.var,
                )
                print(f"   ✓ {npz_path.name} (mean shape: {vec_norm.obs_rms.mean.shape})")
            else:
                print("   ✓ norm_obs=False — no obs normalization stats needed")
        except Exception as e:
            print(f"   ⚠ Could not read normalization: {e}")
    else:
        print("2. No normalization file — using raw observations (correct for norm_obs=False)")

    # ---- 3. Copy config ----
    print("3. Copying config...")
    shutil.copy2(run_dir / "config.yaml", deploy_dir / "config.yaml")
    print(f"   ✓ config.yaml")

    # ---- 4. Copy inference node ----
    print("4. Copying inference node...")
    node_src = project_root / "f1tenth_rl" / "ros2" / "inference_node.py"
    if node_src.exists():
        shutil.copy2(node_src, deploy_dir / "inference_node.py")
        print(f"   ✓ inference_node.py")
    else:
        print(f"   ✗ Not found: {node_src}")

    # ---- 5. Check if localized — copy centerline ----
    import yaml
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    obs_type = config.get("observation", {}).get("type", "lidar_state")
    is_localized = (obs_type == "lidar_waypoint")

    if is_localized:
        print("5. Copying centerline (localized model)...")
        map_path = config.get("env", {}).get("map_path", "")
        for suffix in ["_centerline.csv", "_raceline.csv"]:
            candidate = Path(map_path + suffix)
            if candidate.exists():
                shutil.copy2(candidate, deploy_dir / candidate.name)
                print(f"   ✓ {candidate.name}")
                break
        else:
            print(f"   ✗ No centerline found for {map_path}")
    else:
        print("5. End-to-end model — no centerline needed")

    # ---- 6. Create launch script ----
    print("6. Creating launch script...")
    if is_localized:
        centerline_name = Path(map_path).stem + "_centerline.csv"
        launch = (
            "#!/bin/bash\n"
            "cd \"$(dirname \"$0\")\"\n"
            "python3 inference_node.py --ros-args \\\n"
            "    -p model_path:=final_model.onnx \\\n"
            "    -p config_path:=config.yaml \\\n"
            "    -p use_onnx:=true \\\n"
            f"    -p waypoint_path:={centerline_name} \\\n"
            "    -p max_speed:=2.0\n"
        )
    else:
        launch = (
            "#!/bin/bash\n"
            "cd \"$(dirname \"$0\")\"\n"
            "python3 inference_node.py --ros-args \\\n"
            "    -p model_path:=final_model.onnx \\\n"
            "    -p config_path:=config.yaml \\\n"
            "    -p use_onnx:=true \\\n"
            "    -p max_speed:=2.0\n"
        )

    launch_path = deploy_dir / "run.sh"
    with open(launch_path, "w") as f:
        f.write(launch)
    os.chmod(launch_path, 0o755)
    print(f"   ✓ run.sh")

    # ---- Summary ----
    files = list(deploy_dir.iterdir())
    print(f"\n{'='*50}")
    print(f"  Deployment package ready!")
    print(f"{'='*50}")
    print(f"  Directory: {deploy_dir}")
    print(f"  Mode:      {'LOCALIZED' if is_localized else 'END-TO-END'}")
    print(f"  Files:")
    for f in sorted(files):
        size = os.path.getsize(f)
        if size > 1024:
            print(f"    {f.name:30s} {size/1024:.0f} KB")
        else:
            print(f"    {f.name:30s} {size} B")

    print(f"\n  Copy to car:")
    print(f"    scp -r {deploy_dir} jetson@<CAR_IP>:~/f1tenth_rl/")
    print(f"\n  Run on car:")
    print(f"    cd ~/f1tenth_rl/{run_name}")
    print(f"    ./run.sh")
    print()


if __name__ == "__main__":
    main()
