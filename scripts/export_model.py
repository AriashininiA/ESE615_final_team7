#!/usr/bin/env python3
"""
Export & Benchmark Model for Jetson Deployment
=================================================
Exports trained SB3 models to ONNX for fast inference on Jetson Orin Nano.

ONNX Runtime on Jetson is 5-10x faster than PyTorch for small MLPs.
With TensorRT EP, inference drops to <1ms.

Usage:
    # Export from run directory
    python scripts/export_model.py --run runs/sim2real_2026-04-04_23-19-05

    # Export specific model
    python scripts/export_model.py --model checkpoints/my_model

    # Benchmark inference speed
    python scripts/export_model.py --run runs/sim2real_* --benchmark

    # Full pipeline: export + benchmark + test
    python scripts/export_model.py --run runs/sim2real_* --benchmark --test-obs 112
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_model(run_or_model: str):
    """Resolve run dir or model path."""
    p = Path(run_or_model)
    if p.is_dir():
        return str(p / "final_model"), str(p / "config.yaml"), str(p / "final_vecnormalize.pkl")
    return str(p), str(p) + "_config.yaml", str(p) + "_vecnormalize.pkl"


def export_onnx(model_path: str, config: dict, output_path: str = None):
    """
    Export SB3 model to ONNX.

    The exported model takes a flat observation vector and outputs
    the deterministic action (no value head, no log_prob).
    """
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO, SAC, TD3

    algo_type = config.get("algorithm", {}).get("type", "ppo")
    AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo_type]

    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    model = AlgoClass.load(zip_path, device="cpu")
    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]

    print(f"Model: {algo_type.upper()}")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")

    # Extract just the policy MLP (actor_mean for deterministic action)
    policy = model.policy

    # Create a wrapper that only outputs the deterministic action
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

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Export
    if output_path is None:
        output_path = model_path.replace(".zip", "") + ".onnx"

    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    # Verify
    file_size = os.path.getsize(output_path) / 1024
    print(f"\n  Exported: {output_path} ({file_size:.0f} KB)")

    # Quick verification
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, obs_dim).astype(np.float32)
        result = session.run(None, {"observation": test_input})
        print(f"  Verification: output shape = {result[0].shape}, values = {result[0].squeeze()}")
        print(f"  ✓ ONNX export verified")
    except ImportError:
        print("  [INFO] Install onnxruntime to verify: pip install onnxruntime")

    return output_path


def benchmark(model_path: str, obs_dim: int, n_iterations: int = 1000):
    """
    Benchmark inference speed for PyTorch and ONNX.

    Target: <5ms on Jetson Orin Nano for real-time control at 40Hz.
    """
    print(f"\n{'='*60}")
    print(f"  Inference Benchmark ({n_iterations} iterations)")
    print(f"{'='*60}")

    dummy_obs = np.random.randn(1, obs_dim).astype(np.float32)

    # ---- PyTorch benchmark ----
    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if os.path.exists(zip_path):
        try:
            from stable_baselines3 import PPO
            import torch

            model = PPO.load(zip_path, device="cpu")
            torch.set_num_threads(2)  # Simulate Jetson

            # Warmup
            for _ in range(50):
                model.predict(dummy_obs.squeeze(), deterministic=True)

            # Benchmark
            times = []
            for _ in range(n_iterations):
                t = time.perf_counter()
                model.predict(dummy_obs.squeeze(), deterministic=True)
                times.append((time.perf_counter() - t) * 1000)

            pt_mean = np.mean(times)
            pt_p95 = np.percentile(times, 95)
            pt_p99 = np.percentile(times, 99)
            print(f"\n  PyTorch (CPU, 2 threads):")
            print(f"    Mean: {pt_mean:.2f} ms")
            print(f"    P95:  {pt_p95:.2f} ms")
            print(f"    P99:  {pt_p99:.2f} ms")
            print(f"    Max Hz: {1000/pt_mean:.0f}")
        except Exception as e:
            print(f"  PyTorch benchmark failed: {e}")

    # ---- ONNX benchmark ----
    onnx_path = model_path.replace(".zip", "") + ".onnx"
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            print(f"\n  ONNX Runtime providers: {providers}")

            for provider in ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                if provider not in providers:
                    continue

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 2

                session = ort.InferenceSession(
                    onnx_path, sess_options=sess_options, providers=[provider]
                )
                input_name = session.get_inputs()[0].name

                # Warmup
                for _ in range(50):
                    session.run(None, {input_name: dummy_obs})

                # Benchmark
                times = []
                for _ in range(n_iterations):
                    t = time.perf_counter()
                    session.run(None, {input_name: dummy_obs})
                    times.append((time.perf_counter() - t) * 1000)

                mean = np.mean(times)
                p95 = np.percentile(times, 95)
                p99 = np.percentile(times, 99)
                print(f"\n  ONNX ({provider}):")
                print(f"    Mean: {mean:.2f} ms")
                print(f"    P95:  {p95:.2f} ms")
                print(f"    P99:  {p99:.2f} ms")
                print(f"    Max Hz: {1000/mean:.0f}")

        except ImportError:
            print("\n  ONNX benchmark skipped (install onnxruntime)")
    else:
        print(f"\n  No ONNX model found at {onnx_path}")
        print(f"  Run without --benchmark first to export")

    print(f"\n  Target for real-time (40Hz): <25ms per inference")
    print(f"{'='*60}")


def count_parameters(model_path: str):
    """Count model parameters."""
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        total = sum(p.numel() for p in model.policy.parameters())
        trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        print(f"\n  Parameters: {total:,} total ({trainable:,} trainable)")
        print(f"  Size: ~{total * 4 / 1024:.0f} KB (float32)")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX for Jetson deployment")
    parser.add_argument("--run", type=str, default=None, help="Run directory")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--test-obs", type=int, default=None, help="Observation dim for benchmark")
    args = parser.parse_args()

    source = args.run or args.model
    if not source:
        parser.error("Provide --run or --model")

    model_path, config_path, norm_path = find_model(source)

    # Load config
    import yaml
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"algorithm": {"type": "ppo"}}

    # Export
    onnx_path = export_onnx(model_path, config, args.output)

    # Count params
    count_parameters(model_path)

    # Benchmark
    if args.benchmark:
        obs_dim = args.test_obs
        if obs_dim is None:
            try:
                from stable_baselines3 import PPO
                m = PPO.load(model_path, device="cpu")
                obs_dim = m.observation_space.shape[0]
            except Exception:
                obs_dim = 112  # default for 108 beams + 4 state features
        benchmark(model_path, obs_dim)

    print(f"\nDeployment files:")
    print(f"  ONNX model:  {onnx_path}")
    if os.path.exists(norm_path):
        print(f"  Norm stats:  {norm_path}")
    print(f"  Config:      {config_path}")
    print(f"\nDeploy on Jetson:")
    print(f"  ros2 run f1tenth_rl inference_node --ros-args \\")
    print(f"    -p model_path:={onnx_path} \\")
    print(f"    -p use_onnx:=true \\")
    print(f"    -p max_speed:=2.0")


if __name__ == "__main__":
    main()
