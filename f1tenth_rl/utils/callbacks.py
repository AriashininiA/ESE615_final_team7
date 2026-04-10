"""
Training Callbacks
==================
These run automatically during training to track and log racing metrics.

SB3 calls these callbacks at specific points during training (after each
rollout, after each evaluation, etc.). They collect racing-specific stats
like average speed, lap times, collision rate, and progress — things that
SB3 doesn't track by default because they're specific to our environment.

The metrics show up in WandB and TensorBoard under the "racing/" prefix.
"""

import os
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class RacingMetricsCallback(BaseCallback):
    """
    Log racing-specific metrics: speed, progress, collisions, lap times.

    Logs to both TensorBoard (via SB3 logger) and optionally WandB.
    """

    def __init__(self, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.use_wandb = use_wandb

        # Rolling buffers for episode-level metrics
        self.episode_speeds = deque(maxlen=100_000)
        self.episode_max_speeds = deque(maxlen=500)
        self.episode_progresses = deque(maxlen=500)
        self.episode_collisions = deque(maxlen=500)
        self.episode_lap_times = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=500)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # Track per-step speed
            speed = info.get("ego_speed", None)
            if speed is not None:
                self.episode_speeds.append(speed)

            # On episode end, record summary metrics
            done = dones[i] if i < len(dones) else False
            if done:
                progress = info.get("progress", 0)
                collision = info.get("ego_collision", False)
                lap_time = info.get("ego_lap_time", 0)
                steps = info.get("step", 0)

                self.episode_progresses.append(progress)
                self.episode_collisions.append(float(collision))
                self.episode_lengths.append(steps)
                if speed is not None:
                    self.episode_max_speeds.append(max(list(self.episode_speeds)[-steps:]) if steps > 0 else 0)
                if lap_time > 0:
                    self.episode_lap_times.append(lap_time)

        # Log every 5k steps
        if self.n_calls % 5000 == 0 and self.n_calls > 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Compute and log racing metrics."""
        metrics = {}

        if self.episode_speeds:
            recent_speeds = list(self.episode_speeds)[-50000:]
            metrics["racing/avg_speed"] = np.mean(recent_speeds)
            metrics["racing/max_speed"] = np.max(recent_speeds)

        if self.episode_progresses:
            recent = list(self.episode_progresses)[-100:]
            metrics["racing/avg_progress"] = np.mean(recent)
            metrics["racing/max_progress"] = np.max(recent)
            metrics["racing/median_progress"] = np.median(recent)

        if self.episode_collisions:
            recent = list(self.episode_collisions)[-100:]
            metrics["racing/collision_rate"] = np.mean(recent)
            metrics["racing/survival_rate"] = 1.0 - np.mean(recent)

        if self.episode_lap_times:
            recent = list(self.episode_lap_times)[-20:]
            metrics["racing/avg_lap_time"] = np.mean(recent)
            metrics["racing/best_lap_time"] = np.min(recent)

        if self.episode_lengths:
            recent = list(self.episode_lengths)[-100:]
            metrics["racing/avg_episode_length"] = np.mean(recent)

        # Log to SB3 logger (TensorBoard)
        for key, value in metrics.items():
            self.logger.record(key, value)

        # Log to WandB
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.num_timesteps)
            except Exception:
                pass


class WandbSafeCallback(BaseCallback):
    """Lightweight WandB callback that syncs TensorBoard logs."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        try:
            import wandb
            wandb.run.summary["total_timesteps"] = self.num_timesteps
        except Exception:
            pass


class CurriculumDRCallback(BaseCallback):
    """
    Logs domain randomization curriculum strength to TensorBoard/WandB.

    The DR wrapper self-tracks its own steps inside each subprocess.
    This callback computes the expected strength from training progress
    (same math, guaranteed to match).
    """

    def __init__(self, total_timesteps: int, use_wandb: bool = False,
                 warmup: float = 0.2, full: float = 0.6, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.use_wandb = use_wandb
        self.warmup = warmup
        self.full = full

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            progress = self.num_timesteps / self.total_timesteps

            if progress < self.warmup:
                strength = 0.0
            elif progress > self.full:
                strength = 1.0
            else:
                strength = (progress - self.warmup) / (self.full - self.warmup)

            self.logger.record("dr/strength", round(strength, 3))
            self.logger.record("dr/progress", round(progress, 3))

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({"dr/strength": strength}, step=self.num_timesteps)
                except Exception:
                    pass

        return True


class SelfPlayCallback(BaseCallback):
    """
    Self-play: periodically saves ego policy to a shared file so
    opponent subprocesses can load it on reset.

    Works with SubprocVecEnv by avoiding cross-process calls.
    Each F1TenthWrapper checks for updated opponent weights on reset().
    """

    def __init__(self, update_freq: int = 50000, use_wandb: bool = False,
                 save_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq  # Steps between updates
        self.use_wandb = use_wandb
        self.save_path = save_path or os.path.join("runs", ".selfplay_opponent.zip")
        self.update_count = 0
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            self.update_count += 1

            # Save ego policy — opponents load this on reset
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)

            self.logger.record("self_play/update_count", self.update_count)
            if self.verbose:
                print(f"[SelfPlay] Opponent updated (#{self.update_count}) at step {self.num_timesteps}")

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({"self_play/update_count": self.update_count},
                              step=self.num_timesteps)
                except Exception:
                    pass

        return True

    def _on_training_end(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
