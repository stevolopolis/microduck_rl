"""FROZEN RULER for the outerloop ``microduck-jump`` benchmark. NOT author-editable.

Trains a policy under the committed jump config at a reduced proxy budget, then
scores a VALIDITY-GATED apex height over K seeded eval episodes. The metric is
read from physics state only (heights / contacts / velocities) — never from
reward terms — so the author cannot move the number by reshaping rewards. The
eval env's init and the validity gates are controlled HERE, not by the
(author-editable) task cfg.

Output (stdout, one JSON line): ``{"apex_height_m": <median>, ...diagnostics}``.
Only ``apex_height_m`` is the scored metric. The paired seed comes from
``$MD_JUMP_SEED`` (injected by the orchestrator; baseline and candidate share it
— common random numbers). Invoked by outerloop as:

    uv run python -m mjlab_microduck.ol_eval_jump --task Mjlab-Jump-Flat-MicroDuck \
        --iters 1000 --envs 4096 --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import torch

STAND_TILT_DEG = 15.0      # gate (1): upright-at-t0 tolerance
LAND_TILT_DEG = 30.0       # gate (4): upright-at-settle tolerance
VZ_ARTIFACT_MAX = 5.0      # gate (3): |vz| bound for a 25 cm robot (m/s)
PENETRATION_Z = -0.02      # gate (3): trunk-below-floor pop (m)
MIN_FLIGHT_S = 0.05        # gate (2): shortest window that counts as flight (s)
STAND_WINDOW_S = 0.2       # gate (1): stand-check window at episode start (s)
SETTLE_WINDOW_S = 1.0      # gate (4): settle window at episode end (s)


# ── Pure scoring core (unit-testable without a GPU / env) ─────────────────────

def apex_of_episode(
    z: torch.Tensor,          # (T,) trunk height above terrain, per step
    vz: torch.Tensor,         # (T,) vertical CoM velocity
    cos_tilt: torch.Tensor,   # (T,) 1 - 2(qx^2+qy^2); 1.0 = upright
    airborne: torch.Tensor,   # (T,) bool: no robot geom touches terrain
    feet_off: torch.Tensor,   # (T,) bool: both feet off the ground
    finite: torch.Tensor,     # (T,) bool: root state finite this step
    stand_z: float,
    step_dt: float,
) -> float | None:
    """Validity-gated apex height ABOVE standing for one episode, or None if the
    episode is not a real, upright, artifact-free jump that lands and survives."""
    T = z.shape[0]
    w0 = max(1, int(STAND_WINDOW_S / step_dt))
    settle = max(1, int(SETTLE_WINDOW_S / step_dt))
    min_flight = max(2, int(MIN_FLIGHT_S / step_dt))

    # (1) stable stand at t=0: upright and near standing height (bounded overshoot).
    if not (cos_tilt[:w0].min().item() > math.cos(math.radians(STAND_TILT_DEG))
            and z[:w0].max().item() < stand_z + 0.02):
        return None

    # (2)+(3) a real, artifact-free flight window.
    flight = airborne & finite & (vz.abs() < VZ_ARTIFACT_MAX) & (z > PENETRATION_Z)
    idx = flight.nonzero().flatten()
    if idx.numel() < min_flight:
        return None
    onset = idx[0].item()
    if not (vz[onset].item() > 0.0 and bool(feet_off[onset].item())):
        return None  # left the ground going UP off both feet (a jump, not a fall)

    # (4) landed and survived to the settle window.
    end = idx[-1].item()
    if end >= T - settle:
        return None
    relanded = (~airborne[end + 1:]).float().mean().item() > 0.8
    upright_settled = cos_tilt[T - settle:].min().item() > math.cos(math.radians(LAND_TILT_DEG))
    height_settled = z[T - settle:].min().item() > stand_z - 0.03
    if not (relanded and upright_settled and height_settled):
        return None

    return max(0.0, z[idx].max().item() - stand_z)


def aggregate(apexes: list[float | None]) -> dict:
    """Median over episodes (invalid → 0.0). Bimodal jumps-vs-falls → median."""
    scored = [a if a is not None else 0.0 for a in apexes]
    valid = sum(1 for a in apexes if a is not None)
    med = float(torch.tensor(scored).median().item()) if scored else 0.0
    return {
        "apex_height_m": round(med, 5),
        "jump_success_rate": round(valid / len(apexes), 3) if apexes else 0.0,
        "n_valid_episodes": valid,
        "n_episodes": len(apexes),
    }


# ── State reads (match src/mjlab_microduck/tasks/mdp.py conventions) ───────────

def _no_ground_contact(raw_env, sensor_name: str) -> torch.Tensor:
    found = raw_env.scene.sensors[sensor_name].data.found.flatten(start_dim=1)
    return found.sum(dim=1) == 0


def _both_feet_off(raw_env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    found = raw_env.scene.sensors[sensor_name].data.found.flatten(start_dim=1)
    return found.sum(dim=1) == 0


def _trunk_z(raw_env, robot) -> torch.Tensor:
    return torch.nan_to_num(
        robot.data.root_link_pos_w[:, 2] - raw_env.scene.terrain.env_origins[:, 2], nan=0.0
    )


def _cos_tilt(robot) -> torch.Tensor:
    q = robot.data.root_link_quat_w  # [w, x, y, z]
    return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)


# ── Train + rollout (mjlab / rsl_rl API mirrors export.py) ─────────────────────

def _seed_everything(seed: int, env_cfg) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # B4-CONFIRM: how mjlab 1.3.0 wants the env seed set (env_cfg.seed vs a util).
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed


def train_policy(task_id: str, iters: int, envs: int, seed: int, device: str, log_dir: str):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.runners import OnPolicyRunner

    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=False)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = envs
    _seed_everything(seed, env_cfg)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task_id) or OnPolicyRunner)(
        env, asdict(agent_cfg), log_dir=log_dir, device=device
    )
    # B4-CONFIRM: rsl_rl 1.3.0 learn() signature (num_learning_iterations kw).
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    return runner, agent_cfg


@torch.no_grad()
def score(task_id: str, runner, agent_cfg, episodes: int, horizon_s: float, device: str) -> dict:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    # Ruler-owned eval env: FORCE a stand init so a tampered training reset can't
    # fake the "started from a stable stand" gate. One env per scored episode.
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = episodes
    ev = env_cfg.events["set_ground_state"].params  # B4-CONFIRM event/param names
    ev.update({"standing_prob": 1.0, "sitting_prob": 0.0,
               "face_down_prob": 0.0, "face_up_prob": 0.0})
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    robot = raw.scene["robot"]
    policy = runner.get_inference_policy(device=device)

    T = int(horizon_s / raw.step_dt)
    z = torch.zeros(T, episodes, device=device)
    vz = torch.zeros_like(z)
    cos = torch.zeros_like(z)
    air = torch.zeros(T, episodes, dtype=torch.bool, device=device)
    feet = torch.zeros_like(air)
    finite = torch.ones_like(air)

    obs, _ = env.reset()
    stand_z = float(torch.median(_trunk_z(raw, robot)).item())  # measured, not guessed
    for t in range(T):
        obs, _, _, _ = env.step(policy(obs))
        z[t] = _trunk_z(raw, robot)
        vz[t] = torch.nan_to_num(robot.data.root_link_lin_vel_w[:, 2], nan=0.0)
        cos[t] = _cos_tilt(robot)
        air[t] = _no_ground_contact(raw, "robot_ground_contact")
        feet[t] = _both_feet_off(raw)
        finite[t] = (torch.isfinite(robot.data.root_link_lin_vel_w).all(dim=1)
                     & torch.isfinite(robot.data.root_link_quat_w).all(dim=1))
    env.close()

    apexes = [
        apex_of_episode(z[:, n], vz[:, n], cos[:, n], air[:, n], feet[:, n], finite[:, n],
                        stand_z, raw.step_dt)
        for n in range(episodes)
    ]
    out = aggregate(apexes)
    out["standing_trunk_z"] = round(stand_z, 4)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-Jump-Flat-MicroDuck")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--eval-horizon-s", type=float, default=4.0)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    import mjlab_microduck.tasks  # noqa: F401  (populate the registry)

    seed = int(os.environ.get("MD_JUMP_SEED", "0")) or random.randint(1, 2**31 - 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as log_dir:
        runner, agent_cfg = train_policy(a.task, a.iters, a.envs, seed, device, log_dir)
        out = score(a.task, runner, agent_cfg, a.eval_episodes, a.eval_horizon_s, device)
    out["run_seed"] = seed
    out["proxy_iters"] = a.iters
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
