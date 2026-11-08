"""Resumable compact Halton sweep for fixed-volume vertebra classifiers."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from skp.search import generate_sequence


BACKBONES = ("x3d_s", "x3d_l", "csn_r26")
SHAPES = ((48, 320, 320), (64, 288, 288), (80, 256, 256))
BATCH_SIZES = (2, 4)
EPOCHS = (8, 12, 16)
FEATURE_DIMS = {"x3d_s": 432, "x3d_l": 432, "csn_r26": 2048}
# One anchor plus these five deterministic Halton points provides coverage of
# every backbone, shape, batch size, and training horizon in only six runs.
SELECTED_HALTON_INDICES = (6, 1, 3, 11, 35)


@dataclass(frozen=True)
class Trial:
    halton_index: int
    backbone: str
    shape: tuple[int, int, int]
    batch_size: int
    num_epochs: int
    lr: float

    @property
    def trial_id(self) -> str:
        depth, size, _ = self.shape
        backbone = self.backbone.replace("_", "")
        return (
            f"h{self.halton_index:02d}-{backbone}-d{depth}-r{size}-"
            f"bs{self.batch_size}-e{self.num_epochs}-lr{self.lr:.2e}"
        )


def _pick(values: tuple, point: float):
    return values[min(int(point * len(values)), len(values) - 1)]


def generate_trials() -> list[Trial]:
    points = generate_sequence(
        max(SELECTED_HALTON_INDICES) + 1,
        num_dims=5,
        skip=100,
        per_dim_shift=True,
        shuffle_sequence=False,
    )
    trials = []
    for index in SELECTED_HALTON_INDICES:
        point = points[index]
        trials.append(
            Trial(
                halton_index=index,
                backbone=_pick(BACKBONES, point[0]),
                shape=_pick(SHAPES, point[1]),
                batch_size=_pick(BATCH_SIZES, point[2]),
                num_epochs=_pick(EPOCHS, point[3]),
                lr=1e-4 * math.exp(point[4] * math.log(4.0)),
            )
        )
    return trials


def parse_auc_trajectory(log_path: Path) -> dict[str, float]:
    """Read completed epoch macro AUCs, excluding Lightning's sanity check."""
    if not log_path.is_file():
        return {}
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log_path.read_text(errors="replace"))
    values = [float(value) for value in re.findall(r"auc_mean\s*\|\s*([0-9.]+)", text)]
    if values:
        values = values[1:]
    return {str(epoch): value for epoch, value in enumerate(values, start=1)}


def _atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _training_command(
    trial: Trial,
    *,
    run_id: str,
    pruning_json: Path | None,
    dry_run: bool,
) -> list[str]:
    depth, height, width = trial.shape
    command = [
        sys.executable,
        "-m",
        "skp.train",
        "cspine_vertebra_3d_halton",
        "--run_id",
        run_id,
        "--devices",
        "2",
        "--strategy",
        "ddp",
        "--precision",
        "bf16-mixed",
        "--log_every_n_steps",
        "10",
        "--backbone",
        trial.backbone,
        "--x3d_feature_dim",
        str(FEATURE_DIMS[trial.backbone]),
        "--runtime_output_shape",
        str(trial.shape),
        "--num_slices",
        str(depth),
        "--batch_size",
        str(trial.batch_size),
        "--num_epochs",
        str(trial.num_epochs),
        "--scheduler_num_epochs",
        str(trial.num_epochs),
        "--optimizer_params.lr",
        str(trial.lr),
        "--scheduler_params.pct_start",
        str(1.0 / trial.num_epochs),
        "--notes",
        (
            f"Compact Halton trial {trial.halton_index}; fixed shape "
            f"{depth}x{height}x{width}; global 3D pooling; temporal stride 1."
        ),
    ]
    if pruning_json is not None:
        command.extend(["--halton_pruning", str(pruning_json)])
    if dry_run:
        command.extend(
            [
                "--dry_run",
                "--limit_train_batches",
                "2",
                "--limit_val_batches",
                "2",
            ]
        )
    return command


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


def _is_oom(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(errors="replace").lower()
    return "out of memory" in text or "cuda error: memory allocation" in text


def run_sweep(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trials = generate_trials()
    plan = {
        "anchor": args.anchor_run_id,
        "anchor_log": str(args.anchor_log.resolve()),
        "halton_skip": 100,
        "selected_halton_indices": list(SELECTED_HALTON_INDICES),
        "trials": [asdict(trial) | {"trial_id": trial.trial_id} for trial in trials],
        "pruning_margins": {"4": 0.05, "8": 0.03},
    }
    _atomic_json(root / "plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    status_path = root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if args.wait_pid and _pid_alive(args.wait_pid):
        print(f"Waiting for anchor PID {args.wait_pid} to finish.", flush=True)
        while _pid_alive(args.wait_pid):
            time.sleep(30)

    anchor_trajectory = parse_auc_trajectory(args.anchor_log)
    if not anchor_trajectory:
        raise RuntimeError(f"No completed anchor epochs found in {args.anchor_log}.")
    pruning_state = {
        "best_by_epoch": anchor_trajectory,
        "best_trial_by_epoch": {
            epoch: args.anchor_run_id for epoch in anchor_trajectory
        },
        "observations": [],
    }
    pruning_state_path = root / "pruning_state.json"
    if not pruning_state_path.is_file():
        _atomic_json(pruning_state_path, pruning_state)

    for original_trial in trials:
        if status.get(original_trial.trial_id, {}).get("status") in {
            "completed",
            "pruned",
        }:
            continue
        trial = original_trial
        run_id = f"cspine-halton-{trial.trial_id}-v1"
        smoke_log = root / "logs" / f"{trial.trial_id}-smoke.log"
        smoke_rc = _run_command(
            _training_command(trial, run_id=f"smoke-{trial.trial_id}", pruning_json=None, dry_run=True),
            smoke_log,
        )
        if smoke_rc != 0 and _is_oom(smoke_log) and trial.batch_size == 4:
            trial = replace(trial, batch_size=2)
            run_id += "-bs2fallback"
            fallback_log = root / "logs" / f"{original_trial.trial_id}-smoke-bs2.log"
            smoke_rc = _run_command(
                _training_command(trial, run_id=f"smoke-{trial.trial_id}", pruning_json=None, dry_run=True),
                fallback_log,
            )
        if smoke_rc != 0:
            status[original_trial.trial_id] = {
                "status": "rejected_smoke",
                "return_code": smoke_rc,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(status_path, status)
            continue

        pruning_config = {
            "state_path": str(pruning_state_path),
            "trial_id": run_id,
            "margins": {"4": 0.05, "8": 0.03},
            "metric": "val_metric",
            "mode": "max",
        }
        pruning_json = root / "trial_configs" / f"{run_id}-pruning.json"
        _atomic_json(pruning_json, pruning_config)
        status[original_trial.trial_id] = {
            "status": "running",
            "effective_trial": asdict(trial),
            "run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(status_path, status)
        train_log = root / "logs" / f"{run_id}.log"
        return_code = _run_command(
            _training_command(trial, run_id=run_id, pruning_json=pruning_json, dry_run=False),
            train_log,
        )
        observations = json.loads(pruning_state_path.read_text()).get("observations", [])
        pruned = any(
            observation.get("trial_id") == run_id and observation.get("pruned")
            for observation in observations
        )
        status[original_trial.trial_id] = {
            "status": "pruned" if pruned else ("completed" if return_code == 0 else "failed"),
            "effective_trial": asdict(trial),
            "run_id": run_id,
            "return_code": return_code,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(status_path, status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-log", type=Path, required=True)
    parser.add_argument("--anchor-run-id", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
