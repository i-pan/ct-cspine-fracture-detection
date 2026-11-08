from pathlib import Path

from skp.cspine.halton_sweep import generate_trials, parse_auc_trajectory


def test_compact_halton_plan_covers_search_dimensions():
    trials = generate_trials()
    assert len(trials) == 5
    assert {trial.backbone for trial in trials} == {"x3d_s", "x3d_l", "csn_r26"}
    assert {trial.shape for trial in trials} == {
        (48, 320, 320),
        (64, 288, 288),
        (80, 256, 256),
    }
    assert {trial.batch_size for trial in trials} == {2, 4}
    assert {trial.num_epochs for trial in trials} == {8, 12, 16}
    assert all(1e-4 <= trial.lr <= 4e-4 for trial in trials)


def test_parse_auc_trajectory_ignores_sanity_check(tmp_path: Path):
    log = tmp_path / "training.log"
    log.write_text(
        "Sanity Checking\nauc_mean | 0.5000\n"
        "epoch zero\nauc_mean   | 0.7000\n"
        "epoch one\nauc_mean | 0.8000\n"
    )
    assert parse_auc_trajectory(log) == {"1": 0.7, "2": 0.8}
