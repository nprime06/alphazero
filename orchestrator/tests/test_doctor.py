"""Tests for coordinator run integrity checks."""

from __future__ import annotations

import json

from orchestrator.coordinator import PipelineState
from orchestrator.doctor import inspect_run


def _write_state(run_dir, **overrides):
    state = PipelineState(**{
        "iteration": 3,
        "best_model_version": 2,
        "total_games": 2,
        "total_train_steps": 100,
        **overrides,
    })
    state.save(str(run_dir / "pipeline_state.yaml"))
    return state


def _write_weight(run_dir, version: int) -> None:
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / f"model_v{version:06d}.pt").write_bytes(b"model")


def _write_ledger(run_dir, candidate_version: int, decision: str = "promote") -> None:
    (run_dir / "promotion_ledger.jsonl").write_text(
        json.dumps(
            {
                "iteration": 3,
                "candidate_version": candidate_version,
                "incumbent_version": 1,
                "decision": decision,
                "reason": "test",
            }
        )
        + "\n"
    )


def test_inspect_run_accepts_auditable_state(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "data" / "game_1.msgpack").write_bytes(b"game")
    (run_dir / "data" / "game_2.msgpack").write_bytes(b"game")
    (run_dir / "checkpoints" / "checkpoint.pt").write_bytes(b"ckpt")
    _write_state(run_dir)
    _write_weight(run_dir, 1)
    _write_weight(run_dir, 2)
    (run_dir / "weights" / "latest.txt").write_text("2")
    _write_ledger(run_dir, 2)

    report = inspect_run(run_dir)

    assert report.ok is True
    assert report.latest_version == 2
    assert report.weight_versions == [1, 2]
    assert report.game_files == 2
    assert report.checkpoint_files == 1
    assert report.ledger_entries == 1
    assert report.issues == []


def test_inspect_run_flags_missing_best_weight(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    _write_state(run_dir, best_model_version=20, total_games=0)
    _write_weight(run_dir, 31)
    (run_dir / "weights" / "latest.txt").write_text("31")

    report = inspect_run(run_dir)

    assert report.ok is False
    assert {issue.code for issue in report.issues} >= {
        "BEST_WEIGHT_MISSING",
        "PROMOTION_LEDGER_MISSING",
    }


def test_inspect_run_requires_promotion_ledger_for_non_initial_best(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    _write_state(run_dir, total_games=0)
    _write_weight(run_dir, 2)
    (run_dir / "weights" / "latest.txt").write_text("2")

    report = inspect_run(run_dir)

    assert report.ok is False
    assert any(issue.code == "PROMOTION_LEDGER_MISSING" for issue in report.issues)


def test_inspect_run_flags_ledger_state_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    _write_state(run_dir, best_model_version=3, total_games=0)
    _write_weight(run_dir, 2)
    _write_weight(run_dir, 3)
    (run_dir / "weights" / "latest.txt").write_text("3")
    _write_ledger(run_dir, 2)

    report = inspect_run(run_dir)

    assert report.ok is False
    assert any(issue.code == "PROMOTION_LEDGER_MISMATCH" for issue in report.issues)


def test_inspect_run_flags_latest_weight_missing(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    _write_state(run_dir, best_model_version=1, iteration=1, total_games=0)
    _write_weight(run_dir, 1)
    (run_dir / "weights" / "latest.txt").write_text("2")

    report = inspect_run(run_dir)

    assert report.ok is False
    assert any(issue.code == "LATEST_WEIGHT_MISSING" for issue in report.issues)
