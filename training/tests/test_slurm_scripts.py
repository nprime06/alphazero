"""Regression tests for Slurm submit wrappers.

These tests never contact Slurm. They put fake ``sbatch``/``squeue`` commands
first on ``PATH`` and verify the wrappers fail or write status as expected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "training" / "scripts"


def _run_script(
    args: list[str],
    *,
    tmp_path: Path,
    fake_sbatch: str = "123456.mock\n",
    fake_squeue: str = "",
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    sbatch = fake_bin / "sbatch"
    squeue = fake_bin / "squeue"
    sbatch.write_text(f"#!/bin/bash\nprintf '%s' {fake_sbatch!r}\n")
    squeue.write_text(f"#!/bin/bash\nprintf '%s' {fake_squeue!r}\n")
    sbatch.chmod(0o755)
    squeue.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ensure_fake_selfplay() -> None:
    binary = PROJECT_ROOT / "target" / "release" / "self-play"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        binary.write_text("#!/bin/bash\nexit 0\n")
        binary.chmod(0o755)


def _cleanup_fake_selfplay() -> None:
    binary = PROJECT_ROOT / "target" / "release" / "self-play"
    if binary.exists() and binary.read_text(errors="ignore") == "#!/bin/bash\nexit 0\n":
        binary.unlink()
    for path in [PROJECT_ROOT / "target" / "release", PROJECT_ROOT / "target"]:
        try:
            path.rmdir()
        except OSError:
            pass


def _restore_status_file(tmp_path: Path):
    status_dir = PROJECT_ROOT / "runs" / "slurm_setup"
    status_file = status_dir / "latest_orcd_jobs.txt"
    backup = tmp_path / "latest_orcd_jobs.backup"
    had_status = status_file.exists()
    if had_status:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(status_file, backup)

    def restore() -> None:
        if had_status:
            status_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(backup, status_file)
        else:
            status_file.unlink(missing_ok=True)
            try:
                status_dir.rmdir()
            except OSError:
                pass

    return status_file, restore


def test_cluster_smoke_preflight_rejects_run_dir(tmp_path):
    result = _run_script(
        [
            "bash",
            str(SCRIPTS / "submit_cluster_smoke.sh"),
            "--run-dir",
            str(tmp_path / "bad"),
        ],
        tmp_path=tmp_path,
    )

    assert result.returncode == 2
    assert "always starts a fresh run" in result.stderr


def test_cluster_smoke_preflight_rejects_existing_alphazero_job(tmp_path):
    _ensure_fake_selfplay()
    try:
        result = _run_script(
            [
                "bash",
                str(SCRIPTS / "submit_cluster_smoke.sh"),
                "--preflight-only",
            ],
            tmp_path=tmp_path,
            fake_squeue="123|az-coord|RUNNING\n",
        )
    finally:
        _cleanup_fake_selfplay()

    assert result.returncode == 2
    assert "existing AlphaZero Slurm jobs" in result.stderr


def test_cluster_smoke_preflight_rejects_missing_selfplay_binary(tmp_path):
    binary = PROJECT_ROOT / "target" / "release" / "self-play"
    if binary.exists():
        pytest.skip("real self-play binary exists")
    result = _run_script(
        [
            "bash",
            str(SCRIPTS / "submit_cluster_smoke.sh"),
            "--preflight-only",
        ],
        tmp_path=tmp_path,
    )

    assert result.returncode == 2
    assert "cargo build --release -p self-play" in result.stderr


def test_cluster_smoke_submit_records_status(tmp_path):
    status_file, restore_status = _restore_status_file(tmp_path)
    _ensure_fake_selfplay()
    try:
        result = _run_script(
            [
                "bash",
                str(SCRIPTS / "submit_cluster_smoke.sh"),
                "--time",
                "0:30:00",
            ],
            tmp_path=tmp_path,
            fake_sbatch="424242.mock\n",
        )
        status_text = status_file.read_text()
    finally:
        restore_status()
        _cleanup_fake_selfplay()
        for line in status_text.splitlines() if "status_text" in locals() else []:
            if line.startswith("cluster_pilot_run_dir="):
                run_dir = Path(line.split("=", 1)[1])
                if run_dir.is_dir() and not any(run_dir.iterdir()):
                    run_dir.rmdir()
                break

    assert result.returncode == 0
    assert "preflight ok" in result.stdout
    assert "submitted job 424242.mock" in result.stdout
    assert "cluster_pilot_job=424242.mock" in status_text
    assert "cluster_pilot_gpus=1" in status_text
    assert "cluster_pilot_time=0:30:00" in status_text


def test_train_and_selfplay_wrappers_enforce_orcd_limits(tmp_path):
    cases = [
        (
            [
                "bash",
                str(SCRIPTS / "submit_train.sh"),
                "--time",
                "7:00:00",
                "--dummy-data",
                "--network",
                "tiny",
                "--steps",
                "1",
            ],
            "training wall time must be <= 6:00:00",
        ),
        (
            [
                "bash",
                str(SCRIPTS / "submit_selfplay.sh"),
                "--gpus",
                "2",
                "--model",
                "model.pt",
                "--games",
                "1",
                "--output",
                str(tmp_path / "games"),
            ],
            "self-play currently uses one GPU",
        ),
        (
            [
                "bash",
                str(SCRIPTS / "submit_selfplay.sh"),
                "--time",
                "7:00:00",
                "--model",
                "model.pt",
                "--games",
                "1",
                "--output",
                str(tmp_path / "games"),
            ],
            "self-play wall time must be <= 6:00:00",
        ),
    ]

    for args, expected in cases:
        result = _run_script(args, tmp_path=tmp_path)
        assert result.returncode == 2
        assert expected in result.stderr
