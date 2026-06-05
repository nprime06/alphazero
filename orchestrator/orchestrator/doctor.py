"""Run integrity checks for AlphaZero coordinator campaigns."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from orchestrator.coordinator import PipelineState


_MODEL_RE = re.compile(r"^model_v(\d{6})\.pt$")


@dataclass(frozen=True)
class DoctorIssue:
    """A single run-integrity finding."""

    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass
class RunDoctorReport:
    """Structured result from inspecting a run directory."""

    run_dir: str
    state: dict[str, int] | None
    latest_version: int
    weight_versions: list[int]
    game_files: int
    checkpoint_files: int
    ledger_entries: int
    issues: list[DoctorIssue]

    @property
    def ok(self) -> bool:
        """Whether the run is safe to use as a trusted campaign state."""
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _read_latest_version(latest_file: Path, issues: list[DoctorIssue]) -> int:
    if not latest_file.exists():
        return 0
    text = latest_file.read_text().strip()
    if not text:
        issues.append(
            DoctorIssue(
                "ERROR",
                "LATEST_EMPTY",
                "latest.txt exists but does not contain a version.",
                str(latest_file),
            )
        )
        return 0
    try:
        version = int(text)
    except ValueError:
        issues.append(
            DoctorIssue(
                "ERROR",
                "LATEST_INVALID",
                f"latest.txt contains a non-integer version: {text!r}.",
                str(latest_file),
            )
        )
        return 0
    if version < 0:
        issues.append(
            DoctorIssue(
                "ERROR",
                "LATEST_INVALID",
                f"latest.txt contains a negative version: {version}.",
                str(latest_file),
            )
        )
        return 0
    return version


def _weight_versions(weights_dir: Path, issues: list[DoctorIssue]) -> list[int]:
    versions: list[int] = []
    if not weights_dir.exists():
        return versions
    for path in sorted(weights_dir.glob("model_v*.pt")):
        match = _MODEL_RE.match(path.name)
        if match is None:
            issues.append(
                DoctorIssue(
                    "WARNING",
                    "WEIGHT_NAME_UNRECOGNIZED",
                    f"Ignoring weight file with unexpected name: {path.name}.",
                    str(path),
                )
            )
            continue
        versions.append(int(match.group(1)))
    return versions


def _read_ledger(path: Path, issues: list[DoctorIssue]) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "PROMOTION_LEDGER_INVALID",
                    f"promotion_ledger.jsonl line {line_no} is invalid JSON: {exc}.",
                    str(path),
                )
            )
            continue
        if not isinstance(entry, dict):
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "PROMOTION_LEDGER_INVALID",
                    f"promotion_ledger.jsonl line {line_no} is not an object.",
                    str(path),
                )
            )
            continue
        entries.append(entry)
    return entries


def _ledger_best_version(entries: list[dict[str, Any]]) -> int | None:
    best: int | None = None
    for entry in entries:
        if entry.get("decision") != "promote":
            continue
        candidate = entry.get("candidate_version")
        if isinstance(candidate, int) and candidate > 0:
            best = candidate
    return best


def inspect_run(run_dir: str | Path) -> RunDoctorReport:
    """Inspect a coordinator run directory for resumability hazards."""
    root = Path(run_dir)
    issues: list[DoctorIssue] = []

    if not root.exists():
        issues.append(
            DoctorIssue(
                "ERROR",
                "RUN_DIR_MISSING",
                f"Run directory does not exist: {root}.",
                str(root),
            )
        )
        return RunDoctorReport(
            run_dir=str(root),
            state=None,
            latest_version=0,
            weight_versions=[],
            game_files=0,
            checkpoint_files=0,
            ledger_entries=0,
            issues=issues,
        )
    if not root.is_dir():
        issues.append(
            DoctorIssue(
                "ERROR",
                "RUN_DIR_NOT_DIRECTORY",
                f"Run path is not a directory: {root}.",
                str(root),
            )
        )

    state_path = root / "pipeline_state.yaml"
    state: PipelineState | None = None
    if state_path.exists():
        try:
            state = PipelineState(**{
                key: value
                for key, value in _read_yaml(state_path).items()
                if key in PipelineState.__dataclass_fields__
            })
        except Exception as exc:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "STATE_INVALID",
                    f"Could not read pipeline_state.yaml: {exc}.",
                    str(state_path),
                )
            )
    else:
        issues.append(
            DoctorIssue(
                "ERROR",
                "STATE_MISSING",
                "pipeline_state.yaml is required to resume a run.",
                str(state_path),
            )
        )

    data_dir = root / "data"
    weights_dir = root / "weights"
    checkpoint_dir = root / "checkpoints"
    ledger_path = root / "promotion_ledger.jsonl"

    if not data_dir.exists():
        issues.append(
            DoctorIssue("WARNING", "DATA_DIR_MISSING", "data/ is missing.", str(data_dir))
        )
    if not weights_dir.exists():
        issues.append(
            DoctorIssue(
                "ERROR",
                "WEIGHTS_DIR_MISSING",
                "weights/ is required to resume self-play/evaluation.",
                str(weights_dir),
            )
        )
    if not checkpoint_dir.exists():
        issues.append(
            DoctorIssue(
                "WARNING",
                "CHECKPOINT_DIR_MISSING",
                "checkpoints/ is missing.",
                str(checkpoint_dir),
            )
        )

    latest_version = _read_latest_version(weights_dir / "latest.txt", issues)
    weight_versions = _weight_versions(weights_dir, issues)
    weight_set = set(weight_versions)
    game_files = len(list(data_dir.glob("**/*.msgpack"))) if data_dir.exists() else 0
    checkpoint_files = (
        len([path for path in checkpoint_dir.iterdir() if path.is_file()])
        if checkpoint_dir.exists()
        else 0
    )
    ledger_entries = _read_ledger(ledger_path, issues)

    if latest_version > 0 and latest_version not in weight_set:
        issues.append(
            DoctorIssue(
                "ERROR",
                "LATEST_WEIGHT_MISSING",
                f"latest.txt points to v{latest_version}, but that weight file is missing.",
                str(weights_dir / f"model_v{latest_version:06d}.pt"),
            )
        )

    if state is not None:
        if state.best_model_version > 0 and state.best_model_version not in weight_set:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "BEST_WEIGHT_MISSING",
                    "pipeline_state.yaml points to best model "
                    f"v{state.best_model_version}, but that weight file is missing.",
                    str(weights_dir / f"model_v{state.best_model_version:06d}.pt"),
                )
            )
        if latest_version and state.best_model_version > latest_version:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "BEST_NEWER_THAN_LATEST",
                    f"best model v{state.best_model_version} is newer than latest.txt v{latest_version}.",
                    str(state_path),
                )
            )
        if state.total_games != game_files:
            issues.append(
                DoctorIssue(
                    "WARNING",
                    "GAME_COUNT_MISMATCH",
                    f"state total_games={state.total_games}, but data/ contains {game_files} msgpack files.",
                    str(data_dir),
                )
            )
        if (
            latest_version > state.best_model_version > 0
            and latest_version in weight_set
        ):
            issues.append(
                DoctorIssue(
                    "INFO",
                    "UNEVALUATED_LATEST_WEIGHT",
                    f"latest weight v{latest_version} is newer than best v{state.best_model_version}.",
                    str(weights_dir / f"model_v{latest_version:06d}.pt"),
                )
            )
        if state.iteration > 0 and state.best_model_version > 1 and not ledger_entries:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "PROMOTION_LEDGER_MISSING",
                    "promotion_ledger.jsonl is missing, so best-model lineage cannot be proven.",
                    str(ledger_path),
                )
            )

    ledger_best = _ledger_best_version(ledger_entries)
    if state is not None and ledger_best is not None:
        if ledger_best != state.best_model_version:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "PROMOTION_LEDGER_MISMATCH",
                    f"ledger last promoted v{ledger_best}, but state best is v{state.best_model_version}.",
                    str(ledger_path),
                )
            )
        if ledger_best not in weight_set:
            issues.append(
                DoctorIssue(
                    "ERROR",
                    "PROMOTED_WEIGHT_MISSING",
                    f"ledger last promoted v{ledger_best}, but that weight file is missing.",
                    str(weights_dir / f"model_v{ledger_best:06d}.pt"),
                )
            )

    return RunDoctorReport(
        run_dir=str(root),
        state=asdict(state) if state is not None else None,
        latest_version=latest_version,
        weight_versions=weight_versions,
        game_files=game_files,
        checkpoint_files=checkpoint_files,
        ledger_entries=len(ledger_entries),
        issues=issues,
    )


def format_report(report: RunDoctorReport) -> str:
    """Format a run doctor report for humans."""
    lines = [
        f"Run: {report.run_dir}",
        f"Status: {'OK' if report.ok else 'FAILED'}",
        f"Latest weight: v{report.latest_version}" if report.latest_version else "Latest weight: none",
        f"Weights: {len(report.weight_versions)} files",
        f"Games: {report.game_files}",
        f"Checkpoints: {report.checkpoint_files}",
        f"Promotion ledger entries: {report.ledger_entries}",
    ]
    if report.state is not None:
        lines.append(
            "State: iteration={iteration}, best=v{best_model_version}, "
            "total_games={total_games}, train_steps={total_train_steps}".format(
                **report.state
            )
        )
    if report.issues:
        lines.append("Issues:")
        for issue in report.issues:
            suffix = f" ({issue.path})" if issue.path else ""
            lines.append(
                f"  [{issue.severity}] {issue.code}: {issue.message}{suffix}"
            )
    else:
        lines.append("Issues: none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m orchestrator.doctor``."""
    parser = argparse.ArgumentParser(description="Inspect an AlphaZero run directory")
    parser.add_argument("--run-dir", required=True, help="Coordinator run directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = inspect_run(args.run_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
