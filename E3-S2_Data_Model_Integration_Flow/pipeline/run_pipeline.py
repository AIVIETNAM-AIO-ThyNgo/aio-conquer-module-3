"""run_pipeline.py -- single entry point for the full SPY modeling pipeline.

Execution path (one command):

    python -m pipeline.run_pipeline            # run all stages
    python -m pipeline.run_pipeline baseline   # run a single stage
    python -m pipeline.run_pipeline --force    # regenerate all outputs

Stages, in order:
    1. data_foundation  -> raw CSV, provenance, canonical CSV, manifest, data dictionary
    2. baseline         -> baseline_zero_oos_predictions.csv, fold_metrics, summary
    3. lightgbm         -> lightgbm_oos_predictions.csv, fold_metrics, summary
    4. validation       -> fold_boundary_audit.csv, walk_forward_validation_summary.json
    5. canonical_oos    -> results/oos_predictions.csv + manifest

Each stage's contract is recorded in pipeline_manifest.json. On the next run,
the pipeline compares the contract the new run *would* produce against the
recorded one and raises StaleOutputError if outputs exist but were produced
from different inputs -- so a stale downstream output can never be mistaken
for a current one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, load_config
from .contract import StageContract, StaleOutputError, now_utc_iso
from .data_foundation import run_data_foundation
from .model import run_baseline, run_lightgbm, run_validation, run_canonical_oos


def _save_master_manifest(cfg, contracts: dict[str, StageContract]) -> None:
    """Write the top-level pipeline manifest aggregating all stage contracts."""
    manifest_path = cfg.resolve(cfg.paths.pipeline_manifest)
    master = {
        "pipeline_manifest_version": "1.0",
        "generated_at_utc": now_utc_iso(),
        "config_hash": cfg.config_hash,
        "stages": {name: c.to_dict() for name, c in contracts.items()},
    }
    manifest_path.write_text(json.dumps(master, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote pipeline manifest: {manifest_path}")


def run_all(cfg: Config, force: bool = False) -> dict[str, StageContract]:
    """Run all stages in order, returning the dict of stage contracts."""
    contracts: dict[str, StageContract] = {}

    print("=" * 60)
    print("Stage 1/5: data_foundation")
    print("=" * 60)
    contracts["data_foundation"] = run_data_foundation(cfg, force=force)
    print(f"  -> {contracts['data_foundation'].generated_at_utc}")

    print("\n" + "=" * 60)
    print("Stage 2/5: baseline")
    print("=" * 60)
    contracts["baseline"] = run_baseline(cfg, force=force)
    print(f"  -> {contracts['baseline'].generated_at_utc}")

    print("\n" + "=" * 60)
    print("Stage 3/5: lightgbm")
    print("=" * 60)
    contracts["lightgbm"] = run_lightgbm(cfg, force=force)
    print(f"  -> {contracts['lightgbm'].generated_at_utc}")

    print("\n" + "=" * 60)
    print("Stage 4/5: validation")
    print("=" * 60)
    contracts["validation"] = run_validation(cfg, force=force)
    print(f"  -> {contracts['validation'].generated_at_utc}")

    print("\n" + "=" * 60)
    print("Stage 5/5: canonical_oos")
    print("=" * 60)
    contracts["canonical_oos"] = run_canonical_oos(cfg, force=force)
    print(f"  -> {contracts['canonical_oos'].generated_at_utc}")

    _save_master_manifest(cfg, contracts)
    return contracts


def run_single(cfg: Config, stage: str, force: bool = False) -> StageContract:
    """Run a single stage by name."""
    runners = {
        "data_foundation": run_data_foundation,
        "baseline": run_baseline,
        "lightgbm": run_lightgbm,
        "validation": run_validation,
        "canonical_oos": run_canonical_oos,
    }
    if stage not in runners:
        raise ValueError(f"unknown stage: {stage}. Choose from: {list(runners)}")
    return runners[stage](cfg, force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SPY modeling pipeline end-to-end."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default=None,
        help="Run a single stage (data_foundation|baseline|lightgbm|validation|canonical_oos). "
             "If omitted, runs all stages in order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even if inputs are unchanged.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline_config.yaml (default: pipeline_config.yaml at repo root).",
    )
    args = parser.parse_args(argv)

    if args.config:
        from pipeline.config import load_config as _load
        cfg = _load(Path(args.config))
    else:
        cfg = load_config()

    print(f"Config hash: {cfg.config_hash[:16]}...")
    print(f"Repo root:   {cfg._repo_root}")

    if args.stage:
        contract = run_single(cfg, args.stage, force=args.force)
        print(f"\nStage '{args.stage}' complete.")
        print(f"  Generated at: {contract.generated_at_utc}")
        print(f"  Outputs: {list(contract.output_hashes.keys())}")
    else:
        contracts = run_all(cfg, force=args.force)
        print("\n" + "=" * 60)
        print("All stages complete.")
        print("=" * 60)
        for name, c in contracts.items():
            print(f"  {name}: {c.generated_at_utc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
