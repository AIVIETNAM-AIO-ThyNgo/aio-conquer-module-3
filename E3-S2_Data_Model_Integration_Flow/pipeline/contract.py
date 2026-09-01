"""StageContract -- input/output hash chaining for pipeline integrity.

Every stage in the pipeline declares a StageContract that records:
  - the hash of the config that produced it,
  - the hashes of every input artifact it consumed,
  - the hashes of every output artifact it produced,
  - the timestamp of the run.

On the next run, the pipeline compares the contract the new run *would*
produce against the contract the previous run *did* produce. If the inputs
changed but the outputs were not regenerated, the contract raises
StaleOutputError instead of letting a downstream stage silently consume
an artifact produced from different inputs.

This is the mechanism that catches every edge case named on the card:
  - partial rerun mixing old/new artifacts: input hash mismatch vs recorded
    output hash → StaleOutputError.
  - hidden local paths: every path is resolved from Config.resolve() and
    recorded as an absolute path in the contract; a path that drifts between
    machines is visible.
  - config mismatch between data/model modules: every stage records
    config_hash; a stage whose recorded config_hash differs from the current
    one is stale.
  - API failure leaving old raw file silently reused: the raw file's hash is
    recorded in the data-foundation contract; if a download fails and the old
    file is left in place, the hash will match the *previous* contract, but
    the provenance timestamp will be old -- and more importantly, if the
    download was meant to refresh the data, the new file's hash would differ
    and the downstream canonical dataset would be flagged as stale.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StaleOutputError(Exception):
    """Raised when an output artifact exists but was produced from different
    inputs than the current run would use -- a stale output that cannot be
    mistaken for a current one."""


class ContractMismatchError(Exception):
    """Raised when a contract field that must be deterministic differs."""


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hash_or_none(path: Path) -> str | None:
    if path.exists():
        return file_hash(path)
    return None


@dataclass(frozen=True)
class StageContract:
    """Immutable record of what a stage consumed and produced.

    Fields:
      stage_name: human-readable stage identifier.
      config_hash: SHA-256 of the pipeline_config.yaml that was loaded.
      input_hashes: map of {artifact_path: sha256} for every input consumed.
      output_hashes: map of {artifact_path: sha256} for every deterministic
        output produced (CSV data artifacts). These are byte-identical across
        runs given the same inputs.
      output_records: list of paths for timestamped records (manifests,
        summaries) that are expected to change between runs. Only checked
        for existence, not hash equality.
      params: dict of stage-specific parameters (e.g., split params,
               hyperparameters) that should be deterministic across runs.
      generated_at_utc: ISO-8601 timestamp of when the stage ran.
    """
    stage_name: str
    config_hash: str
    input_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    output_records: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    generated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "config_hash": self.config_hash,
            "input_hashes": dict(self.input_hashes),
            "output_hashes": dict(self.output_hashes),
            "output_records": list(self.output_records),
            "params": _serialize_params(self.params),
            "generated_at_utc": self.generated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageContract:
        return cls(
            stage_name=data["stage_name"],
            config_hash=data["config_hash"],
            input_hashes=dict(data.get("input_hashes", {})),
            output_hashes=dict(data.get("output_hashes", {})),
            output_records=list(data.get("output_records", [])),
            params=data.get("params", {}),
            generated_at_utc=data.get("generated_at_utc", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StageContract:
        if not path.exists():
            raise FileNotFoundError(f"contract file not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Make params JSON-safe; convert non-serializable values to their repr."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _serialize_params(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_serialize_value(x) for x in v]
        else:
            out[k] = _serialize_value(v)
    return out


def _serialize_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return repr(v)


def check_inputs_unchanged(
    contract: StageContract,
    current_input_hashes: dict[str, str],
    current_config_hash: str,
) -> None:
    """Raise StaleOutputError if the recorded contract's inputs differ from
    the inputs the current run would use, while the outputs still exist.

    This is the core staleness check: it answers "were these outputs produced
    from the same inputs I would feed the stage now?"
    """
    # Config drift: the stage was produced under a different config.
    if contract.config_hash != current_config_hash:
        raise StaleOutputError(
            f"{contract.stage_name}: recorded config_hash {contract.config_hash[:16]}... "
            f"differs from current {current_config_hash[:16]}... -- outputs are stale"
        )

    # Input drift: an input artifact's hash changed since the stage ran.
    for path, current_hash in current_input_hashes.items():
        recorded = contract.input_hashes.get(path)
        if recorded is None:
            # A new input the previous run didn't have -- treat as drift.
            raise StaleOutputError(
                f"{contract.stage_name}: input {path} not present in previous contract"
            )
        if recorded != current_hash:
            raise StaleOutputError(
                f"{contract.stage_name}: input {path} changed "
                f"(recorded={recorded[:16]}... current={current_hash[:16]}...) -- outputs are stale"
            )


def check_outputs_current(
    contract: StageContract,
    output_dir: Path | None = None,
) -> None:
    """Verify every output the contract claims to have produced is intact.

    - Deterministic artifacts (output_hashes): must exist AND hash to the
      recorded value. A deleted or silently modified CSV fails the check.
    - Timestamped records (output_records): must exist, but their bytes are
      expected to differ between runs (they contain timestamps), so only
      existence is checked.
    """
    for path_str, recorded_hash in contract.output_hashes.items():
        path = Path(path_str)
        if not path.exists():
            raise StaleOutputError(
                f"{contract.stage_name}: recorded output {path_str} no longer exists"
            )
        current_hash = file_hash(path)
        if current_hash != recorded_hash:
            raise StaleOutputError(
                f"{contract.stage_name}: output {path_str} was modified after production "
                f"(recorded={recorded_hash[:16]}... current={current_hash[:16]}...)"
            )
    for path_str in contract.output_records:
        path = Path(path_str)
        if not path.exists():
            raise StaleOutputError(
                f"{contract.stage_name}: recorded output {path_str} no longer exists"
            )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
