"""
E4-S1 [P0][QA] Pre-Model Leakage & Data Quality Gate
====================================================

Independent verification of the E1-S6 canonical modeling dataset.

This suite deliberately does NOT import the notebook. It re-derives the whole
feature/target/regime pipeline from the immutable E1-S1 raw artifact and then
asks whether the published canonical dataset agrees. If the notebook and this
file disagree, one of them is wrong -- which is the point of an audit.

Every test maps to a line on the Leakage Audit Checklist; the mapping is in the
docstring of each test.

Run:
    python -m pytest tests/test_E1_S6_canonical_dataset.py -v

Owner: QA/QC.  Depends on: E1-S6.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# Frozen constants -- must mirror the E1-S3/S4/S5 cards exactly.
# --------------------------------------------------------------------------

HORIZON_TRADING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252
MIN_HISTORICAL_VOL_OBSERVATIONS = 252

RETURN_WINDOWS = [5, 10, 20]
VOLATILITY_WINDOWS = [5, 10, 20]
TREND_WINDOWS = [10, 20, 60]

DAILY_RETURN_COL = "return_1d"
TARGET_COL = "forward_return_5d"
REGIME_VOLATILITY_COL = "volatility_20d"
REGIME_THRESHOLD_COL = "historical_volatility_threshold_20d"
REGIME_COL = "regime"

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "trend_10d", "trend_20d", "trend_60d",
    "volume_ratio_20d",
]

# Tolerance for float round-trips through CSV text.
ATOL = 1e-12
RTOL = 1e-9

# How many prediction dates get the expensive point-in-time rebuild.
POINT_IN_TIME_SAMPLE_SIZE = 12
POINT_IN_TIME_SEED = 20260824

# A single daily feature that genuinely predicted 5-day SPY returns this well
# would be a research result, not a feature. Treat it as a leakage alarm.
MAX_PLAUSIBLE_ABS_CORRELATION = 0.20


# --------------------------------------------------------------------------
# Artifact discovery -- tolerates both the flat layout and the README layout.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _locate(*candidates: str) -> Path:
    for candidate in candidates:
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    matches = sorted(REPO_ROOT.rglob(Path(candidates[0]).name))
    if matches:
        return matches[0]
    pytest.fail(
        f"Required artifact not found. Looked for {candidates} under {REPO_ROOT}."
    )


RAW_CSV_PATH = _locate(
    "data/raw/E1-S1_SPY_OHLCV_auto_adjusted.csv",
    "E1-S1_SPY_OHLCV_auto_adjusted.csv",
)
RAW_PROVENANCE_PATH = _locate(
    "data/raw/E1-S1_SPY_OHLCV_auto_adjusted.provenance.json",
    "E1-S1_SPY_OHLCV_auto_adjusted.provenance.json",
)
CANONICAL_CSV_PATH = _locate(
    "data/processed/E1-S6_canonical_modeling_dataset.csv",
    "E1-S6_canonical_modeling_dataset.csv",
)
MANIFEST_PATH = _locate(
    "data/processed/E1-S6_dataset_manifest.json",
    "E1-S6_dataset_manifest.json",
)
DATA_DICTIONARY_PATH = _locate(
    "docs/E1-S6_data_dictionary.csv",
    "E1-S6_data_dictionary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Reference pipeline -- re-derived independently from raw OHLCV.
# --------------------------------------------------------------------------

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the 11 E1-S4 features. Every window is trailing and right-aligned.

    This function must never look past the last row it is handed. That property
    is what `test_features_are_point_in_time` proves empirically.
    """
    out = raw.copy()

    out[DAILY_RETURN_COL] = out["Close"].pct_change(fill_method=None)

    for window in RETURN_WINDOWS:
        out[f"return_{window}d"] = out["Close"] / out["Close"].shift(window) - 1

    for window in VOLATILITY_WINDOWS:
        out[f"volatility_{window}d"] = (
            out[DAILY_RETURN_COL]
            .rolling(window=window, min_periods=window, center=False)
            .std(ddof=1)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    for window in TREND_WINDOWS:
        moving_average = (
            out["Close"].rolling(window=window, min_periods=window, center=False).mean()
        )
        out[f"trend_{window}d"] = out["Close"] / moving_average - 1

    volume_ma_20d = (
        out["Volume"].rolling(window=20, min_periods=20, center=False).mean()
    )
    out["volume_ratio_20d"] = out["Volume"] / volume_ma_20d

    return out


def build_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the t+5 forward return as a label on prediction date t."""
    out = frame.copy()
    out[TARGET_COL] = (
        out["Close"].shift(-HORIZON_TRADING_DAYS) / out["Close"] - 1
    )
    return out


def build_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Label Low/High vol against an expanding median of STRICTLY PRIOR values.

    The `.shift(1)` is the entire leakage guard: without it the threshold at t
    would include volatility_20d[t] itself.
    """
    out = frame.copy()
    out[REGIME_THRESHOLD_COL] = (
        out[REGIME_VOLATILITY_COL]
        .expanding(min_periods=MIN_HISTORICAL_VOL_OBSERVATIONS)
        .median()
        .shift(1)
    )
    out[REGIME_COL] = pd.Series(pd.NA, index=out.index, dtype="string")
    eligible = out[[REGIME_VOLATILITY_COL, REGIME_THRESHOLD_COL]].notna().all(axis=1)
    out.loc[
        eligible & (out[REGIME_VOLATILITY_COL] >= out[REGIME_THRESHOLD_COL]), REGIME_COL
    ] = "HighVol"
    out.loc[
        eligible & (out[REGIME_VOLATILITY_COL] < out[REGIME_THRESHOLD_COL]), REGIME_COL
    ] = "LowVol"
    return out


def build_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    return build_regime(build_target(build_features(raw)))


def build_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the E1-S6 admitted row set."""
    full = build_pipeline(raw)
    canonical = full[FEATURE_COLUMNS + [TARGET_COL, REGIME_COL]].dropna().copy()
    canonical = canonical.reset_index()
    canonical["Date"] = pd.to_datetime(canonical["Date"])
    return canonical[["Date", *FEATURE_COLUMNS, TARGET_COL, REGIME_COL]]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    frame = pd.read_csv(RAW_CSV_PATH, parse_dates=["Date"]).set_index("Date")
    frame = frame.sort_index()
    return frame


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def published() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_CSV_PATH, parse_dates=["Date"])


@pytest.fixture(scope="session")
def recomputed(raw: pd.DataFrame) -> pd.DataFrame:
    return build_canonical(raw)


# ==========================================================================
# CHECKLIST ITEM 1
# "For every feature, document the latest timestamp of information used and
#  verify it is <= prediction time t."
# ==========================================================================

def test_data_dictionary_documents_every_canonical_column(manifest):
    """Item 1 (documentation half): every column has a declared timestamp
    semantic, formula and window -- no silent columns."""
    dictionary = pd.read_csv(DATA_DICTIONARY_PATH)
    documented = list(dictionary["column"])

    assert documented == manifest["columns"], (
        "Data dictionary columns must match the manifest schema exactly.\n"
        f"dictionary={documented}\nmanifest={manifest['columns']}"
    )
    for required in ("formula", "window", "timestamp_semantics", "role"):
        blank = dictionary[dictionary[required].isna() | (dictionary[required].astype(str).str.strip() == "")]
        assert blank.empty, f"Columns missing '{required}': {list(blank['column'])}"


def test_features_are_point_in_time(raw, published):
    """Item 1 (verification half) -- THE decisive leakage test.

    For sampled prediction dates t, rebuild all 11 features from a raw frame
    TRUNCATED at t. A feature that peeks into the future cannot survive having
    the future physically removed: its truncated value would differ from the
    published one.
    """
    rng = np.random.default_rng(POINT_IN_TIME_SEED)
    published_indexed = published.set_index("Date")

    eligible_dates = published_indexed.index
    sampled = sorted(
        rng.choice(
            len(eligible_dates),
            size=min(POINT_IN_TIME_SAMPLE_SIZE, len(eligible_dates)),
            replace=False,
        )
    )
    # Always include the first and last admitted date.
    sampled = sorted(set(sampled) | {0, len(eligible_dates) - 1})

    failures = []
    for position in sampled:
        as_of = eligible_dates[position]
        truncated = raw.loc[:as_of]
        assert truncated.index[-1] == as_of

        rebuilt = build_features(truncated).iloc[-1]
        for column in FEATURE_COLUMNS:
            expected = published_indexed.at[as_of, column]
            actual = rebuilt[column]
            if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
                failures.append(
                    f"{as_of.date()} {column}: published={expected!r} "
                    f"point_in_time={actual!r} delta={actual - expected!r}"
                )

    assert not failures, (
        "LEAKAGE: feature values changed when future rows were removed.\n"
        + "\n".join(failures)
    )


def test_no_feature_correlates_implausibly_with_target(published):
    """Item 1 (sanity half): a daily price/volume feature that predicts 5-day
    forward SPY returns this strongly is leakage, not alpha."""
    suspicious = {}
    for column in FEATURE_COLUMNS:
        correlation = published[column].corr(published[TARGET_COL])
        if abs(correlation) > MAX_PLAUSIBLE_ABS_CORRELATION:
            suspicious[column] = correlation

    assert not suspicious, (
        "Implausibly predictive features -- investigate for leakage: "
        f"{suspicious}"
    )


# ==========================================================================
# TARGET INTEGRITY (supports items 1 and 3)
# ==========================================================================

def test_target_matches_manual_forward_return(raw, published):
    """y[t] must be exactly Close[t+5]/Close[t] - 1, computed by hand."""
    close = raw["Close"]
    positions = {0, len(published) // 3, len(published) // 2, len(published) - 1}

    for position in sorted(positions):
        as_of = published["Date"].iloc[position]
        raw_position = close.index.get_loc(as_of)
        manual = (
            close.iloc[raw_position + HORIZON_TRADING_DAYS] / close.iloc[raw_position] - 1
        )
        reported = published[TARGET_COL].iloc[position]
        assert np.isclose(manual, reported, rtol=RTOL, atol=ATOL), (
            f"{as_of.date()}: manual={manual!r} published={reported!r}"
        )


def test_target_is_not_a_feature(manifest):
    """The label must never appear in X."""
    assert TARGET_COL not in manifest["feature_columns"]
    assert REGIME_COL not in manifest["feature_columns"]
    assert manifest["target_column"] == TARGET_COL
    assert len(manifest["feature_columns"]) == len(set(manifest["feature_columns"])) == 11


def test_target_horizon_is_not_silently_shifted(raw, published):
    """Guard against an off-by-one horizon: the last admitted date must be
    exactly HORIZON trading days before the last raw date."""
    last_admitted = published["Date"].iloc[-1]
    raw_position = raw.index.get_loc(last_admitted)
    trailing_rows = len(raw) - 1 - raw_position
    assert trailing_rows == HORIZON_TRADING_DAYS, (
        f"Expected exactly {HORIZON_TRADING_DAYS} unlabelled trailing rows, "
        f"found {trailing_rows}. Last admitted={last_admitted.date()}, "
        f"last raw={raw.index[-1].date()}."
    )


# ==========================================================================
# CHECKLIST ITEM 2
# "Verify no centered rolling, backfill, future-derived imputation or
#  full-sample statistics are used in features/regime construction."
# ==========================================================================

FORBIDDEN_SOURCE_PATTERNS = [
    (r"center\s*=\s*True", "centered rolling window"),
    (r"\.bfill\s*\(", "backfill"),
    (r"method\s*=\s*['\"]bfill['\"]", "backfill"),
    (r"method\s*=\s*['\"]backfill['\"]", "backfill"),
    (r"\.interpolate\s*\(", "interpolation (can borrow from the future)"),
    (r"\.fit_transform\s*\(", "fit_transform on a full frame (fit-before-split)"),
    (r"\.shift\s*\(\s*-\s*(?!%d\b)\d+" % HORIZON_TRADING_DAYS,
     "negative shift other than the sanctioned target horizon"),
]


def _notebook_paths() -> list[Path]:
    return sorted(
        path for path in REPO_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def _code_cells(path: Path) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_pipeline_source_has_no_forward_looking_constructs():
    """Item 2: scan the executable source, not just the prose claims."""
    notebooks = _notebook_paths()
    assert notebooks, f"No notebook found under {REPO_ROOT} to audit."

    findings = []
    for path in notebooks:
        for index, cell in enumerate(_code_cells(path)):
            source = "".join(cell["source"])
            for pattern, label in FORBIDDEN_SOURCE_PATTERNS:
                for match in re.finditer(pattern, source):
                    findings.append(
                        f"{path.name} code cell {index}: {label} -> {match.group(0)!r}"
                    )

    assert not findings, "Forward-looking constructs found:\n" + "\n".join(findings)


def test_regime_threshold_uses_only_prior_observations(raw, published):
    """Item 2 (the full-sample-statistic trap).

    Rebuild each sampled regime label from truncated history and require a
    match. Then prove the guard is not vacuous: a full-sample median threshold
    must disagree with the published labels somewhere. If both agree
    everywhere, the test can't tell a leaky threshold from a safe one.
    """
    published_indexed = published.set_index("Date")
    rng = np.random.default_rng(POINT_IN_TIME_SEED)
    positions = sorted(
        set(
            rng.choice(len(published_indexed), size=min(10, len(published_indexed)), replace=False).tolist()
        )
        | {0, len(published_indexed) - 1}
    )

    failures = []
    for position in positions:
        as_of = published_indexed.index[position]
        truncated = build_features(raw.loc[:as_of])
        prior_vol = truncated[REGIME_VOLATILITY_COL].iloc[:-1].dropna()
        assert len(prior_vol) >= MIN_HISTORICAL_VOL_OBSERVATIONS, (
            f"{as_of.date()} admitted with only {len(prior_vol)} prior volatility "
            f"observations; minimum is {MIN_HISTORICAL_VOL_OBSERVATIONS}."
        )
        threshold = prior_vol.median()
        current_vol = truncated[REGIME_VOLATILITY_COL].iloc[-1]
        expected = "HighVol" if current_vol >= threshold else "LowVol"
        actual = published_indexed.at[as_of, REGIME_COL]
        if expected != actual:
            failures.append(
                f"{as_of.date()}: past-only={expected} published={actual} "
                f"(vol={current_vol:.6f}, threshold={threshold:.6f})"
            )

    assert not failures, (
        "LEAKAGE: regime labels do not match a strictly-past threshold.\n"
        + "\n".join(failures)
    )

    full_sample_threshold = build_features(raw)[REGIME_VOLATILITY_COL].median()
    leaky_labels = np.where(
        published_indexed[REGIME_VOLATILITY_COL] >= full_sample_threshold,
        "HighVol",
        "LowVol",
    )
    disagreements = int((leaky_labels != published_indexed[REGIME_COL].to_numpy()).sum())
    assert disagreements > 0, (
        "Non-vacuity check failed: a full-sample threshold produces identical "
        "labels, so this test cannot detect a leaky threshold."
    )


def test_regime_labels_are_clean(published):
    """Item 2: no stray categories, both regimes actually populated."""
    values = set(published[REGIME_COL].dropna().unique())
    assert values == {"LowVol", "HighVol"}, f"Unexpected regime values: {values}"
    counts = published[REGIME_COL].value_counts()
    assert counts.min() >= 100, f"A regime is too thin to evaluate: {counts.to_dict()}"


# ==========================================================================
# CHECKLIST ITEM 3
# "Verify target/feature/regime indices remain aligned after rolling-window
#  NaNs and target shift are removed."
# ==========================================================================

def test_canonical_row_set_matches_independent_rebuild(published, recomputed):
    """Item 3: the admitted row set is exactly what the pipeline produces."""
    assert list(published.columns) == list(recomputed.columns)
    pd.testing.assert_series_equal(
        published["Date"], recomputed["Date"], check_names=False
    )
    for column in FEATURE_COLUMNS + [TARGET_COL]:
        np.testing.assert_allclose(
            published[column].to_numpy(),
            recomputed[column].to_numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Column {column} disagrees with the independent rebuild.",
        )
    assert (
        published[REGIME_COL].astype(str).to_numpy()
        == recomputed[REGIME_COL].astype(str).to_numpy()
    ).all()


def test_no_missing_values_and_no_duplicate_dates(published):
    """Item 3: a misalignment usually shows up first as a NaN or a dupe."""
    assert not published.isna().any().any(), (
        f"NaNs present in canonical rows: "
        f"{published.isna().sum()[published.isna().sum() > 0].to_dict()}"
    )
    assert published["Date"].is_unique, "Duplicate prediction dates."
    assert published["Date"].is_monotonic_increasing, "Dates are not sorted."


def test_row_count_accounting_is_explainable(raw, published, manifest):
    """Item 3: the row loss must be fully explained by warm-up + horizon,
    not by an unnoticed misalignment silently dropping rows."""
    full = build_pipeline(raw)
    admitted = full[FEATURE_COLUMNS + [TARGET_COL, REGIME_COL]].notna().all(axis=1)

    first_admitted_position = int(np.flatnonzero(admitted.to_numpy())[0])
    warmup_rows = first_admitted_position
    unlabelled_tail = HORIZON_TRADING_DAYS

    assert len(published) == len(raw) - warmup_rows - unlabelled_tail, (
        f"Unexplained rows. raw={len(raw)}, warm-up={warmup_rows}, "
        f"unlabelled tail={unlabelled_tail}, canonical={len(published)}"
    )
    assert len(published) == manifest["rows"], (
        f"Manifest claims {manifest['rows']} rows, file has {len(published)}."
    )


# ==========================================================================
# CHECKLIST ITEM 4
# "Record audit result as PASS/BLOCKED; any confirmed leakage invalidates
#  downstream metrics until a clean rerun is completed."
# ==========================================================================

def test_raw_artifact_matches_its_provenance_hash():
    """Item 4: the audit is only meaningful if the input is the pinned one."""
    provenance = json.loads(RAW_PROVENANCE_PATH.read_text(encoding="utf-8"))
    actual = _sha256(RAW_CSV_PATH)
    assert actual == provenance["sha256"], (
        "Raw SPY artifact has been modified since acquisition.\n"
        f"provenance={provenance['sha256']}\nactual   ={actual}"
    )


def test_published_artifacts_match_manifest_hashes(manifest):
    """Item 4: the reviewed dataset is the published dataset."""
    mismatches = []
    for label, path, key in (
        ("canonical dataset", CANONICAL_CSV_PATH, "canonical_sha256"),
        ("data dictionary", DATA_DICTIONARY_PATH, "data_dictionary_sha256"),
        ("raw source", RAW_CSV_PATH, "source_raw_sha256"),
    ):
        actual = _sha256(path)
        if actual != manifest[key]:
            mismatches.append(f"{label}: manifest={manifest[key]} actual={actual}")
    assert not mismatches, "Artifact hash mismatch:\n" + "\n".join(mismatches)


def test_manifest_points_at_files_that_exist(manifest):
    """Item 4: a manifest naming a missing file cannot support a rerun."""
    missing = []
    for key in ("notebook", "canonical_path", "data_dictionary_path", "source_raw_path"):
        name = Path(manifest[key]).name
        if not list(REPO_ROOT.rglob(name)):
            missing.append(f"{key}={manifest[key]!r}")
    assert not missing, (
        "Manifest references files that do not exist in the repository: "
        + ", ".join(missing)
    )


def test_pipeline_is_deterministic(raw):
    """Item 4: two runs must produce a byte-identical row set."""
    first = build_canonical(raw)
    second = build_canonical(raw)

    first_hash = hashlib.sha256(
        pd.util.hash_pandas_object(first, index=True).to_numpy().tobytes()
    ).hexdigest()
    second_hash = hashlib.sha256(
        pd.util.hash_pandas_object(second, index=True).to_numpy().tobytes()
    ).hexdigest()

    assert first_hash == second_hash, "Pipeline is not deterministic across runs."
    assert first.shape == second.shape


def test_notebook_was_executed_top_to_bottom():
    """Item 4 / EDGE CASE 'manual notebook edits'.

    Out-of-order execution counts mean the published artifacts came from a
    kernel state nobody can reproduce -- the numbers may be fine and still be
    unauditable.
    """
    problems = []
    for path in _notebook_paths():
        counts = [cell.get("execution_count") for cell in _code_cells(path)]
        if any(count is None for count in counts):
            problems.append(f"{path.name}: contains unexecuted code cells {counts}")
            continue
        if counts != sorted(counts):
            problems.append(f"{path.name}: out-of-order execution counts {counts}")
        if len(set(counts)) != len(counts):
            problems.append(f"{path.name}: duplicate execution counts {counts}")

    assert not problems, (
        "Notebook state is not reproducible (restart kernel + run all):\n"
        + "\n".join(problems)
    )


def test_reproduce_instructions_are_actionable():
    """Item 4: README promises a rerun path. If it does not exist, the
    determinism claim cannot be independently checked by a reviewer."""
    readme = (REPO_ROOT / "README.md")
    if not readme.exists():
        pytest.skip("No README.md to check.")
    text = readme.read_text(encoding="utf-8")

    missing = []
    for referenced in re.findall(r"`([\w./\\-]+\.(?:py|txt|csv|json|ipynb))`", text):
        name = Path(referenced).name
        if not list(REPO_ROOT.rglob(name)):
            missing.append(referenced)

    assert not missing, (
        "README references files that do not exist: " + ", ".join(sorted(set(missing)))
    )
