# ============================================================
# engine/results.py
# ============================================================

"""
Result normalization and eligibility evaluation for WorldQuant BRAIN.

This module does NOT:
    - authenticate
    - submit simulations
    - modify ace_lib
    - generate expressions

It does:
    - normalize raw BRAIN simulation results
    - extract train/test metrics
    - preserve BRAIN submission-test results
    - evaluate NORMAL alpha criteria
    - evaluate POWER_POOL alpha criteria
    - distinguish PASS / FAIL / UNKNOWN / NOT_REQUIRED
    - produce clean DataFrame-ready records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd


# ============================================================
# STATUS CONSTANTS
# ============================================================

SIMULATION_ERROR = "SIMULATION_ERROR"
SIMULATED = "SIMULATED"

ELIGIBLE = "ELIGIBLE"
REJECTED = "REJECTED"
UNKNOWN = "UNKNOWN"
NOT_REQUIRED = "NOT_REQUIRED"


# ============================================================
# ALPHA TYPES
# ============================================================

NORMAL = "NORMAL"
POWER_POOL = "POWER_POOL"


# ============================================================
# YOUR LOCKED RESEARCH CRITERIA
# ============================================================

# Normal alpha
NORMAL_SHARPE_MIN = 1.58
NORMAL_FITNESS_MIN = 1.00

# Power Pool
POWER_POOL_SHARPE_MIN = 1.00

# Both
MIN_TURNOVER = 0.01
MAX_TURNOVER = 0.70

MAX_WEIGHT = 0.10

# Correlation thresholds
NORMAL_SELF_CORRELATION_MAX = 0.70
POWER_POOL_CORRELATION_MAX = 0.50

# Correlation improvement exception
CORRELATION_SHARPE_MULTIPLIER = 1.10

# Power Pool complexity limits
POWER_POOL_MAX_OPERATOR_OCCURRENCES = 8
POWER_POOL_MAX_FIELDS = 3

# BRAIN grouping fields excluded from PP's 3-field count
POWER_POOL_GROUPING_FIELDS = {
    "country",
    "industry",
    "subindustry",
    "currency",
    "market",
    "sector",
    "exchange",
}


# ============================================================
# SMALL HELPERS
# ============================================================

def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _safe_float(value: Any) -> float | None:
    """
    Convert numeric-like values to float.

    Returns None instead of guessing when unavailable.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(
    section: Mapping[str, Any],
    key: str,
) -> float | None:
    if not isinstance(section, Mapping):
        return None

    return _safe_float(
        section.get(key)
    )


def _status_from_boolean(
    condition: bool | None,
) -> str:
    if condition is None:
        return UNKNOWN

    return (
        "PASS"
        if condition
        else "FAIL"
    )


# ============================================================
# RAW BRAIN EXPRESSION
# ============================================================

def extract_expression(
    record: Mapping[str, Any],
) -> str:
    """
    Extract the exact expression stored in simulate_data.

    Successful REGULAR result:
        record["simulate_data"]["regular"]

    Failed results from the existing ace_lib may contain the same
    simulate_data payload.
    """

    simulate_data = _as_dict(
        record.get(
            "simulate_data"
        )
    )

    regular = simulate_data.get(
        "regular"
    )

    if regular is None:
        return ""

    if isinstance(
        regular,
        Mapping,
    ):
        value = regular.get(
            "code"
        )
    else:
        value = regular

    if value is None:
        return ""

    return str(
        value
    ).strip()


# ============================================================
# BRAIN TEST TABLE NORMALIZATION
# ============================================================

def tests_to_dataframe(
    tests: Any,
) -> pd.DataFrame:
    """
    Normalize BRAIN's is_tests value.

    ace_lib currently commonly returns a pandas DataFrame.
    """

    if isinstance(
        tests,
        pd.DataFrame,
    ):
        return tests.copy()

    if isinstance(
        tests,
        Mapping,
    ):
        return pd.DataFrame(
            [dict(tests)]
        )

    if isinstance(
        tests,
        list,
    ):

        if not tests:
            return pd.DataFrame()

        if all(
            isinstance(
                item,
                Mapping,
            )
            for item in tests
        ):

            return pd.DataFrame(
                tests
            )

    return pd.DataFrame()


def get_test_result(
    tests: Any,
    test_name: str,
) -> str:
    """
    Return one BRAIN test's result.

    Possible return values:
        PASS
        FAIL
        WARNING
        PENDING
        UNKNOWN
    """

    frame = tests_to_dataframe(
        tests
    )

    if frame.empty:
        return UNKNOWN

    if "name" not in frame.columns:
        return UNKNOWN

    rows = frame[
        frame["name"]
        .astype(str)
        .str.upper()
        == str(test_name).upper()
    ]

    if rows.empty:
        return UNKNOWN

    value = rows.iloc[0].get(
        "result"
    )

    if pd.isna(value):
        return UNKNOWN

    return str(
        value
    ).strip().upper()


def all_failed_tests(
    tests: Any,
) -> list[dict[str, Any]]:
    """
    Return every BRAIN submission-test record whose result is FAIL.
    """

    frame = tests_to_dataframe(
        tests
    )

    if frame.empty:
        return []

    if "result" not in frame.columns:
        return []

    result_column = (
        frame["result"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return frame[
        result_column == "FAIL"
    ].to_dict(
        orient="records"
    )


def all_warning_tests(
    tests: Any,
) -> list[dict[str, Any]]:
    """
    Return every BRAIN submission-test record whose result is WARNING.
    """

    frame = tests_to_dataframe(
        tests
    )

    if frame.empty:
        return []

    if "result" not in frame.columns:
        return []

    result_column = (
        frame["result"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return frame[
        result_column == "WARNING"
    ].to_dict(
        orient="records"
    )


# ============================================================
# METRIC EXTRACTION
# ============================================================

@dataclass
class PerformanceMetrics:
    """
    Core train/test metrics from a BRAIN simulation.
    """

    train_sharpe: float | None = None
    test_sharpe: float | None = None

    train_fitness: float | None = None
    test_fitness: float | None = None

    train_turnover: float | None = None
    test_turnover: float | None = None

    train_returns: float | None = None
    test_returns: float | None = None

    train_drawdown: float | None = None
    test_drawdown: float | None = None

    train_margin: float | None = None
    test_margin: float | None = None

    train_pnl: float | None = None
    test_pnl: float | None = None


def extract_metrics(
    record: Mapping[str, Any],
) -> PerformanceMetrics:
    """
    Extract train/test performance from a raw BRAIN record.
    """

    train = _as_dict(
        record.get(
            "train"
        )
    )

    test = _as_dict(
        record.get(
            "test"
        )
    )

    return PerformanceMetrics(
        train_sharpe=_metric(
            train,
            "sharpe",
        ),
        test_sharpe=_metric(
            test,
            "sharpe",
        ),

        train_fitness=_metric(
            train,
            "fitness",
        ),
        test_fitness=_metric(
            test,
            "fitness",
        ),

        train_turnover=_metric(
            train,
            "turnover",
        ),
        test_turnover=_metric(
            test,
            "turnover",
        ),

        train_returns=_metric(
            train,
            "returns",
        ),
        test_returns=_metric(
            test,
            "returns",
        ),

        train_drawdown=_metric(
            train,
            "drawdown",
        ),
        test_drawdown=_metric(
            test,
            "drawdown",
        ),

        train_margin=_metric(
            train,
            "margin",
        ),
        test_margin=_metric(
            test,
            "margin",
        ),

        train_pnl=_metric(
            train,
            "pnl",
        ),
        test_pnl=_metric(
            test,
            "pnl",
        ),
    )


# ============================================================
# SIMULATION STATUS
# ============================================================

def determine_simulation_status(
    record: Mapping[str, Any],
) -> str:
    """
    BRAIN/ace_lib considers the simulation successful when an
    alpha_id is returned.
    """

    alpha_id = record.get(
        "alpha_id"
    )

    if alpha_id is not None:
        return SIMULATED

    return SIMULATION_ERROR


# ============================================================
# TEST GATES
# ============================================================

@dataclass
class GateResult:
    name: str
    status: str
    value: Any = None
    threshold: Any = None
    detail: str = ""


def metric_gate(
    *,
    name: str,
    value: float | None,
    threshold: float,
    comparator: str = ">",
) -> GateResult:
    """
    Generic numeric gate.
    """

    if value is None:
        return GateResult(
            name=name,
            status=UNKNOWN,
            value=None,
            threshold=threshold,
            detail="Metric unavailable.",
        )

    if comparator == ">":
        passed = (
            value > threshold
        )

    elif comparator == ">=":
        passed = (
            value >= threshold
        )

    elif comparator == "<":
        passed = (
            value < threshold
        )

    elif comparator == "<=":
        passed = (
            value <= threshold
        )

    else:
        raise ValueError(
            f"Unsupported comparator: {comparator}"
        )

    return GateResult(
        name=name,
        status=(
            "PASS"
            if passed
            else "FAIL"
        ),
        value=value,
        threshold=threshold,
        detail=(
            f"{value} {comparator} {threshold}"
        ),
    )


def turnover_gate(
    *,
    name: str,
    value: float | None,
) -> GateResult:
    """
    User-specified turnover criterion:

        1% < turnover < 70%

    Both bounds are strict.
    """

    if value is None:

        return GateResult(
            name=name,
            status=UNKNOWN,
            value=None,
            threshold=(
                f"{MIN_TURNOVER} < x < "
                f"{MAX_TURNOVER}"
            ),
            detail="Turnover unavailable.",
        )

    passed = (
        MIN_TURNOVER
        < value
        < MAX_TURNOVER
    )

    return GateResult(
        name=name,
        status=(
            "PASS"
            if passed
            else "FAIL"
        ),
        value=value,
        threshold=(
            f"{MIN_TURNOVER} < x < "
            f"{MAX_TURNOVER}"
        ),
        detail=(
            f"{MIN_TURNOVER} < "
            f"{value} < "
            f"{MAX_TURNOVER}"
        ),
    )


# ============================================================
# BRAIN TEST GATES
# ============================================================

def brain_test_gate(
    *,
    tests: Any,
    name: str,
    display_name: str | None = None,
    required: bool = True,
) -> GateResult:
    """
    Evaluate a named BRAIN test.

    WARNING and PENDING are preserved as non-pass states.
    """

    result = get_test_result(
        tests,
        name,
    )

    label = (
        display_name
        or name
    )

    if result == "PASS":

        return GateResult(
            name=label,
            status="PASS",
            value=result,
            detail=f"{name}: PASS",
        )

    if result == "FAIL":

        return GateResult(
            name=label,
            status="FAIL",
            value=result,
            detail=f"{name}: FAIL",
        )

    if not required:

        return GateResult(
            name=label,
            status=NOT_REQUIRED,
            value=result,
            detail=(
                f"{name}: not required "
                "for this alpha/context."
            ),
        )

    return GateResult(
        name=label,
        status=UNKNOWN,
        value=result,
        detail=(
            f"{name}: {result}"
        ),
    )


# ============================================================
# NORMAL ALPHA EVALUATION
# ============================================================

@dataclass
class EligibilityResult:
    alpha_type: str

    status: str

    gates: list[GateResult] = field(
        default_factory=list
    )

    failed_gates: list[str] = field(
        default_factory=list
    )

    unknown_gates: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "alpha_type": self.alpha_type,
            "eligibility": self.status,
            "failed_gates": list(
                self.failed_gates
            ),
            "unknown_gates": list(
                self.unknown_gates
            ),
            "warnings": list(
                self.warnings
            ),
        }


def _finalize_eligibility(
    alpha_type: str,
    gates: list[GateResult],
    *,
    allow_unknown: bool = False,
) -> EligibilityResult:
    """
    Combine individual gates conservatively.
    """

    failed = [
        gate.name
        for gate in gates
        if gate.status == "FAIL"
    ]

    unknown = [
        gate.name
        for gate in gates
        if gate.status == UNKNOWN
    ]

    warnings = [
        gate.name
        for gate in gates
        if gate.status == "WARNING"
    ]

    if failed:

        status = REJECTED

    elif unknown and not allow_unknown:

        status = UNKNOWN

    else:

        status = ELIGIBLE

    return EligibilityResult(
        alpha_type=alpha_type,
        status=status,
        gates=gates,
        failed_gates=failed,
        unknown_gates=unknown,
        warnings=warnings,
    )


def evaluate_normal(
    record: Mapping[str, Any],
    *,
    require_benchmark_tests: bool = True,
) -> EligibilityResult:
    """
    Evaluate a NORMAL alpha using the locked criteria.

    Criteria:
        Sharpe > 1.58
        Fitness > 1.00
        1% < turnover < 70%
        Weight test PASS
        Sub-universe PASS
        Robust-universe PASS where applicable
        Self-correlation rule
        Other required BRAIN tests available in the result
    """

    gates: list[GateResult] = []

    if (
        determine_simulation_status(record)
        != SIMULATED
    ):

        return EligibilityResult(
            alpha_type=NORMAL,
            status=REJECTED,
            gates=[
                GateResult(
                    name="simulation",
                    status="FAIL",
                    detail=(
                        "Simulation did not "
                        "return an alpha_id."
                    ),
                )
            ],
            failed_gates=[
                "simulation"
            ],
        )

    metrics = extract_metrics(
        record
    )

    tests = record.get(
        "is_tests"
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    gates.append(
        metric_gate(
            name="test_sharpe",
            value=metrics.test_sharpe,
            threshold=NORMAL_SHARPE_MIN,
            comparator=">",
        )
    )

    gates.append(
        metric_gate(
            name="test_fitness",
            value=metrics.test_fitness,
            threshold=NORMAL_FITNESS_MIN,
            comparator=">",
        )
    )

    # --------------------------------------------------------
    # Turnover
    # --------------------------------------------------------

    gates.append(
        turnover_gate(
            name="test_turnover",
            value=metrics.test_turnover,
        )
    )

    # --------------------------------------------------------
    # BRAIN submission tests
    # --------------------------------------------------------

    gates.append(
        brain_test_gate(
            tests=tests,
            name="CONCENTRATED_WEIGHT",
            display_name="weight_test",
        )
    )

    gates.append(
        brain_test_gate(
            tests=tests,
            name="LOW_SUB_UNIVERSE_SHARPE",
            display_name="sub_universe",
        )
    )

    # Robust-universe is context-dependent, so only require it
    # when BRAIN actually exposes the test.
    robust_result = get_test_result(
        tests,
        "LOW_ROBUST_UNIVERSE_SHARPE",
    )

    if robust_result == UNKNOWN:

        gates.append(
            GateResult(
                name="robust_universe",
                status=NOT_REQUIRED,
                value=robust_result,
                detail=(
                    "Robust-universe test was not "
                    "returned by BRAIN."
                ),
            )
        )

    else:

        gates.append(
            brain_test_gate(
                tests=tests,
                name="LOW_ROBUST_UNIVERSE_SHARPE",
                display_name="robust_universe",
            )
        )

    # --------------------------------------------------------
    # Self correlation
    #
    # BRAIN's exact correlation value is not always embedded
    # in is_tests. We therefore do not invent a value.
    # --------------------------------------------------------

    self_corr = (
        record.get(
            "self_correlation"
        )
    )

    correlated_sharpe = (
        record.get(
            "most_correlated_alpha_sharpe"
        )
    )

    if self_corr is None:

        gates.append(
            GateResult(
                name="self_correlation",
                status=UNKNOWN,
                value=None,
                threshold=(
                    NORMAL_SELF_CORRELATION_MAX
                ),
                detail=(
                    "Self-correlation value "
                    "was not supplied."
                ),
            )
        )

    else:

        corr = _safe_float(
            self_corr
        )

        if corr is None:

            gates.append(
                GateResult(
                    name="self_correlation",
                    status=UNKNOWN,
                    value=self_corr,
                    threshold=(
                        NORMAL_SELF_CORRELATION_MAX
                    ),
                    detail=(
                        "Invalid self-correlation value."
                    ),
                )
            )

        elif (
            corr
            < NORMAL_SELF_CORRELATION_MAX
        ):

            gates.append(
                GateResult(
                    name="self_correlation",
                    status="PASS",
                    value=corr,
                    threshold=(
                        NORMAL_SELF_CORRELATION_MAX
                    ),
                    detail=(
                        f"{corr} < "
                        f"{NORMAL_SELF_CORRELATION_MAX}"
                    ),
                )
            )

        else:

            alpha_sharpe = (
                metrics.test_sharpe
            )

            comparison_sharpe = (
                _safe_float(
                    correlated_sharpe
                )
            )

            if (
                alpha_sharpe is not None
                and comparison_sharpe is not None
                and alpha_sharpe
                >= (
                    CORRELATION_SHARPE_MULTIPLIER
                    * comparison_sharpe
                )
            ):

                gates.append(
                    GateResult(
                        name="self_correlation",
                        status="PASS",
                        value=corr,
                        threshold=(
                            NORMAL_SELF_CORRELATION_MAX
                        ),
                        detail=(
                            "Correlation exception "
                            "satisfied by 10% Sharpe improvement."
                        ),
                    )
                )

            else:

                gates.append(
                    GateResult(
                        name="self_correlation",
                        status="FAIL",
                        value=corr,
                        threshold=(
                            NORMAL_SELF_CORRELATION_MAX
                        ),
                        detail=(
                            "Correlation exceeds the "
                            "Normal limit and the Sharpe "
                            "exception was not established."
                        ),
                    )
                )

    # --------------------------------------------------------
    # Optional/general BRAIN checks
    #
    # These are only enforced if they are present.
    # We do not manufacture missing BRAIN checks.
    # --------------------------------------------------------

    optional_checks = [
        (
            "LOW_SHARPE",
            "brain_low_sharpe",
        ),
        (
            "LOW_FITNESS",
            "brain_low_fitness",
        ),
        (
            "LOW_TURNOVER",
            "brain_low_turnover",
        ),
        (
            "HIGH_TURNOVER",
            "brain_high_turnover",
        ),
        (
            "LOW_2Y_SHARPE",
            "brain_2y_sharpe",
        ),
        (
            "PROD_CORRELATION",
            "production_correlation",
        ),
        (
            "REGULAR_SUBMISSION",
            "regular_submission",
        ),
        (
            "IS_LADDER",
            "is_ladder",
        ),
    ]

    for brain_name, local_name in optional_checks:

        result = get_test_result(
            tests,
            brain_name,
        )

        if result == UNKNOWN:
            continue

        gates.append(
            brain_test_gate(
                tests=tests,
                name=brain_name,
                display_name=local_name,
            )
        )

    return _finalize_eligibility(
        NORMAL,
        gates,
    )


# ============================================================
# POWER POOL COMPLEXITY
# ============================================================

def count_operator_occurrences(
    expression: str,
    operator_names: Iterable[str],
) -> int:
    """
    Count syntactic operator occurrences.

    This intentionally counts occurrences in the expression tree,
    including repeats, matching the Power Pool rule.

    Word boundaries prevent counting part of a longer identifier.
    """

    import re

    expression = str(
        expression
    )

    names = [
        str(name).strip()
        for name in operator_names
        if str(name).strip()
    ]

    if not names:
        return 0

    pattern = (
        r"\b(?:"
        + "|".join(
            re.escape(name)
            for name in names
        )
        + r")\s*\("
    )

    return len(
        re.findall(
            pattern,
            expression,
            flags=re.IGNORECASE,
        )
    )


def count_fields(
    fields: Iterable[str] | None,
    *,
    grouping_fields: Iterable[str] = POWER_POOL_GROUPING_FIELDS,
) -> int:
    """
    Count unique non-grouping data fields.

    `fields` should ideally come from the compiler metadata rather
    than trying to recover them from raw FASTEXPR text.
    """

    grouping = {
        str(value).strip().lower()
        for value
        in grouping_fields
    }

    unique = set()

    for field_name in (
        fields or []
    ):

        normalized = str(
            field_name
        ).strip()

        if not normalized:
            continue

        if (
            normalized.lower()
            in grouping
        ):
            continue

        unique.add(
            normalized
        )

    return len(
        unique
    )


# ============================================================
# POWER POOL EVALUATION
# ============================================================

def evaluate_power_pool(
    record: Mapping[str, Any],
    *,
    fields: Iterable[str] | None = None,
    operator_names: Iterable[str] | None = None,
) -> EligibilityResult:
    """
    Evaluate a Power Pool alpha.

    Locked criteria:
        Sharpe >= 1.00
        Fitness NOT REQUIRED
        Operators <= 8 occurrences
        Fields <= 3 non-grouping fields
        PP correlation < 0.50
          OR 10% Sharpe improvement exception
        Turnover tests PASS
        Sub-universe PASS
        Robust-universe PASS where applicable

    Important:
        Missing correlation/complexity metadata is UNKNOWN rather
        than silently accepted.
    """

    gates: list[GateResult] = []

    if (
        determine_simulation_status(record)
        != SIMULATED
    ):

        return EligibilityResult(
            alpha_type=POWER_POOL,
            status=REJECTED,
            gates=[
                GateResult(
                    name="simulation",
                    status="FAIL",
                    detail=(
                        "Simulation did not return "
                        "an alpha_id."
                    ),
                )
            ],
            failed_gates=[
                "simulation"
            ],
        )

    metrics = extract_metrics(
        record
    )

    tests = record.get(
        "is_tests"
    )

    expression = extract_expression(
        record
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    gates.append(
        metric_gate(
            name="test_sharpe",
            value=metrics.test_sharpe,
            threshold=POWER_POOL_SHARPE_MIN,
            comparator=">=",
        )
    )

    # --------------------------------------------------------
    # Fitness
    # --------------------------------------------------------

    gates.append(
        GateResult(
            name="test_fitness",
            status=NOT_REQUIRED,
            value=metrics.test_fitness,
            threshold=None,
            detail=(
                "Fitness is not a Power Pool criterion."
            ),
        )
    )

    # --------------------------------------------------------
    # Operator count
    # --------------------------------------------------------

    if operator_names is None:

        # We cannot safely infer the whole live operator catalog
        # from an arbitrary expression. The caller should provide
        # the verified operator names from operators.py.
        gates.append(
            GateResult(
                name="operator_count",
                status=UNKNOWN,
                value=None,
                threshold=(
                    POWER_POOL_MAX_OPERATOR_OCCURRENCES
                ),
                detail=(
                    "Verified operator catalog was not supplied."
                ),
            )
        )

    else:

        operator_count = (
            count_operator_occurrences(
                expression,
                operator_names,
            )
        )

        gates.append(
            GateResult(
                name="operator_count",
                status=(
                    "PASS"
                    if operator_count
                    <= POWER_POOL_MAX_OPERATOR_OCCURRENCES
                    else "FAIL"
                ),
                value=operator_count,
                threshold=(
                    POWER_POOL_MAX_OPERATOR_OCCURRENCES
                ),
                detail=(
                    f"{operator_count} <= "
                    f"{POWER_POOL_MAX_OPERATOR_OCCURRENCES}"
                ),
            )
        )

    # --------------------------------------------------------
    # Field count
    # --------------------------------------------------------

    if fields is None:

        gates.append(
            GateResult(
                name="field_count",
                status=UNKNOWN,
                value=None,
                threshold=(
                    POWER_POOL_MAX_FIELDS
                ),
                detail=(
                    "Compiler field metadata was not supplied."
                ),
            )
        )

    else:

        field_count = count_fields(
            fields
        )

        gates.append(
            GateResult(
                name="field_count",
                status=(
                    "PASS"
                    if field_count
                    <= POWER_POOL_MAX_FIELDS
                    else "FAIL"
                ),
                value=field_count,
                threshold=(
                    POWER_POOL_MAX_FIELDS
                ),
                detail=(
                    f"{field_count} <= "
                    f"{POWER_POOL_MAX_FIELDS} "
                    "non-grouping fields"
                ),
            )
        )

    # --------------------------------------------------------
    # Turnover
    # --------------------------------------------------------

    gates.append(
        turnover_gate(
            name="test_turnover",
            value=metrics.test_turnover,
        )
    )

    # --------------------------------------------------------
    # BRAIN turnover tests
    # --------------------------------------------------------

    for test_name, display_name in [
        (
            "LOW_TURNOVER",
            "turnover_low_test",
        ),
        (
            "HIGH_TURNOVER",
            "turnover_high_test",
        ),
    ]:

        result = get_test_result(
            tests,
            test_name,
        )

        if result != UNKNOWN:

            gates.append(
                brain_test_gate(
                    tests=tests,
                    name=test_name,
                    display_name=display_name,
                )
            )

    # --------------------------------------------------------
    # Sub-universe
    # --------------------------------------------------------

    gates.append(
        brain_test_gate(
            tests=tests,
            name="LOW_SUB_UNIVERSE_SHARPE",
            display_name="sub_universe",
        )
    )

    # --------------------------------------------------------
    # Robust universe
    # --------------------------------------------------------

    robust_result = get_test_result(
        tests,
        "LOW_ROBUST_UNIVERSE_SHARPE",
    )

    if robust_result == UNKNOWN:

        gates.append(
            GateResult(
                name="robust_universe",
                status=NOT_REQUIRED,
                value=None,
                threshold=None,
                detail=(
                    "Robust-universe test was not "
                    "returned by BRAIN."
                ),
            )
        )

    else:

        gates.append(
            brain_test_gate(
                tests=tests,
                name="LOW_ROBUST_UNIVERSE_SHARPE",
                display_name="robust_universe",
            )
        )

    # --------------------------------------------------------
    # Power Pool correlation
    # --------------------------------------------------------

    pp_corr = _safe_float(
        record.get(
            "power_pool_correlation",
            record.get(
                "self_correlation"
            ),
        )
    )

    correlated_sharpe = _safe_float(
        record.get(
            "most_correlated_alpha_sharpe"
        )
    )

    if pp_corr is None:

        gates.append(
            GateResult(
                name="power_pool_correlation",
                status=UNKNOWN,
                value=None,
                threshold=(
                    POWER_POOL_CORRELATION_MAX
                ),
                detail=(
                    "Power Pool correlation was not supplied."
                ),
            )
        )

    elif (
        pp_corr
        < POWER_POOL_CORRELATION_MAX
    ):

        gates.append(
            GateResult(
                name="power_pool_correlation",
                status="PASS",
                value=pp_corr,
                threshold=(
                    POWER_POOL_CORRELATION_MAX
                ),
                detail=(
                    f"{pp_corr} < "
                    f"{POWER_POOL_CORRELATION_MAX}"
                ),
            )
        )

    else:

        alpha_sharpe = metrics.test_sharpe

        if (
            alpha_sharpe is not None
            and correlated_sharpe is not None
            and alpha_sharpe
            >= (
                CORRELATION_SHARPE_MULTIPLIER
                * correlated_sharpe
            )
        ):

            gates.append(
                GateResult(
                    name="power_pool_correlation",
                    status="PASS",
                    value=pp_corr,
                    threshold=(
                        POWER_POOL_CORRELATION_MAX
                    ),
                    detail=(
                        "Correlation exception "
                        "satisfied by 10% Sharpe improvement."
                    ),
                )
            )

        else:

            gates.append(
                GateResult(
                    name="power_pool_correlation",
                    status="FAIL",
                    value=pp_corr,
                    threshold=(
                        POWER_POOL_CORRELATION_MAX
                    ),
                    detail=(
                        "Power Pool correlation is too high "
                        "and the Sharpe exception was not established."
                    ),
                )
            )

    # --------------------------------------------------------
    # Other PP-specific BRAIN checks
    # --------------------------------------------------------

    for test_name, display_name in [
        (
            "POWERPOOL_SUBMISSION",
            "power_pool_submission",
        ),
        (
            "REGULAR_SUBMISSION",
            "regular_submission",
        ),
    ]:

        result = get_test_result(
            tests,
            test_name,
        )

        if result != UNKNOWN:

            gates.append(
                brain_test_gate(
                    tests=tests,
                    name=test_name,
                    display_name=display_name,
                )
            )

    return _finalize_eligibility(
        POWER_POOL,
        gates,
    )


# ============================================================
# MASTER EVALUATOR
# ============================================================

def evaluate_alpha(
    record: Mapping[str, Any],
    *,
    alpha_type: str = NORMAL,
    fields: Iterable[str] | None = None,
    operator_names: Iterable[str] | None = None,
) -> EligibilityResult:
    """
    Evaluate one simulation result.
    """

    alpha_type = str(
        alpha_type
    ).strip().upper()

    if alpha_type == NORMAL:

        return evaluate_normal(
            record
        )

    if alpha_type == POWER_POOL:

        return evaluate_power_pool(
            record,
            fields=fields,
            operator_names=operator_names,
        )

    raise ValueError(
        "alpha_type must be NORMAL or POWER_POOL."
    )


# ============================================================
# NORMALIZED RESULT
# ============================================================

@dataclass
class NormalizedResult:
    """
    Complete normalized representation of one BRAIN result.
    """

    alpha_id: str | None

    expression: str

    alpha_type: str

    simulation_status: str

    eligibility_status: str

    metrics: PerformanceMetrics

    failed_brain_tests: list[dict[str, Any]]

    warning_brain_tests: list[dict[str, Any]]

    gates: list[GateResult]

    raw_record: dict[str, Any]

    # Generation metadata
    fields: list[str] = field(
        default_factory=list
    )

    template: str | None = None

    compiler_expression: str = ""

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Produce a flat dictionary suitable for a DataFrame.
        """

        return {
            "alpha_id": self.alpha_id,
            "expression": self.expression,
            "compiler_expression": (
                self.compiler_expression
            ),
            "alpha_type": self.alpha_type,

            "simulation_status": (
                self.simulation_status
            ),

            "eligibility_status": (
                self.eligibility_status
            ),

            "train_sharpe": (
                self.metrics.train_sharpe
            ),
            "test_sharpe": (
                self.metrics.test_sharpe
            ),

            "train_fitness": (
                self.metrics.train_fitness
            ),
            "test_fitness": (
                self.metrics.test_fitness
            ),

            "train_turnover": (
                self.metrics.train_turnover
            ),
            "test_turnover": (
                self.metrics.test_turnover
            ),

            "train_returns": (
                self.metrics.train_returns
            ),
            "test_returns": (
                self.metrics.test_returns
            ),

            "train_drawdown": (
                self.metrics.train_drawdown
            ),
            "test_drawdown": (
                self.metrics.test_drawdown
            ),

            "train_margin": (
                self.metrics.train_margin
            ),
            "test_margin": (
                self.metrics.test_margin
            ),

            "train_pnl": (
                self.metrics.train_pnl
            ),
            "test_pnl": (
                self.metrics.test_pnl
            ),

            "failed_test_count": len(
                self.failed_brain_tests
            ),

            "warning_test_count": len(
                self.warning_brain_tests
            ),

            "failed_tests": [
                x.get("name")
                for x
                in self.failed_brain_tests
            ],

            "warning_tests": [
                x.get("name")
                for x
                in self.warning_brain_tests
            ],

            "fields": list(
                self.fields
            ),

            "template": self.template,

            "gate_failures": [
                gate.name
                for gate
                in self.gates
                if gate.status == "FAIL"
            ],

            "gate_unknowns": [
                gate.name
                for gate
                in self.gates
                if gate.status == UNKNOWN
            ],
        }


# ============================================================
# NORMALIZE ONE
# ============================================================

def normalize_result(
    record: Mapping[str, Any],
    *,
    alpha_type: str = NORMAL,
    fields: Iterable[str] | None = None,
    operator_names: Iterable[str] | None = None,
    template: str | None = None,
    compiler_expression: str = "",
) -> NormalizedResult:
    """
    Normalize one raw BRAIN result and evaluate it.
    """

    raw = dict(
        record
    )

    alpha_type = str(
        alpha_type
    ).strip().upper()

    simulation_status = (
        determine_simulation_status(
            raw
        )
    )

    expression = extract_expression(
        raw
    )

    metrics = extract_metrics(
        raw
    )

    failed_tests = (
        all_failed_tests(
            raw.get(
                "is_tests"
            )
        )
    )

    warning_tests = (
        all_warning_tests(
            raw.get(
                "is_tests"
            )
        )
    )

    eligibility = evaluate_alpha(
        raw,
        alpha_type=alpha_type,
        fields=fields,
        operator_names=operator_names,
    )

    # If simulation failed, eligibility is always rejected.
    if (
        simulation_status
        != SIMULATED
    ):

        eligibility_status = (
            REJECTED
        )

    else:

        eligibility_status = (
            eligibility.status
        )

    return NormalizedResult(
        alpha_id=(
            str(
                raw.get("alpha_id")
            )
            if raw.get("alpha_id") is not None
            else None
        ),

        expression=expression,

        alpha_type=alpha_type,

        simulation_status=(
            simulation_status
        ),

        eligibility_status=(
            eligibility_status
        ),

        metrics=metrics,

        failed_brain_tests=(
            failed_tests
        ),

        warning_brain_tests=(
            warning_tests
        ),

        gates=(
            eligibility.gates
        ),

        raw_record=raw,

        fields=[
            str(value)
            for value
            in (fields or [])
        ],

        template=template,

        compiler_expression=(
            compiler_expression
        ),
    )


# ============================================================
# MANY RESULTS
# ============================================================

def normalize_results(
    records: Iterable[Mapping[str, Any]],
    *,
    alpha_type: str = NORMAL,
    fields_by_index: Iterable[Iterable[str]] | None = None,
    templates: Iterable[str | None] | None = None,
    compiler_expressions: Iterable[str] | None = None,
    operator_names: Iterable[str] | None = None,
) -> list[NormalizedResult]:
    """
    Normalize a batch of results.
    """

    records = list(
        records
    )

    n = len(
        records
    )

    if fields_by_index is None:

        fields_by_index = [
            []
            for _ in range(n)
        ]

    else:

        fields_by_index = [
            list(value)
            for value
            in fields_by_index
        ]

        if len(
            fields_by_index
        ) != n:

            raise ValueError(
                "fields_by_index length "
                "must match records length."
            )

    if templates is None:

        templates = [
            None
            for _ in range(n)
        ]

    else:

        templates = list(
            templates
        )

        if len(
            templates
        ) != n:

            raise ValueError(
                "templates length "
                "must match records length."
            )

    if compiler_expressions is None:

        compiler_expressions = [
            ""
            for _ in range(n)
        ]

    else:

        compiler_expressions = list(
            compiler_expressions
        )

        if len(
            compiler_expressions
        ) != n:

            raise ValueError(
                "compiler_expressions length "
                "must match records length."
            )

    return [
        normalize_result(
            record,
            alpha_type=alpha_type,
            fields=fields_by_index[index],
            operator_names=operator_names,
            template=templates[index],
            compiler_expression=(
                compiler_expressions[index]
            ),
        )
        for index, record
        in enumerate(records)
    ]


# ============================================================
# DATAFRAME
# ============================================================

def results_to_dataframe(
    results: Iterable[NormalizedResult],
) -> pd.DataFrame:
    """
    Convert normalized results into a flat DataFrame.
    """

    rows = [
        result.to_dict()
        for result
        in results
    ]

    return pd.DataFrame(
        rows
    )


def raw_records_to_dataframe(
    records: Iterable[Mapping[str, Any]],
    *,
    alpha_type: str = NORMAL,
    fields_by_index: Iterable[Iterable[str]] | None = None,
    templates: Iterable[str | None] | None = None,
    compiler_expressions: Iterable[str] | None = None,
    operator_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience function:

        raw BRAIN results
            ->
        normalized records
            ->
        DataFrame
    """

    normalized = normalize_results(
        records,
        alpha_type=alpha_type,
        fields_by_index=fields_by_index,
        templates=templates,
        compiler_expressions=compiler_expressions,
        operator_names=operator_names,
    )

    return results_to_dataframe(
        normalized
    )


# ============================================================
# FILTERING
# ============================================================

def eligible_results(
    results: Iterable[NormalizedResult],
) -> list[NormalizedResult]:
    """
    Return only explicitly eligible results.
    """

    return [
        result
        for result
        in results
        if result.eligibility_status
        == ELIGIBLE
    ]


def rejected_results(
    results: Iterable[NormalizedResult],
) -> list[NormalizedResult]:
    """
    Return explicitly rejected results.
    """

    return [
        result
        for result
        in results
        if result.eligibility_status
        == REJECTED
    ]


def unknown_results(
    results: Iterable[NormalizedResult],
) -> list[NormalizedResult]:
    """
    Return results where one or more required criteria are unavailable.
    """

    return [
        result
        for result
        in results
        if result.eligibility_status
        == UNKNOWN
    ]


# ============================================================
# REPORTING
# ============================================================

def print_result(
    result: NormalizedResult,
) -> None:
    """
    Print one result compactly.
    """

    print("=" * 80)
    print("ALPHA RESULT")
    print("=" * 80)

    print(
        "Alpha ID:",
        result.alpha_id,
    )

    print(
        "Alpha type:",
        result.alpha_type,
    )

    print(
        "Simulation:",
        result.simulation_status,
    )

    print(
        "Eligibility:",
        result.eligibility_status,
    )

    print()

    print(
        "Compiler expression:",
        result.compiler_expression,
    )

    print(
        "BRAIN expression:",
        result.expression,
    )

    print()

    print(
        "Train Sharpe:",
        result.metrics.train_sharpe,
    )

    print(
        "Test Sharpe:",
        result.metrics.test_sharpe,
    )

    print(
        "Train Fitness:",
        result.metrics.train_fitness,
    )

    print(
        "Test Fitness:",
        result.metrics.test_fitness,
    )

    print(
        "Train Turnover:",
        result.metrics.train_turnover,
    )

    print(
        "Test Turnover:",
        result.metrics.test_turnover,
    )

    print()

    if result.failed_brain_tests:

        print(
            "BRAIN FAILURES:"
        )

        for failure in (
            result.failed_brain_tests
        ):

            print(
                " -",
                failure.get(
                    "name"
                ),
            )

    if result.warning_brain_tests:

        print(
            "BRAIN WARNINGS:"
        )

        for warning in (
            result.warning_brain_tests
        ):

            print(
                " -",
                warning.get(
                    "name"
                ),
            )

    failures = [
        gate
        for gate
        in result.gates
        if gate.status == "FAIL"
    ]

    unknowns = [
        gate
        for gate
        in result.gates
        if gate.status == UNKNOWN
    ]

    if failures:

        print()
        print(
            "FAILED GATES:"
        )

        for gate in failures:

            print(
                " -",
                gate.name,
                ":",
                gate.detail,
            )

    if unknowns:

        print()
        print(
            "UNKNOWN GATES:"
        )

        for gate in unknowns:

            print(
                " -",
                gate.name,
                ":",
                gate.detail,
            )


def print_batch_summary(
    results: Iterable[NormalizedResult],
) -> None:
    """
    Print a compact batch summary.
    """

    results = list(
        results
    )

    simulated = sum(
        result.simulation_status
        == SIMULATED
        for result in results
    )

    simulation_errors = sum(
        result.simulation_status
        == SIMULATION_ERROR
        for result in results
    )

    eligible = sum(
        result.eligibility_status
        == ELIGIBLE
        for result in results
    )

    rejected = sum(
        result.eligibility_status
        == REJECTED
        for result in results
    )

    unknown = sum(
        result.eligibility_status
        == UNKNOWN
        for result in results
    )

    print("=" * 80)
    print("BATCH RESULT SUMMARY")
    print("=" * 80)

    print(
        "Total:",
        len(results),
    )

    print(
        "Simulated:",
        simulated,
    )

    print(
        "Simulation errors:",
        simulation_errors,
    )

    print(
        "Eligible:",
        eligible,
    )

    print(
        "Rejected:",
        rejected,
    )

    print(
        "Unknown:",
        unknown,
    )


# ============================================================
# GATE TABLE
# ============================================================

def gates_to_dataframe(
    result: NormalizedResult,
) -> pd.DataFrame:
    """
    Convert one result's individual gates into a diagnostic table.
    """

    rows = []

    for gate in result.gates:

        rows.append({
            "alpha_id": result.alpha_id,
            "alpha_type": result.alpha_type,
            "gate": gate.name,
            "status": gate.status,
            "value": gate.value,
            "threshold": gate.threshold,
            "detail": gate.detail,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SMOKE TEST HELPER
# ============================================================

def smoke_result_summary(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Evaluate the previously demonstrated smoke result without
    needing compiler metadata.
    """

    normalized = normalize_result(
        record,
        alpha_type=NORMAL,
    )

    return {
        "alpha_id": normalized.alpha_id,
        "expression": normalized.expression,
        "simulation_status": (
            normalized.simulation_status
        ),
        "eligibility_status": (
            normalized.eligibility_status
        ),
        "test_sharpe": (
            normalized.metrics.test_sharpe
        ),
        "test_fitness": (
            normalized.metrics.test_fitness
        ),
        "test_turnover": (
            normalized.metrics.test_turnover
        ),
        "failed_tests": [
            test.get("name")
            for test
            in normalized.failed_brain_tests
        ],
    }