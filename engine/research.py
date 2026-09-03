"""
Deterministic research scoring for normalized BRAIN alpha results.

This module sits ABOVE results.py eligibility logic.

Eligibility answers:
    "Does this alpha currently satisfy the submission gates?"

Research scoring answers:
    "How useful is this result for deciding what to try next?"

A rejected alpha can therefore still be PROMISING when its out-of-sample
performance is strong but a structural/robustness gate prevents submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Sequence, Any


PROMISING = "PROMISING"
WEAK = "WEAK"
FAILURE = "FAILURE"


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

TEST_SHARPE_STRONG = 1.50
TEST_SHARPE_EXCELLENT = 2.50
TEST_FITNESS_STRONG = 1.00
TEST_FITNESS_EXCELLENT = 2.50

TEST_SHARPE_FAILURE = -1.00
TEST_FITNESS_FAILURE = -1.00

TRAIN_TEST_SHARPE_GAP_WARNING = 2.00
TRAIN_TEST_FITNESS_GAP_WARNING = 3.00

MIN_TURNOVER = 0.01
MAX_TURNOVER = 0.70


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchScore:
    """Deterministic research value assigned to one normalized result."""

    alpha_id: str | None
    score: float
    research_class: str

    oos_score: float
    consistency_score: float
    turnover_score: float
    robustness_score: float

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    failed_gate_names: list[str] = field(default_factory=list)
    failed_brain_test_names: list[str] = field(default_factory=list)
    warning_brain_test_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for missing/non-finite data."""

    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(result):
        return None

    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp value into [lower, upper]."""

    return max(lower, min(upper, value))


def _names(items: Iterable[Any]) -> list[str]:
    """Extract non-empty names from mappings or objects."""

    output: list[str] = []

    for item in items:
        if isinstance(item, dict):
            value = item.get("name")
        else:
            value = getattr(item, "name", None)

        if value is not None:
            text = str(value).strip()
            if text:
                output.append(text)

    return output


def _status_for_gate(gate: Any) -> str:
    """Read status from a GateResult-like object."""

    if isinstance(gate, dict):
        return str(gate.get("status", "")).upper()

    return str(getattr(gate, "status", "")).upper()


def _name_for_gate(gate: Any) -> str:
    """Read name from a GateResult-like object."""

    if isinstance(gate, dict):
        return str(gate.get("name", ""))

    return str(getattr(gate, "name", ""))


def _brain_test_names(result: Any, attribute: str) -> list[str]:
    """Extract BRAIN test names from a normalized result."""

    return _names(getattr(result, attribute, []) or [])


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def _score_oos(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score out-of-sample performance on a 0-40 scale."""

    test_sharpe = _safe_float(getattr(metrics, "test_sharpe", None))
    test_fitness = _safe_float(getattr(metrics, "test_fitness", None))

    score = 0.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []

    if test_sharpe is None:
        weaknesses.append("Test Sharpe unavailable.")
    else:
        sharpe_points = _clamp(
            (test_sharpe + 1.0) / 3.5 * 25.0,
            0.0,
            25.0,
        )
        score += sharpe_points

        if test_sharpe >= TEST_SHARPE_EXCELLENT:
            strengths.append(f"Excellent OOS Sharpe ({test_sharpe:.2f}).")
        elif test_sharpe >= TEST_SHARPE_STRONG:
            strengths.append(f"Strong OOS Sharpe ({test_sharpe:.2f}).")
        elif test_sharpe <= TEST_SHARPE_FAILURE:
            weaknesses.append(f"Poor OOS Sharpe ({test_sharpe:.2f}).")
        else:
            weaknesses.append(f"Limited OOS Sharpe ({test_sharpe:.2f}).")

    if test_fitness is None:
        weaknesses.append("Test fitness unavailable.")
    else:
        fitness_points = _clamp(
            (test_fitness + 1.0) / 3.5 * 15.0,
            0.0,
            15.0,
        )
        score += fitness_points

        if test_fitness >= TEST_FITNESS_EXCELLENT:
            strengths.append(f"Excellent OOS fitness ({test_fitness:.2f}).")
        elif test_fitness >= TEST_FITNESS_STRONG:
            strengths.append(f"Strong OOS fitness ({test_fitness:.2f}).")
        elif test_fitness <= TEST_FITNESS_FAILURE:
            weaknesses.append(f"Poor OOS fitness ({test_fitness:.2f}).")
        else:
            weaknesses.append(f"Limited OOS fitness ({test_fitness:.2f}).")

    if (
        test_sharpe is not None
        and test_fitness is not None
        and test_sharpe >= TEST_SHARPE_STRONG
        and test_fitness >= TEST_FITNESS_STRONG
    ):
        reasons.append("Both primary OOS metrics are strong.")

    return score, strengths, weaknesses, reasons


def _score_consistency(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score train/test consistency on a 0-20 scale."""

    train_sharpe = _safe_float(getattr(metrics, "train_sharpe", None))
    test_sharpe = _safe_float(getattr(metrics, "test_sharpe", None))
    train_fitness = _safe_float(getattr(metrics, "train_fitness", None))
    test_fitness = _safe_float(getattr(metrics, "test_fitness", None))

    score = 10.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []

    if train_sharpe is None or test_sharpe is None:
        score -= 3.0
        weaknesses.append("Train/test Sharpe comparison unavailable.")
    else:
        gap = abs(train_sharpe - test_sharpe)

        if gap <= TRAIN_TEST_SHARPE_GAP_WARNING:
            score += 5.0
            strengths.append("Train/test Sharpe is reasonably consistent.")
        else:
            score -= 5.0
            weaknesses.append(f"Large train/test Sharpe gap ({gap:.2f}).")

        if train_sharpe > 0 and test_sharpe > 0:
            score += 5.0
            reasons.append("Train and test Sharpe have the same positive sign.")
        elif train_sharpe <= 0 < test_sharpe:
            weaknesses.append(
                "Test Sharpe is positive while train Sharpe is non-positive."
            )
        elif train_sharpe > 0 and test_sharpe <= 0:
            score -= 5.0
            weaknesses.append("Test Sharpe lost the positive train signal.")

    if train_fitness is None or test_fitness is None:
        score -= 2.0
        weaknesses.append("Train/test fitness comparison unavailable.")
    else:
        gap = abs(train_fitness - test_fitness)

        if gap <= TRAIN_TEST_FITNESS_GAP_WARNING:
            score += 2.5
        else:
            score -= 2.5
            weaknesses.append(f"Large train/test fitness gap ({gap:.2f}).")

    return _clamp(score, 0.0, 20.0), strengths, weaknesses, reasons


def _score_turnover(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score turnover suitability on a 0-10 scale."""

    test_turnover = _safe_float(getattr(metrics, "test_turnover", None))

    if test_turnover is None:
        return 4.0, [], ["Test turnover unavailable."], []

    if MIN_TURNOVER < test_turnover < MAX_TURNOVER:
        if test_turnover >= 0.02:
            return (
                10.0,
                [f"Turnover is comfortably investable ({test_turnover:.4f})."],
                [],
                [],
            )

        return (
            8.0,
            [f"Turnover passes the local gate ({test_turnover:.4f})."],
            [],
            [],
        )

    if test_turnover <= MIN_TURNOVER:
        return (
            2.0,
            [],
            [f"Turnover is too low ({test_turnover:.4f})."],
            ["Low turnover is often a structural/fixable issue."],
        )

    return (
        4.0,
        [],
        [f"Turnover is too high ({test_turnover:.4f})."],
        ["High turnover may be reduced by smoothing/decay."],
    )


def _score_robustness(result: Any) -> tuple[float, list[str], list[str], list[str]]:
    """
    Score BRAIN robustness on a 0-15 scale.

    Some failures are treated as fixable structural problems; broad negative
    evidence receives a larger penalty.
    """

    failed = [
        name.upper()
        for name in _brain_test_names(result, "failed_brain_tests")
    ]
    warnings = [
        name.upper()
        for name in _brain_test_names(result, "warning_brain_tests")
    ]

    score = 15.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []

    fixable_tests = {
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
        "CONCENTRATED_WEIGHT",
        "REGULAR_SUBMISSION",
        "POWERPOOL_SUBMISSION",
    }

    severe_tests = {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_2Y_SHARPE",
        "LOW_SUB_UNIVERSE_SHARPE",
        "LOW_ROBUST_UNIVERSE_SHARPE",
    }

    regional_tests = {
        "LOW_GLB_AMER_SHARPE",
        "LOW_GLB_EMEA_SHARPE",
        "LOW_GLB_APAC_SHARPE",
    }

    severe_count = sum(name in severe_tests for name in failed)
    regional_count = sum(name in regional_tests for name in failed)
    fixable_count = sum(name in fixable_tests for name in failed)

    score -= 2.5 * severe_count
    score -= 1.5 * regional_count
    score -= 0.75 * fixable_count
    score -= 0.25 * len(warnings)

    if not failed:
        strengths.append("No failed BRAIN tests.")
    else:
        if severe_count:
            weaknesses.append(
                f"{severe_count} major BRAIN robustness test(s) failed."
            )
        if regional_count:
            weaknesses.append(
                f"{regional_count} regional robustness test(s) failed."
            )
        if fixable_count:
            reasons.append("Some failed BRAIN tests are structural/fixable.")

    if warnings:
        reasons.append(f"{len(warnings)} BRAIN warning(s) recorded.")

    return _clamp(score, 0.0, 15.0), strengths, weaknesses, reasons


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(*, result: Any, score: float) -> str:
    """Classify a result without overriding BRAIN eligibility."""

    simulation_status = str(
        getattr(result, "simulation_status", "")
    ).upper()

    if simulation_status != "SIMULATED":
        return FAILURE

    metrics = getattr(result, "metrics", None)

    test_sharpe = _safe_float(getattr(metrics, "test_sharpe", None))
    test_fitness = _safe_float(getattr(metrics, "test_fitness", None))

    strong_oos = (
        (test_sharpe is not None and test_sharpe >= TEST_SHARPE_STRONG)
        or
        (test_fitness is not None and test_fitness >= TEST_FITNESS_STRONG)
    )

    clearly_bad_oos = (
        (test_sharpe is not None and test_sharpe <= TEST_SHARPE_FAILURE)
        and
        (test_fitness is not None and test_fitness <= TEST_FITNESS_FAILURE)
    )

    if clearly_bad_oos:
        return FAILURE

    if strong_oos and score >= 45.0:
        return PROMISING

    if score < 25.0:
        return FAILURE

    return WEAK


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_result(result: Any) -> ResearchScore:
    """
    Score one results.NormalizedResult.

    This function never changes eligibility_status and never treats research
    score as a replacement for results.evaluate_alpha().
    """

    metrics = getattr(result, "metrics", None)

    if metrics is None:
        return ResearchScore(
            alpha_id=getattr(result, "alpha_id", None),
            score=0.0,
            research_class=FAILURE,
            oos_score=0.0,
            consistency_score=0.0,
            turnover_score=0.0,
            robustness_score=0.0,
            weaknesses=["Performance metrics are unavailable."],
            reasons=["Unable to score a result without metrics."],
        )

    oos_score, oos_strengths, oos_weaknesses, oos_reasons = _score_oos(
        metrics
    )

    consistency_score, consistency_strengths, consistency_weaknesses, consistency_reasons = (
        _score_consistency(metrics)
    )

    turnover_score, turnover_strengths, turnover_weaknesses, turnover_reasons = (
        _score_turnover(metrics)
    )

    robustness_score, robustness_strengths, robustness_weaknesses, robustness_reasons = (
        _score_robustness(result)
    )

    total_score = _clamp(
        oos_score
        + consistency_score
        + turnover_score
        + robustness_score,
        0.0,
        85.0,
    )

    failed_gate_names = [
        _name_for_gate(gate)
        for gate in (getattr(result, "gates", []) or [])
        if _status_for_gate(gate) == "FAIL"
    ]

    failed_brain_test_names = _brain_test_names(
        result,
        "failed_brain_tests",
    )

    warning_brain_test_names = _brain_test_names(
        result,
        "warning_brain_tests",
    )

    strengths = (
        oos_strengths
        + consistency_strengths
        + turnover_strengths
        + robustness_strengths
    )

    weaknesses = (
        oos_weaknesses
        + consistency_weaknesses
        + turnover_weaknesses
        + robustness_weaknesses
    )

    reasons = (
        oos_reasons
        + consistency_reasons
        + turnover_reasons
        + robustness_reasons
    )

    eligibility_status = str(
        getattr(result, "eligibility_status", "")
    ).upper()

    if eligibility_status == "ELIGIBLE":
        reasons.append("Current eligibility gates are satisfied.")
    elif eligibility_status == "REJECTED":
        reasons.append(
            "Alpha is rejected for submission but remains evaluated for research value."
        )
    elif eligibility_status:
        reasons.append(f"Current eligibility status: {eligibility_status}.")

    research_class = _classify(
        result=result,
        score=total_score,
    )

    return ResearchScore(
        alpha_id=getattr(result, "alpha_id", None),
        score=round(total_score, 2),
        research_class=research_class,
        oos_score=round(oos_score, 2),
        consistency_score=round(consistency_score, 2),
        turnover_score=round(turnover_score, 2),
        robustness_score=round(robustness_score, 2),
        strengths=strengths,
        weaknesses=weaknesses,
        reasons=reasons,
        failed_gate_names=failed_gate_names,
        failed_brain_test_names=failed_brain_test_names,
        warning_brain_test_names=warning_brain_test_names,
    )


def score_batch(
    results: Sequence[Any] | Iterable[Any],
) -> list[ResearchScore]:
    """Score and rank a batch of normalized results."""

    scored = [
        score_result(result)
        for result in results
    ]

    def sort_key(item: ResearchScore) -> tuple[float, float]:
        return (
            item.score,
            item.oos_score,
        )

    return sorted(
        scored,
        key=sort_key,
        reverse=True,
    )


def scores_to_dataframe(
    scores: Sequence[ResearchScore] | Iterable[ResearchScore],
):
    """Convert ResearchScore objects to a pandas DataFrame."""

    import pandas as pd

    rows = []

    for score in scores:
        rows.append(
            {
                "alpha_id": score.alpha_id,
                "research_score": score.score,
                "research_class": score.research_class,
                "oos_score": score.oos_score,
                "consistency_score": score.consistency_score,
                "turnover_score": score.turnover_score,
                "robustness_score": score.robustness_score,
                "strengths": " | ".join(score.strengths),
                "weaknesses": " | ".join(score.weaknesses),
                "reasons": " | ".join(score.reasons),
                "failed_gates": " | ".join(score.failed_gate_names),
                "failed_brain_tests": " | ".join(
                    score.failed_brain_test_names
                ),
                "warning_brain_tests": " | ".join(
                    score.warning_brain_test_names
                ),
            }
        )

    return pd.DataFrame(rows)


def print_score(score: ResearchScore) -> None:
    """Print one research score in a compact human-readable form."""

    print("=" * 80)
    print("RESEARCH SCORE")
    print("=" * 80)
    print(f"Alpha:             {score.alpha_id}")
    print(f"Score:             {score.score:.2f} / 85")
    print(f"Class:             {score.research_class}")
    print(f"OOS:               {score.oos_score:.2f} / 40")
    print(f"Consistency:       {score.consistency_score:.2f} / 20")
    print(f"Turnover:          {score.turnover_score:.2f} / 10")
    print(f"Robustness:        {score.robustness_score:.2f} / 15")

    if score.strengths:
        print("\nStrengths:")
        for item in score.strengths:
            print(f"  + {item}")

    if score.weaknesses:
        print("\nWeaknesses:")
        for item in score.weaknesses:
            print(f"  - {item}")

    if score.reasons:
        print("\nResearch notes:")
        for item in score.reasons:
            print(f"  * {item}")
