"""
Deterministic IS-first research scoring for normalized BRAIN alpha results.

Research scoring is deliberately separate from BRAIN eligibility.  The
primary objective is to identify alphas that have credible, robust
In-Sample performance.  The BRAIN Test Period is treated as an IS stability
check, not as true OOS.  True OOS is unavailable until an Alpha is submitted
for OOS testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Sequence, Any

PROMISING = "PROMISING"
WEAK = "WEAK"
FAILURE = "FAILURE"

# ------------------------- scoring weights -------------------------
IS_PERFORMANCE_MAX = 35.0
IS_ROBUSTNESS_MAX = 30.0
IS_STABILITY_MAX = 15.0
TURNOVER_MAX = 10.0
TEST_PERIOD_MAX = 10.0
TOTAL_MAX = 100.0

# BRAIN/consultant-style practical reference points.  These are scoring
# anchors, not substitutes for BRAIN's own eligibility rules.
TRAIN_SHARPE_GOOD = 1.25
TRAIN_SHARPE_EXCELLENT = 2.00
TRAIN_FITNESS_GOOD = 1.00
TRAIN_FITNESS_EXCELLENT = 2.00
TEST_SHARPE_GOOD = 1.25
TEST_FITNESS_GOOD = 1.00
TRAIN_SHARPE_HARD_FAIL = 0.0
TRAIN_FITNESS_HARD_FAIL = 0.0
MIN_TURNOVER = 0.01
MAX_TURNOVER = 0.70

# Large train/test-period disagreement is a strong overfitting warning.
SHARPE_GAP_WARNING = 1.50
SHARPE_GAP_SEVERE = 2.50
FITNESS_GAP_WARNING = 2.00
FITNESS_GAP_SEVERE = 4.00


@dataclass(frozen=True)
class ResearchScore:
    alpha_id: str | None
    score: float
    research_class: str

    # Compatibility fields.  test_period_score is now the primary label;
    # oos_score is retained for older UI/code and mirrors Test Period score.
    oos_score: float
    consistency_score: float
    turnover_score: float
    robustness_score: float

    # New explicit IS-first components.
    is_performance_score: float = 0.0
    is_robustness_score: float = 0.0
    is_stability_score: float = 0.0
    test_period_score: float = 0.0
    true_oos_score: float = 0.0

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    failed_gate_names: list[str] = field(default_factory=list)
    failed_brain_test_names: list[str] = field(default_factory=list)
    warning_brain_test_names: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if isfinite(x) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _names(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("name")
        else:
            value = getattr(item, "name", None)
        if value is not None and str(value).strip():
            out.append(str(value).strip())
    return out


def _status_for_gate(gate: Any) -> str:
    if isinstance(gate, dict):
        return str(gate.get("status", "")).upper()
    return str(getattr(gate, "status", "")).upper()


def _name_for_gate(gate: Any) -> str:
    if isinstance(gate, dict):
        return str(gate.get("name", ""))
    return str(getattr(gate, "name", ""))


def _brain_test_names(result: Any, attribute: str) -> list[str]:
    return _names(getattr(result, attribute, []) or [])


def _raw_branch(result: Any, branch: str, subbranch: str | None = None) -> dict[str, Any]:
    raw = getattr(result, "raw_record", None)
    if not isinstance(raw, dict):
        return {}
    node = raw.get(branch)
    if subbranch is not None and isinstance(node, dict):
        node = node.get(subbranch)
    return node if isinstance(node, dict) else {}


def _metric_value(data: dict[str, Any], key: str) -> float | None:
    return _safe_float(data.get(key))


def _score_train_performance(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score the main In-Sample train performance on 35 points."""
    sharpe = _safe_float(getattr(metrics, "train_sharpe", None))
    fitness = _safe_float(getattr(metrics, "train_fitness", None))
    returns = _safe_float(getattr(metrics, "train_returns", None))
    drawdown = _safe_float(getattr(metrics, "train_drawdown", None))
    margin = _safe_float(getattr(metrics, "train_margin", None))

    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []
    score = 0.0

    if sharpe is None:
        weaknesses.append("Train Sharpe unavailable.")
    else:
        score += _clamp((sharpe + 0.5) / 2.5 * 20.0, 0.0, 20.0)
        if sharpe >= TRAIN_SHARPE_EXCELLENT:
            strengths.append(f"Excellent In-Sample train Sharpe ({sharpe:.2f}).")
        elif sharpe >= TRAIN_SHARPE_GOOD:
            strengths.append(f"Good In-Sample train Sharpe ({sharpe:.2f}).")
        elif sharpe <= TRAIN_SHARPE_HARD_FAIL:
            weaknesses.append(f"In-Sample train Sharpe is non-positive ({sharpe:.2f}).")
        else:
            weaknesses.append(f"In-Sample train Sharpe is weak ({sharpe:.2f}).")

    if fitness is None:
        weaknesses.append("Train fitness unavailable.")
    else:
        score += _clamp((fitness + 0.5) / 2.5 * 10.0, 0.0, 10.0)
        if fitness >= TRAIN_FITNESS_EXCELLENT:
            strengths.append(f"Excellent In-Sample train fitness ({fitness:.2f}).")
        elif fitness >= TRAIN_FITNESS_GOOD:
            strengths.append(f"Good In-Sample train fitness ({fitness:.2f}).")
        elif fitness <= TRAIN_FITNESS_HARD_FAIL:
            weaknesses.append(f"In-Sample train fitness is non-positive ({fitness:.2f}).")

    aux = 0.0
    if returns is not None and returns > 0:
        aux += min(2.0, returns * 20.0)
    if margin is not None and margin > 0:
        aux += min(1.0, margin * 200.0)
    if drawdown is not None:
        if drawdown <= 0.10:
            aux += 2.0
        elif drawdown <= 0.20:
            aux += 1.0
    score += _clamp(aux, 0.0, 5.0)

    return _clamp(score, 0.0, IS_PERFORMANCE_MAX), strengths, weaknesses, reasons


def _score_is_robustness(result: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score the complete BRAIN IS robustness evidence on 30 points."""
    failed = [x.upper() for x in _brain_test_names(result, "failed_brain_tests")]
    warnings = [x.upper() for x in _brain_test_names(result, "warning_brain_tests")]

    severe = {
        "LOW_SHARPE", "LOW_FITNESS", "LOW_2Y_SHARPE",
        "LOW_SUB_UNIVERSE_SHARPE", "LOW_ROBUST_UNIVERSE_SHARPE",
    }
    regional = {"LOW_GLB_AMER_SHARPE", "LOW_GLB_EMEA_SHARPE", "LOW_GLB_APAC_SHARPE"}
    structural = {"CONCENTRATED_WEIGHT", "LOW_TURNOVER", "HIGH_TURNOVER", "REGULAR_SUBMISSION", "POWERPOOL_SUBMISSION"}

    score = 30.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []

    severe_count = sum(x in severe for x in failed)
    regional_count = sum(x in regional for x in failed)
    structural_count = sum(x in structural for x in failed)

    score -= 4.0 * severe_count
    score -= 2.0 * regional_count
    score -= 1.0 * structural_count
    score -= 0.25 * len(warnings)

    if not failed:
        strengths.append("No failed BRAIN robustness tests in the returned IS evidence.")
    else:
        if severe_count:
            weaknesses.append(f"{severe_count} major BRAIN IS test(s) failed.")
        if regional_count:
            weaknesses.append(f"{regional_count} regional IS robustness test(s) failed.")
        if structural_count:
            reasons.append("Some failed BRAIN tests are structural/fixable.")

    # Complete returned train branches are genuine IS diagnostics.
    train = _raw_branch(result, "train")
    rn = train.get("riskNeutralized") if isinstance(train, dict) else None
    ic = train.get("investabilityConstrained") if isinstance(train, dict) else None
    base_sharpe = _metric_value(train, "sharpe") if train else None

    if isinstance(rn, dict) and base_sharpe is not None:
        rn_sharpe = _metric_value(rn, "sharpe")
        if rn_sharpe is not None:
            delta = rn_sharpe - base_sharpe
            if rn_sharpe > 0 and delta >= -0.25:
                score += 1.5
                strengths.append(f"Train risk-neutralized Sharpe survives well ({rn_sharpe:.2f}).")
            elif delta <= -0.75:
                score -= 1.5
                weaknesses.append(f"Train risk-neutralized Sharpe drops materially ({rn_sharpe:.2f}).")

    if isinstance(ic, dict) and base_sharpe is not None:
        ic_sharpe = _metric_value(ic, "sharpe")
        if ic_sharpe is not None:
            delta = ic_sharpe - base_sharpe
            if ic_sharpe > 0 and delta >= -0.25:
                score += 1.5
                strengths.append(f"Train investability-constrained Sharpe survives well ({ic_sharpe:.2f}).")
            elif delta <= -0.75:
                score -= 1.5
                weaknesses.append(f"Train investability-constrained Sharpe drops materially ({ic_sharpe:.2f}).")

    return _clamp(score, 0.0, IS_ROBUSTNESS_MAX), strengths, weaknesses, reasons


def _score_is_stability(metrics: Any, result: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score train vs BRAIN Test Period stability on 15 points."""
    train_sharpe = _safe_float(getattr(metrics, "train_sharpe", None))
    test_sharpe = _safe_float(getattr(metrics, "test_sharpe", None))
    train_fitness = _safe_float(getattr(metrics, "train_fitness", None))
    test_fitness = _safe_float(getattr(metrics, "test_fitness", None))

    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []
    score = 7.5

    if train_sharpe is not None and test_sharpe is not None:
        gap = abs(train_sharpe - test_sharpe)
        if train_sharpe > 0 and test_sharpe > 0:
            score += 4.0
            strengths.append("Train and Test Period Sharpe have the same positive sign.")
        elif train_sharpe <= 0 < test_sharpe:
            score -= 4.0
            weaknesses.append("Test Period Sharpe is positive while train Sharpe is non-positive.")
        elif train_sharpe > 0 >= test_sharpe:
            score -= 4.0
            weaknesses.append("Test Period Sharpe loses the positive train signal.")

        if gap <= SHARPE_GAP_WARNING:
            score += 2.5
        elif gap >= SHARPE_GAP_SEVERE:
            score -= 4.0
            weaknesses.append(f"Severe train/Test Period Sharpe gap ({gap:.2f}).")
        else:
            score -= 1.5
            weaknesses.append(f"Large train/Test Period Sharpe gap ({gap:.2f}).")
    else:
        score -= 2.0
        weaknesses.append("Train/Test Period Sharpe comparison unavailable.")

    if train_fitness is not None and test_fitness is not None:
        gap = abs(train_fitness - test_fitness)
        if gap <= FITNESS_GAP_WARNING:
            score += 1.0
        elif gap >= FITNESS_GAP_SEVERE:
            score -= 2.0
            weaknesses.append(f"Severe train/Test Period fitness gap ({gap:.2f}).")
        else:
            score -= 0.5
    else:
        score -= 1.0

    failed = {x.upper() for x in _brain_test_names(result, "failed_brain_tests")}
    if "LOW_2Y_SHARPE" in failed:
        score -= 2.0
        weaknesses.append("BRAIN reports weak 2Y In-Sample Sharpe.")
    else:
        reasons.append("No LOW_2Y_SHARPE failure was returned.")

    return _clamp(score, 0.0, IS_STABILITY_MAX), strengths, weaknesses, reasons


def _score_turnover(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    turnover = _safe_float(getattr(metrics, "train_turnover", None))
    if turnover is None:
        return 4.0, [], ["Train turnover unavailable."], []
    if MIN_TURNOVER < turnover < MAX_TURNOVER:
        if turnover >= 0.02:
            return 10.0, [f"Train turnover is comfortably investable ({turnover:.4f})."], [], []
        return 8.0, [f"Train turnover passes the local gate ({turnover:.4f})."], [], []
    if turnover <= MIN_TURNOVER:
        return 3.0, [], [f"Train turnover is too low ({turnover:.4f})."], ["Low turnover may signal an overly static signal."]
    return 2.0, [], [f"Train turnover is too high ({turnover:.4f})."], ["High turnover may be reduced by smoothing/decay."]


def _score_test_period(metrics: Any) -> tuple[float, list[str], list[str], list[str]]:
    """Score BRAIN Test Period strength on only 10 points."""
    sharpe = _safe_float(getattr(metrics, "test_sharpe", None))
    fitness = _safe_float(getattr(metrics, "test_fitness", None))
    score = 0.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasons: list[str] = []

    if sharpe is not None:
        score += _clamp((sharpe + 0.5) / 2.5 * 6.0, 0.0, 6.0)
        if sharpe >= 2.0:
            strengths.append(f"Strong Test Period Sharpe ({sharpe:.2f}).")
        elif sharpe <= 0:
            weaknesses.append(f"Test Period Sharpe is non-positive ({sharpe:.2f}).")
    else:
        weaknesses.append("Test Period Sharpe unavailable.")

    if fitness is not None:
        score += _clamp((fitness + 0.5) / 2.5 * 4.0, 0.0, 4.0)
        if fitness >= 2.0:
            strengths.append(f"Strong Test Period fitness ({fitness:.2f}).")
        elif fitness <= 0:
            weaknesses.append(f"Test Period fitness is non-positive ({fitness:.2f}).")
    else:
        weaknesses.append("Test Period fitness unavailable.")

    reasons.append("Test Period is treated as an In-Sample stability check, not true OOS.")
    return _clamp(score, 0.0, TEST_PERIOD_MAX), strengths, weaknesses, reasons


def _classify(*, result: Any, score: float, is_performance: float, is_robustness: float, is_stability: float) -> str:
    simulation_status = str(getattr(result, "simulation_status", "")).upper()
    if simulation_status != "SIMULATED":
        return FAILURE

    metrics = getattr(result, "metrics", None)
    train_sharpe = _safe_float(getattr(metrics, "train_sharpe", None))
    train_fitness = _safe_float(getattr(metrics, "train_fitness", None))

    # Hard IS sanity gate: a huge Test Period result cannot rescue an alpha
    # whose main train-period performance is non-positive.
    if (train_sharpe is not None and train_fitness is not None
            and train_sharpe <= 0 and train_fitness <= 0):
        return FAILURE

    if is_performance < 12.0:
        return FAILURE
    if is_robustness < 14.0:
        return WEAK
    if is_stability < 5.0:
        return WEAK
    if score >= 70.0:
        return PROMISING
    if score < 40.0:
        return FAILURE
    return WEAK


def score_result(result: Any) -> ResearchScore:
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

    is_perf, a, b, c = _score_train_performance(metrics)
    is_rob, d, e, f = _score_is_robustness(result)
    is_stab, g, h, i = _score_is_stability(metrics, result)
    turnover, j, k, l = _score_turnover(metrics)
    test_period, m, n, o = _score_test_period(metrics)

    total = _clamp(is_perf + is_rob + is_stab + turnover + test_period, 0.0, TOTAL_MAX)

    failed_gate_names = [
        _name_for_gate(gate)
        for gate in (getattr(result, "gates", []) or [])
        if _status_for_gate(gate) == "FAIL"
    ]
    failed_brain = _brain_test_names(result, "failed_brain_tests")
    warnings = _brain_test_names(result, "warning_brain_tests")

    strengths = a + d + g + j + m
    weaknesses = b + e + h + k + n
    reasons = c + f + i + l + o

    eligibility = str(getattr(result, "eligibility_status", "")).upper()
    if eligibility == "ELIGIBLE":
        reasons.append("Current BRAIN eligibility gates are satisfied.")
    elif eligibility == "REJECTED":
        reasons.append("Alpha is rejected for submission but is retained for research diagnostics.")

    research_class = _classify(
        result=result,
        score=total,
        is_performance=is_perf,
        is_robustness=is_rob,
        is_stability=is_stab,
    )

    return ResearchScore(
        alpha_id=getattr(result, "alpha_id", None),
        score=round(total, 2),
        research_class=research_class,
        oos_score=round(test_period, 2),
        consistency_score=round(is_stab, 2),
        turnover_score=round(turnover, 2),
        robustness_score=round(is_rob, 2),
        is_performance_score=round(is_perf, 2),
        is_robustness_score=round(is_rob, 2),
        is_stability_score=round(is_stab, 2),
        test_period_score=round(test_period, 2),
        true_oos_score=0.0,
        strengths=strengths,
        weaknesses=weaknesses,
        reasons=reasons,
        failed_gate_names=failed_gate_names,
        failed_brain_test_names=failed_brain,
        warning_brain_test_names=warnings,
    )


def score_batch(results: Sequence[Any] | Iterable[Any]) -> list[ResearchScore]:
    scored = [score_result(result) for result in results]
    return sorted(
        scored,
        key=lambda x: (x.score, x.is_performance_score, x.is_robustness_score),
        reverse=True,
    )


def scores_to_dataframe(scores: Sequence[ResearchScore] | Iterable[ResearchScore]):
    import pandas as pd
    rows = []
    for score in scores:
        rows.append({
            "alpha_id": score.alpha_id,
            "research_score": score.score,
            "research_class": score.research_class,
            "is_performance": score.is_performance_score,
            "is_robustness": score.is_robustness_score,
            "is_stability": score.is_stability_score,
            "test_period": score.test_period_score,
            "turnover": score.turnover_score,
            "true_oos": score.true_oos_score,
            "oos_score": score.oos_score,
            "consistency_score": score.consistency_score,
            "turnover_score": score.turnover_score,
            "robustness_score": score.robustness_score,
            "strengths": " | ".join(score.strengths),
            "weaknesses": " | ".join(score.weaknesses),
            "reasons": " | ".join(score.reasons),
            "failed_gates": " | ".join(score.failed_gate_names),
            "failed_brain_tests": " | ".join(score.failed_brain_test_names),
            "warning_brain_tests": " | ".join(score.warning_brain_test_names),
        })
    return pd.DataFrame(rows)


def print_score(score: ResearchScore) -> None:
    print("=" * 80)
    print("RESEARCH SCORE — IS FIRST")
    print("=" * 80)
    print(f"Alpha:              {score.alpha_id}")
    print(f"Score:              {score.score:.2f} / {TOTAL_MAX:.0f}")
    print(f"Class:              {score.research_class}")
    print(f"IS performance:     {score.is_performance_score:.2f} / {IS_PERFORMANCE_MAX:.0f}")
    print(f"IS robustness:      {score.is_robustness_score:.2f} / {IS_ROBUSTNESS_MAX:.0f}")
    print(f"IS stability:       {score.is_stability_score:.2f} / {IS_STABILITY_MAX:.0f}")
    print(f"Turnover:           {score.turnover_score:.2f} / {TURNOVER_MAX:.0f}")
    print(f"Test Period:        {score.test_period_score:.2f} / {TEST_PERIOD_MAX:.0f}")
    print(f"True OOS:           {score.true_oos_score:.2f} / 0 (not available pre-submission)")
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
