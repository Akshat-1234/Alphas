"""
Evidence-constrained LLM research analyst for the alpha-research loop.

Design goals
------------
The LLM is a research explainer and hypothesis generator, not an authority on
empirical truth. Deterministic research evidence is computed from persistent
experiment memory and is authoritative for:

    * experiment class: PROMISING / WEAK / FAILURE
    * OOS-signal status
    * robustness status
    * research-readiness
    * whether an item has enough evidence to recommend or avoid

The analyst never generates FASTEXPR and never decides BRAIN eligibility.

This module deliberately separates four evidence states:

    FAILURE       negative evidence
    WEAK          insufficient evidence
    OOS_SIGNAL    promising OOS result, but not robust/readiness-proven
    READY         promising + passes deterministic research-readiness checks

The public compatibility fields ``promising_patterns`` and
``recommended_*`` are retained, but the printer uses the more precise term
``Research leads`` for signal-only patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .research_memory import ResearchMemory, memory_for_analyst


INSIGHT_SCHEMA_VERSION = 6

_VALID_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}
_VALID_DIRECTIONS = {"positive", "negative"}
_VALID_CLASSES = {"PROMISING", "WEAK", "FAILURE"}
_VALID_TIERS = {"FAILURE", "WEAK", "OOS_SIGNAL", "READY"}

# These are empirical research-engine heuristics, not official BRAIN rules.
# Official simulation eligibility remains owned by engine.results.
_MIN_READY_ROBUSTNESS_SCORE = 10.0
_MIN_READY_SUPPORTING_EXPERIMENTS = 1
_MAX_READY_FAILED_BRAIN_TESTS = 0


@dataclass
class ResearchInsight:
    """Structured output produced by the research analyst."""

    schema_version: int = INSIGHT_SCHEMA_VERSION
    summary: str = ""

    # Compatibility name retained; semantically these are research leads.
    promising_patterns: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    fixable_patterns: list[str] = field(default_factory=list)

    recommended_templates: list[str] = field(default_factory=list)
    recommended_fields: list[str] = field(default_factory=list)
    recommended_directions: list[str] = field(default_factory=list)

    avoid_templates: list[str] = field(default_factory=list)
    avoid_fields: list[str] = field(default_factory=list)

    next_experiments: list[dict[str, Any]] = field(default_factory=list)

    confidence: str = "LOW"
    validation_warnings: list[str] = field(default_factory=list)
    raw_response: str = ""

    # New explicit deterministic evidence views.
    evidence_tiers: dict[str, Any] = field(default_factory=dict)


class ResearchAnalyst:
    """
    Analyst wrapper around the existing local LLM interface.

    ``live_templates`` must contain actual compiler template names.
    ``live_fields`` must contain actual live BRAIN field IDs.
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_memory_records: int = 25,
        live_templates: Iterable[str] | None = None,
        live_fields: Iterable[str] | None = None,
    ) -> None:
        self.llm = llm
        self.max_memory_records = max_memory_records
        self.live_templates = _normalize_set(live_templates)
        self.live_fields = _normalize_set(live_fields)

    def build_context(self, memory: ResearchMemory) -> dict[str, Any]:
        """Build deterministic context from persistent research memory."""

        context = memory_for_analyst(
            memory,
            latest_limit=self.max_memory_records,
            top_limit=10,
        )

        context["constraints"] = {
            "live_templates": sorted(self.live_templates),
            "live_fields": sorted(self.live_fields),
            "valid_directions": sorted(_VALID_DIRECTIONS),
            "evidence_tiers": sorted(_VALID_TIERS),
        }
        context["evidence"] = _build_evidence_summary(context)
        context["evidence_digest"] = _build_evidence_digest(context["evidence"])
        return context

    def build_prompt(self, context: dict[str, Any]) -> str:
        """Build an evidence-constrained analyst prompt."""

        serialized_context = json.dumps(
            context,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        return f"""
You are a quantitative alpha research analyst.

Analyze the empirical alpha experiments in the supplied research memory and
propose small, testable research directions for the next candidate cycle.

ROLE BOUNDARIES
1. You explain evidence and propose hypotheses. You do not define empirical
   truth yourself.
2. The deterministic evidence section is authoritative.
3. Do not generate FASTEXPR or executable expressions.
4. Do not decide BRAIN submission eligibility.

HARD RULES
1. Use only field IDs listed in constraints.live_fields.
2. Use only compiler template names listed in constraints.live_templates.
3. NEVER use alpha types such as NORMAL or POWER_POOL as templates.
4. Directions must be exactly positive or negative.
5. Never call an OOS-only signal robust, established, validated, or research-
   ready.
6. PROMISING means "research lead" only. It does NOT mean robust or ready.
7. WEAK is not positive evidence.
8. FAILURE is negative evidence unless a next experiment explicitly proposes
   fixing the recorded failure.
9. A field recommendation requires deterministic READY evidence for that field.
10. A template recommendation requires deterministic READY evidence for that
    template.
11. Avoid recommendations require clear deterministic failure evidence and
    must not be contradicted by positive evidence.
12. Do not generalize a field's quality from a single experiment merely because
    the experiment had a positive OOS metric.
13. If there are no READY templates or fields, say that explicitly.
14. Research leads may be discussed even when not READY, but label them as
    OOS signals that require validation.
15. Every next experiment must either:
      a) build on READY evidence, or
      b) explicitly target a known weakness of an OOS signal / failure.
16. A next experiment that merely recombines fields without naming the failure
    it is intended to address is invalid.
17. Keep next_experiments at most 5 and prefer diverse, testable ideas.
18. Return JSON only.

TERMINOLOGY
- Research lead = deterministic OOS signal that is not yet READY.
- READY = deterministic evidence sufficient to support a research recommendation.
- Failure pattern = negative evidence, preferably failure-dominant.
- Fixable pattern = an observed weakness/failure for which a plausible repair
  experiment exists. This is a hypothesis, not proof.

IMPORTANT
The deterministic evidence_digest below is more trustworthy than prose in
memory. When the two appear to conflict, follow the deterministic digest.

Available context:
{serialized_context}

Return exactly one JSON object with these keys:
{{
  "summary": "brief evidence-based synthesis",
  "promising_patterns": ["research lead labels only"],
  "failure_patterns": ["deterministically supported failure labels only"],
  "fixable_patterns": ["observed, repairable weaknesses"],
  "recommended_templates": ["READY live compiler templates only"],
  "recommended_fields": ["READY live BRAIN field IDs only"],
  "recommended_directions": ["positive or negative"],
  "avoid_templates": ["live templates with clear failure evidence"],
  "avoid_fields": ["live fields with clear failure evidence"],
  "next_experiments": [
    {{
      "template": "existing live compiler template",
      "fields": ["existing live BRAIN field id", "..."],
      "direction": "positive or negative",
      "reason": "specific observed evidence or failure being tested/fixed"
    }}
  ],
  "confidence": "LOW | MEDIUM | HIGH"
}}

Never invent a field, template, or evidence claim.
""".strip()

    def analyze(self, memory: ResearchMemory) -> ResearchInsight:
        """Run the LLM analyst, then deterministically validate its output."""

        context = self.build_context(memory)
        raw = _call_llm_json(self.llm, self.build_prompt(context))
        insight = _parse_insight(raw)
        insight.raw_response = json.dumps(raw, ensure_ascii=False)

        warnings = validate_insight(
            insight,
            context=context,
            live_templates=self.live_templates,
            live_fields=self.live_fields,
        )

        # Final deterministic normalization. This means prose/labels cannot
        # contradict the evidence matrix even if the LLM does.
        normalize_warnings = _normalize_insight_to_evidence(
            insight,
            evidence=context.get("evidence", {}),
        )
        warnings.extend(normalize_warnings)

        insight.evidence_tiers = context.get("evidence", {}).get("tiers", {})
        insight.validation_warnings = _dedupe(warnings)

        # Confidence is determined by deterministic evidence, not by an LLM
        # string. This is intentionally conservative with only five records.
        insight.confidence = _deterministic_confidence(context.get("evidence", {}))
        return insight


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


def _call_llm_json(llm: Any, prompt: str) -> dict[str, Any]:
    """Adapt to the local LLM interface."""

    if hasattr(llm, "generate_json"):
        result: Any = llm.generate_json(prompt)
    elif hasattr(llm, "generate"):
        result = llm.generate(prompt)
    elif hasattr(llm, "chat"):
        result = llm.chat(prompt)
    else:
        raise TypeError(
            "LLM object must provide generate_json(), generate(), or chat()."
        )

    if isinstance(result, dict):
        return result

    for attribute in ("content", "text", "response"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return _parse_json_text(value)

    if isinstance(result, str):
        return _parse_json_text(result)

    raise TypeError(f"Unsupported LLM response type: {type(result).__name__}")


def _parse_json_text(text: str) -> dict[str, Any]:
    """Parse strict JSON, with a small fenced-JSON fallback."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Analyst LLM returned non-JSON text.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Analyst LLM JSON response must be an object.")

    return parsed


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _string_list(value: Any) -> list[str]:
    """Normalize a value into a clean stable string list."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _experiment_list(value: Any) -> list[dict[str, Any]]:
    """Normalize next-experiment objects without trusting their contents."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "template": str(item.get("template", "")).strip(),
                "fields": _string_list(item.get("fields", [])),
                "direction": str(item.get("direction", "")).strip().lower(),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return result


def _parse_insight(payload: dict[str, Any]) -> ResearchInsight:
    """Convert LLM JSON into a normalized ResearchInsight."""

    confidence = str(payload.get("confidence", "LOW")).strip().upper()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "LOW"

    return ResearchInsight(
        schema_version=INSIGHT_SCHEMA_VERSION,
        summary=str(payload.get("summary", "")).strip(),
        promising_patterns=_string_list(payload.get("promising_patterns", [])),
        failure_patterns=_string_list(payload.get("failure_patterns", [])),
        fixable_patterns=_string_list(payload.get("fixable_patterns", [])),
        recommended_templates=_string_list(payload.get("recommended_templates", [])),
        recommended_fields=_string_list(payload.get("recommended_fields", [])),
        recommended_directions=_string_list(payload.get("recommended_directions", [])),
        avoid_templates=_string_list(payload.get("avoid_templates", [])),
        avoid_fields=_string_list(payload.get("avoid_fields", [])),
        next_experiments=_experiment_list(payload.get("next_experiments", [])),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------


def validate_insight(
    insight: ResearchInsight,
    *,
    context: Mapping[str, Any] | None = None,
    live_templates: Iterable[str] | None = None,
    live_fields: Iterable[str] | None = None,
) -> list[str]:
    """Validate and sanitize an LLM insight against deterministic evidence."""

    templates = _normalize_set(live_templates)
    fields = _normalize_set(live_fields)
    warnings: list[str] = []

    if templates:
        insight.recommended_templates, warn = _sanitize_live_list(
            insight.recommended_templates, templates, "recommended template"
        )
        warnings.extend(warn)
        insight.avoid_templates, warn = _sanitize_live_list(
            insight.avoid_templates, templates, "avoid template"
        )
        warnings.extend(warn)

    if fields:
        insight.recommended_fields, warn = _sanitize_live_list(
            insight.recommended_fields, fields, "recommended field"
        )
        warnings.extend(warn)
        insight.avoid_fields, warn = _sanitize_live_list(
            insight.avoid_fields, fields, "avoid field"
        )
        warnings.extend(warn)

    insight.next_experiments, warn = _validate_experiments(
        insight.next_experiments,
        templates=templates,
        fields=fields,
    )
    warnings.extend(warn)

    evidence = _build_evidence_summary(context or {})
    tiers = evidence["tiers"]

    # Only READY items can survive as recommendations.
    ready_templates = set(tiers["research_ready_templates"])
    ready_fields = set(tiers["research_ready_fields"])

    original_templates = insight.recommended_templates[:]
    insight.recommended_templates = [
        item for item in original_templates if item in ready_templates
    ]
    for item in original_templates:
        if item not in ready_templates:
            warnings.append(
                f"Removed research-ready template recommendation: {item}; "
                f"deterministic tier is {tiers['template_tiers'].get(item, 'UNOBSERVED')}."
            )

    original_fields = insight.recommended_fields[:]
    insight.recommended_fields = [
        item for item in original_fields if item in ready_fields
    ]
    for item in original_fields:
        if item not in ready_fields:
            warnings.append(
                f"Removed research-ready field recommendation: {item}; "
                f"deterministic tier is {tiers['field_tiers'].get(item, 'UNOBSERVED')}."
            )

    # Avoid only on clear failure, and never when a READY/OOS signal counters it.
    insight.avoid_templates, warn = _filter_avoid_recommendations(
        insight.avoid_templates,
        evidence["template_evidence"],
        "template",
    )
    warnings.extend(warn)
    insight.avoid_fields, warn = _filter_avoid_recommendations(
        insight.avoid_fields,
        evidence["field_evidence"],
        "field",
    )
    warnings.extend(warn)

    insight.next_experiments, warn = _filter_experiments_by_evidence(
        insight.next_experiments,
        evidence,
    )
    warnings.extend(warn)

    # Remove recommendation contradictions.
    overlap = set(insight.recommended_templates) & set(insight.avoid_templates)
    if overlap:
        for item in sorted(overlap):
            warnings.append(f"Removed contradictory template recommendation: {item}")
        insight.recommended_templates = [
            x for x in insight.recommended_templates if x not in overlap
        ]

    overlap = set(insight.recommended_fields) & set(insight.avoid_fields)
    if overlap:
        for item in sorted(overlap):
            warnings.append(f"Removed contradictory field recommendation: {item}")
        insight.recommended_fields = [
            x for x in insight.recommended_fields if x not in overlap
        ]

    # Direction is a hypothesis, never empirical evidence, so only normalize it.
    insight.recommended_directions = [
        direction
        for direction in insight.recommended_directions
        if direction in _VALID_DIRECTIONS
    ]

    # Reconstruct pattern labels from deterministic evidence so the LLM cannot
    # call weak results "failures" or OOS-only results "robust".
    insight.promising_patterns, warn = _deterministic_lead_labels(evidence)
    warnings.extend(warn)

    insight.failure_patterns = _deterministic_failure_labels(evidence)
    insight.fixable_patterns = _filter_fixable_patterns(
        insight.fixable_patterns,
        evidence,
    )

    warnings.extend(_normalize_claim_language(insight, evidence))
    return _dedupe(warnings)


def _sanitize_live_list(
    values: Sequence[str],
    allowed: set[str],
    label: str,
) -> tuple[list[str], list[str]]:
    result: list[str] = []
    warnings: list[str] = []
    for item in values:
        if item in allowed:
            result.append(item)
        else:
            warnings.append(f"Removed unknown/non-live {label}: {item}")
    return result, warnings


def _validate_experiments(
    experiments: Sequence[dict[str, Any]],
    *,
    templates: set[str],
    fields: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sanitize experiments against the actual live search space."""

    result: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    for experiment in experiments:
        template = str(experiment.get("template", "")).strip()
        experiment_fields = _string_list(experiment.get("fields", []))
        direction = str(experiment.get("direction", "")).strip().lower()
        reason = str(experiment.get("reason", "")).strip()

        if templates and template not in templates:
            warnings.append(
                f"Removed experiment with unknown/non-live template: {template}"
            )
            continue
        if fields:
            unknown = [x for x in experiment_fields if x not in fields]
            if unknown:
                warnings.append(
                    "Removed experiment with unknown/non-live fields: "
                    + ", ".join(unknown)
                )
                continue
        if direction not in _VALID_DIRECTIONS:
            warnings.append(f"Removed experiment with invalid direction: {direction}")
            continue
        if not experiment_fields:
            warnings.append(f"Removed experiment with no fields: {template}")
            continue
        if not reason:
            warnings.append(f"Removed experiment without a reason: {template}")
            continue

        key = (template, tuple(experiment_fields), direction)
        if key in seen:
            warnings.append(
                f"Removed duplicate next_experiment: {template} {experiment_fields}"
            )
            continue
        seen.add(key)

        result.append(
            {
                "template": template,
                "fields": experiment_fields,
                "direction": direction,
                "reason": reason,
            }
        )

    return result, warnings


def _filter_avoid_recommendations(
    items: Sequence[str],
    evidence: Mapping[str, Mapping[str, int]],
    label: str,
) -> tuple[list[str], list[str]]:
    """Keep avoid recommendations only where failure evidence is clear."""

    result: list[str] = []
    warnings: list[str] = []
    for item in items:
        counts = evidence.get(item, {})
        promising = int(counts.get("promising", 0))
        failures = int(counts.get("failure", 0))
        weak = int(counts.get("weak", 0))

        if promising > 0:
            warnings.append(
                f"Removed unsupported avoid {label}: {item}; positive "
                "counter-evidence exists."
            )
            continue
        if failures <= 0 or failures <= weak:
            warnings.append(
                f"Removed unsupported avoid {label}: {item}; failure evidence "
                "is not dominant enough."
            )
            continue
        result.append(item)
    return result, warnings


def _filter_experiments_by_evidence(
    experiments: Sequence[dict[str, Any]],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Require every proposed experiment to target observed evidence."""

    tiers = evidence["tiers"]
    observed_templates = set(evidence["observed_templates"])
    observed_fields = set(evidence["observed_fields"])
    template_tiers = tiers["template_tiers"]
    field_tiers = tiers["field_tiers"]

    result: list[dict[str, Any]] = []
    warnings: list[str] = []

    for experiment in experiments:
        template = experiment["template"]
        fields = experiment["fields"]
        reason = experiment["reason"]

        template_known = template in observed_templates
        known_fields = [x for x in fields if x in observed_fields]
        if not template_known and not known_fields:
            warnings.append(
                f"Removed unrelated next_experiment: {template} {fields}"
            )
            continue

        template_tier = template_tiers.get(template, "UNOBSERVED")
        field_tiers_used = [field_tiers.get(x, "UNOBSERVED") for x in fields]

        has_ready = (
            template_tier == "READY"
            or any(tier == "READY" for tier in field_tiers_used)
        )
        has_signal = (
            template_tier == "OOS_SIGNAL"
            or any(tier == "OOS_SIGNAL" for tier in field_tiers_used)
        )
        is_failure = template_tier == "FAILURE" or any(
            tier == "FAILURE" for tier in field_tiers_used
        )

        # Failure repair is allowed, but only with an explicit repair target.
        if is_failure and not _reason_mentions_fix(reason):
            warnings.append(
                f"Removed failure-driven next_experiment without an explicit "
                f"repair target: {template} {fields}"
            )
            continue

        # Signal-only exploration is allowed only when the reason targets a
        # concrete missing robustness/readiness issue.
        if has_signal and not has_ready and not _reason_mentions_robustness_gap(reason):
            warnings.append(
                f"Removed OOS-signal next_experiment without a robustness/readiness "
                f"target: {template} {fields}"
            )
            continue

        # Pure weak/unobserved ideas are rejected. The engine should learn from
        # evidence first, not free-associate from the LLM.
        if not has_ready and not has_signal and not is_failure:
            warnings.append(
                f"Removed next_experiment lacking READY/OOS/failure evidence: "
                f"{template} {fields}"
            )
            continue

        result.append(experiment)

    if len(result) > 5:
        result = result[:5]
        warnings.append("Trimmed next_experiments to the maximum of 5.")

    return result, warnings


def _reason_mentions_fix(reason: str) -> bool:
    text = reason.lower()
    phrases = (
        "fix",
        "repair",
        "address",
        "improve",
        "mitigate",
        "correct",
        "failure",
        "weakness",
        "failed",
        "turnover",
    )
    return any(phrase in text for phrase in phrases)


def _reason_mentions_robustness_gap(reason: str) -> bool:
    text = reason.lower()
    phrases = (
        "robust",
        "robustness",
        "turnover",
        "regional",
        "region",
        "in-sample",
        "in sample",
        "is performance",
        "readiness",
        "gate",
        "test failed",
        "failed test",
        "sub-universe",
        "oos",
        "out-of-sample",
    )
    return any(phrase in text for phrase in phrases)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _build_evidence_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    """Construct a deterministic experiment/template/field evidence matrix."""

    experiments_raw: list[Mapping[str, Any]] = []
    for key in ("latest_experiments", "top_experiments"):
        value = context.get(key, [])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            experiments_raw.extend(
                item for item in value if isinstance(item, Mapping)
            )

    experiment_records: list[dict[str, Any]] = []
    seen_alpha_ids: set[str] = set()
    template_evidence: dict[str, dict[str, int]] = {}
    field_evidence: dict[str, dict[str, int]] = {}
    observed_templates: set[str] = set()
    observed_fields: set[str] = set()

    for raw in experiments_raw:
        alpha_id = str(raw.get("alpha_id", "")).strip()
        if alpha_id and alpha_id in seen_alpha_ids:
            continue
        if alpha_id:
            seen_alpha_ids.add(alpha_id)

        template = str(raw.get("template", "")).strip()
        fields = _string_list(raw.get("fields", []))
        observed_templates.add(template) if template else None
        observed_fields.update(fields)

        research_class = str(raw.get("research_class", "")).strip().upper()
        if research_class not in _VALID_CLASSES:
            research_class = _fallback_classification(raw)

        if template:
            template_evidence.setdefault(template, _empty_evidence_bucket())[
                research_class.lower()
            ] += 1
        for field_id in fields:
            field_evidence.setdefault(field_id, _empty_evidence_bucket())[
                research_class.lower()
            ] += 1

        experiment_records.append(
            {
                "alpha_id": alpha_id,
                "template": template,
                "fields": fields,
                "research_class": research_class,
                "test_sharpe": _to_float(raw.get("test_sharpe")),
                "test_fitness": _to_float(raw.get("test_fitness")),
                "test_turnover": _to_float(raw.get("test_turnover")),
                "robustness_score": _to_float(raw.get("robustness_score")),
                "research_score": _to_float(raw.get("research_score")),
                "failed_gates": _string_list(raw.get("failed_gates", [])),
                "failed_brain_tests": _string_list(raw.get("failed_brain_tests", [])),
            }
        )

    tiers = _build_evidence_tiers(experiment_records)

    return {
        "observed_templates": sorted(x for x in observed_templates if x),
        "observed_fields": sorted(observed_fields),
        "template_evidence": template_evidence,
        "field_evidence": field_evidence,
        "experiments": experiment_records,
        "tiers": tiers,
    }


def _build_evidence_tiers(experiments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assign deterministic OOS, robustness and readiness tiers."""

    by_template: dict[str, list[Mapping[str, Any]]] = {}
    by_field: dict[str, list[Mapping[str, Any]]] = {}

    for record in experiments:
        template = str(record.get("template", "")).strip()
        fields = _string_list(record.get("fields", []))
        if template:
            by_template.setdefault(template, []).append(record)
        for field_id in fields:
            by_field.setdefault(field_id, []).append(record)

    def classify(records: Sequence[Mapping[str, Any]]) -> str:
        if not records:
            return "UNOBSERVED"

        promising = [r for r in records if str(r.get("research_class", "")).upper() == "PROMISING"]
        failures = [r for r in records if str(r.get("research_class", "")).upper() == "FAILURE"]
        weak = [r for r in records if str(r.get("research_class", "")).upper() == "WEAK"]

        # Failure dominates only when there is no promising counter-evidence.
        if promising:
            ready_records = [r for r in promising if _record_is_ready(r)]
            if ready_records:
                return "READY"
            return "OOS_SIGNAL"

        if failures and not weak:
            return "FAILURE"
        if failures and len(failures) > len(weak):
            return "FAILURE"
        return "WEAK"

    template_tiers = {item: classify(records) for item, records in by_template.items()}
    field_tiers = {item: classify(records) for item, records in by_field.items()}

    signal_templates = sorted(x for x, tier in template_tiers.items() if tier == "OOS_SIGNAL")
    signal_fields = sorted(x for x, tier in field_tiers.items() if tier == "OOS_SIGNAL")
    ready_templates = sorted(x for x, tier in template_tiers.items() if tier == "READY")
    ready_fields = sorted(x for x, tier in field_tiers.items() if tier == "READY")
    failure_templates = sorted(x for x, tier in template_tiers.items() if tier == "FAILURE")
    failure_fields = sorted(x for x, tier in field_tiers.items() if tier == "FAILURE")

    return {
        "template_tiers": template_tiers,
        "field_tiers": field_tiers,
        "signal_templates": signal_templates,
        "signal_fields": signal_fields,
        "research_ready_templates": ready_templates,
        "research_ready_fields": ready_fields,
        "failure_templates": failure_templates,
        "failure_fields": failure_fields,
    }


def _record_is_ready(record: Mapping[str, Any]) -> bool:
    """Deterministic readiness test for research recommendations."""

    if str(record.get("research_class", "")).upper() != "PROMISING":
        return False

    failed_tests = {x.upper() for x in _string_list(record.get("failed_brain_tests", []))}
    failed_gates = {x.upper() for x in _string_list(record.get("failed_gates", []))}
    if len(failed_tests) > _MAX_READY_FAILED_BRAIN_TESTS:
        return False
    if failed_gates:
        return False

    robustness = _to_float(record.get("robustness_score"))
    if robustness is None or robustness < _MIN_READY_ROBUSTNESS_SCORE:
        return False

    turnover = _to_float(record.get("test_turnover"))
    if turnover is not None and not (0.01 < turnover < 0.70):
        return False

    return True


def _empty_evidence_bucket() -> dict[str, int]:
    return {"promising": 0, "weak": 0, "failure": 0}


def _fallback_classification(experiment: Mapping[str, Any]) -> str:
    """Conservative metric-only fallback for legacy memory records."""

    test_sharpe = _to_float(experiment.get("test_sharpe"))
    test_fitness = _to_float(experiment.get("test_fitness"))

    if (
        test_sharpe is not None
        and test_fitness is not None
        and test_sharpe >= 1.5
        and test_fitness >= 1.0
    ):
        return "PROMISING"
    if (
        (test_sharpe is not None and test_sharpe <= -1.0)
        or (test_fitness is not None and test_fitness <= -1.0)
    ):
        return "FAILURE"
    return "WEAK"


def _build_evidence_digest(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create a small deterministic summary suitable for the LLM prompt."""

    tiers = evidence.get("tiers", {})
    records = evidence.get("experiments", [])
    return {
        "experiment_count": len(records),
        "template_tiers": tiers.get("template_tiers", {}),
        "field_tiers": tiers.get("field_tiers", {}),
        "research_ready_templates": tiers.get("research_ready_templates", []),
        "research_ready_fields": tiers.get("research_ready_fields", []),
        "oos_signal_templates": tiers.get("signal_templates", []),
        "oos_signal_fields": tiers.get("signal_fields", []),
        "failure_templates": tiers.get("failure_templates", []),
        "failure_fields": tiers.get("failure_fields", []),
        "experiments": records,
    }


# ---------------------------------------------------------------------------
# Post-processing and truthful presentation
# ---------------------------------------------------------------------------


def _deterministic_lead_labels(evidence: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    tiers = evidence["tiers"]
    labels: list[str] = []
    for template in tiers.get("signal_templates", []):
        labels.append(f"{template} (OOS signal; not research-ready)")
    for template in tiers.get("research_ready_templates", []):
        labels.append(f"{template} (research-ready)")
    return labels, []


def _deterministic_failure_labels(evidence: Mapping[str, Any]) -> list[str]:
    tiers = evidence["tiers"]
    labels: list[str] = []
    labels.extend(tiers.get("failure_templates", []))
    labels.extend(tiers.get("failure_fields", []))
    return labels


def _filter_fixable_patterns(
    patterns: Sequence[str],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Keep LLM fixable labels only when they map to observed evidence."""

    observed_templates = set(evidence.get("observed_templates", []))
    observed_fields = set(evidence.get("observed_fields", []))
    allowed = observed_templates | observed_fields

    result: list[str] = []
    for pattern in patterns:
        normalized = str(pattern).strip()
        if not normalized:
            continue
        if any(item.lower() in normalized.lower() for item in allowed):
            result.append(normalized)
    return _dedupe(result)


def _normalize_claim_language(
    insight: ResearchInsight,
    evidence: Mapping[str, Any],
) -> list[str]:
    """Force visible language to agree with deterministic evidence tiers."""

    warnings: list[str] = []
    tiers = evidence["tiers"]
    signal_items = set(tiers.get("signal_templates", [])) | set(tiers.get("signal_fields", []))
    ready_items = set(tiers.get("research_ready_templates", [])) | set(tiers.get("research_ready_fields", []))

    def rewrite(text: str) -> tuple[str, bool]:
        changed = False
        rewritten = text
        lower = rewritten.lower()

        for item in sorted(signal_items - ready_items, key=len, reverse=True):
            if item.lower() not in lower:
                continue
            replacements = (
                (r"\bpromising\b", "a research lead"),
                (r"\brobust\b", "not yet robust"),
                (r"\breliable\b", "not yet validated"),
                (r"\bestablished\b", "observed"),
                (r"\bvalidated\b", "not yet validated"),
                (r"\bresearch-ready\b", "not research-ready"),
            )
            for pattern, replacement in replacements:
                new_text = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
                if new_text != rewritten:
                    rewritten = new_text
                    changed = True

        return rewritten, changed

    insight.summary, changed = rewrite(insight.summary)
    if changed:
        warnings.append("Rewrote summary language to match deterministic evidence tiers.")

    new_patterns: list[str] = []
    for pattern in insight.promising_patterns:
        rewritten, changed = rewrite(pattern)
        new_patterns.append(rewritten)
        if changed:
            warnings.append(
                f"Rewrote research-lead language to match evidence tiers: {pattern}"
            )
    insight.promising_patterns = new_patterns

    return _dedupe(warnings)


def _normalize_insight_to_evidence(
    insight: ResearchInsight,
    *,
    evidence: Mapping[str, Any],
) -> list[str]:
    """Final pass ensuring no LLM field contradicts deterministic evidence."""

    warnings: list[str] = []
    tiers = evidence.get("tiers", {})
    ready_templates = set(tiers.get("research_ready_templates", []))
    ready_fields = set(tiers.get("research_ready_fields", []))
    signal_templates = set(tiers.get("signal_templates", []))
    signal_fields = set(tiers.get("signal_fields", []))

    # The final summary is deterministic when there is no READY evidence.
    # This avoids a model sentence saying "robust" while the structured facts
    # say otherwise.
    if not ready_templates and not ready_fields:
        lead_templates = sorted(signal_templates)
        lead_fields = sorted(signal_fields)
        failure_templates = sorted(tiers.get("failure_templates", []))

        parts = [
            "No research-ready template or field is established by the current memory."
        ]
        if lead_templates:
            parts.append(
                "Research leads with OOS signal but insufficient robustness/readiness: "
                + ", ".join(lead_templates)
                + "."
            )
        if failure_templates:
            parts.append(
                "Failure evidence is concentrated in: "
                + ", ".join(failure_templates)
                + "."
            )
        if lead_fields:
            parts.append(
                "Fields appearing in OOS-signal experiments require independent validation before recommendation: "
                + ", ".join(lead_fields)
                + "."
            )
        deterministic_summary = " ".join(parts)
        if insight.summary.strip() != deterministic_summary.strip():
            insight.summary = deterministic_summary
            warnings.append(
                "Replaced LLM summary with deterministic evidence summary because no research-ready evidence exists."
            )
    else:
        # Even with READY evidence, remove generic certainty language.
        prohibited = (
            "guaranteed",
            "proves",
            "will outperform",
            "certainly",
            "bound to perform",
        )
        for phrase in prohibited:
            if phrase in insight.summary.lower():
                insight.summary = re.sub(
                    re.escape(phrase),
                    "hypothesized",
                    insight.summary,
                    flags=re.IGNORECASE,
                )
                warnings.append(
                    f"Rewrote unsupported certainty phrase in summary: {phrase}"
                )

    # Make the public recommendation fields exactly match deterministic READY sets.
    original = insight.recommended_templates[:]
    insight.recommended_templates = [x for x in original if x in ready_templates]
    for item in original:
        if item not in ready_templates:
            warnings.append(
                f"Removed non-ready template recommendation in final evidence pass: {item}"
            )

    original = insight.recommended_fields[:]
    insight.recommended_fields = [x for x in original if x in ready_fields]
    for item in original:
        if item not in ready_fields:
            warnings.append(
                f"Removed non-ready field recommendation in final evidence pass: {item}"
            )

    return _dedupe(warnings)


def _deterministic_confidence(evidence: Mapping[str, Any]) -> str:
    """Confidence based only on the amount/quality of deterministic evidence."""

    tiers = evidence.get("tiers", {})
    ready_template_count = len(tiers.get("research_ready_templates", []))
    ready_field_count = len(tiers.get("research_ready_fields", []))
    signal_count = len(tiers.get("signal_templates", []))
    experiment_count = len(evidence.get("experiments", []))

    if ready_template_count >= 2 and ready_field_count >= 2:
        return "HIGH"
    if ready_template_count >= 1 and ready_field_count >= 1:
        return "MEDIUM"
    if experiment_count >= 8 and signal_count >= 2:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def insight_to_dict(
    insight: ResearchInsight,
    *,
    include_raw_response: bool = False,
) -> dict[str, Any]:
    """Convert an insight into a plain dictionary."""

    result = {
        "schema_version": insight.schema_version,
        "summary": insight.summary,
        "promising_patterns": insight.promising_patterns,
        "failure_patterns": insight.failure_patterns,
        "fixable_patterns": insight.fixable_patterns,
        "recommended_templates": insight.recommended_templates,
        "recommended_fields": insight.recommended_fields,
        "recommended_directions": insight.recommended_directions,
        "avoid_templates": insight.avoid_templates,
        "avoid_fields": insight.avoid_fields,
        "next_experiments": insight.next_experiments,
        "confidence": insight.confidence,
        "validation_warnings": insight.validation_warnings,
        "evidence_tiers": insight.evidence_tiers,
    }
    if include_raw_response:
        result["raw_response"] = insight.raw_response
    return result


def build_generation_context(insight: ResearchInsight) -> dict[str, Any]:
    """Convert an analyst insight into candidate-generator context."""

    return {
        "schema_version": insight.schema_version,
        "research_summary": insight.summary,
        "research_leads": insight.promising_patterns,
        "promising_patterns": insight.promising_patterns,
        "failure_patterns": insight.failure_patterns,
        "fixable_patterns": insight.fixable_patterns,
        "recommended_templates": insight.recommended_templates,
        "recommended_fields": insight.recommended_fields,
        "recommended_directions": insight.recommended_directions,
        "avoid_templates": insight.avoid_templates,
        "avoid_fields": insight.avoid_fields,
        "next_experiments": insight.next_experiments,
        "confidence": insight.confidence,
        "validation_warnings": insight.validation_warnings,
        "evidence_tiers": insight.evidence_tiers,
    }


def print_insight(insight: ResearchInsight) -> None:
    """Print an analyst result for manual review."""

    print("=" * 80)
    print("RESEARCH ANALYST")
    print("=" * 80)
    print(f"Schema: {insight.schema_version}")
    print(f"Confidence: {insight.confidence}")

    if insight.summary:
        print("\nSummary:")
        print(insight.summary)

    if insight.promising_patterns:
        print("\nResearch leads:")
        for item in insight.promising_patterns:
            print(f"  + {item}")

    if insight.failure_patterns:
        print("\nFailure patterns:")
        for item in insight.failure_patterns:
            print(f"  - {item}")

    if insight.fixable_patterns:
        print("\nPotentially fixable:")
        for item in insight.fixable_patterns:
            print(f"  * {item}")

    if insight.recommended_templates:
        print("\nResearch-ready templates:")
        print("  " + ", ".join(insight.recommended_templates))
    else:
        print("\nResearch-ready templates:")
        print("  None")

    if insight.recommended_fields:
        print("\nResearch-ready fields:")
        print("  " + ", ".join(insight.recommended_fields))
    else:
        print("\nResearch-ready fields:")
        print("  None")

    if insight.recommended_directions:
        print("\nHypothesized directions:")
        for item in insight.recommended_directions:
            print(f"  -> {item}")

    if insight.avoid_templates:
        print("\nAvoid templates:")
        print("  " + ", ".join(insight.avoid_templates))

    if insight.avoid_fields:
        print("\nAvoid fields:")
        print("  " + ", ".join(insight.avoid_fields))

    if insight.next_experiments:
        print("\nNext experiments:")
        for index, experiment in enumerate(insight.next_experiments, start=1):
            print(
                f"  {index}. {experiment.get('template', '')} | "
                f"{experiment.get('fields', [])} | "
                f"{experiment.get('direction', '')}"
            )
            if experiment.get("reason"):
                print(f"     {experiment['reason']}")

    if insight.evidence_tiers:
        print("\nEvidence tiers:")
        tiers = insight.evidence_tiers
        print("  OOS-signal templates:", ", ".join(tiers.get("signal_templates", [])) or "None")
        print("  OOS-signal fields:", ", ".join(tiers.get("signal_fields", [])) or "None")
        print("  Research-ready templates:", ", ".join(tiers.get("research_ready_templates", [])) or "None")
        print("  Research-ready fields:", ", ".join(tiers.get("research_ready_fields", [])) or "None")
        print("  Failure templates:", ", ".join(tiers.get("failure_templates", [])) or "None")
        print("  Failure fields:", ", ".join(tiers.get("failure_fields", [])) or "None")

    if insight.validation_warnings:
        print("\nValidation warnings:")
        for warning in insight.validation_warnings:
            print(f"  ! {warning}")


def _normalize_set(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {text for value in values if (text := str(value).strip())}


def _dedupe(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
