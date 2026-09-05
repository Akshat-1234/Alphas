"""
Deterministic structured candidate generator for the alpha-research loop.

Pipeline role:

    validated ResearchInsight
            |
            v
    structured ResearchSpec
            |
            v
    FastExprCompiler
            |
            v
    FastExprValidator
            |
            v
           BRAIN

This module NEVER generates FASTEXPR and NEVER calls BRAIN.

Design goals:
    1. Never repeat an already-tested exact structured experiment.
    2. Never silently invent missing historical parameters.
    3. Generate variants according to the documented repair target.
    4. Do not automatically flip direction unless direction is an explicit
       unresolved research question.
    5. Do not fall back to OOS-only "research leads" as though they were
       research-ready evidence.
    6. Preserve deterministic ordering, deduplication, and candidate caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 4

VALID_DIRECTIONS = {"positive", "negative"}

DEFAULT_MAX_CANDIDATES = 12
DEFAULT_MAX_PER_TEMPLATE = 4


@dataclass(frozen=True)
class ResearchSpec:
    """One structured research hypothesis before compilation."""

    template: str
    fields: tuple[str, ...]
    window: int = 60
    backfill_window: int = 60
    direction: str = "positive"
    family: str = ""
    intuition: str = ""
    repair_target: str = ""
    source: str = "analyst"
    source_rank: int = 0
    repair_reason: str = ""

    def signature(self) -> tuple[Any, ...]:
        """Canonical exact identity of a structured experiment."""
        return (
            self.template,
            self.fields,
            self.window,
            self.backfill_window,
            self.direction,
        )

    def structure_signature(self) -> tuple[Any, ...]:
        """
        Canonical identity without time/direction parameters.

        This is useful for deciding whether a base hypothesis already exists
        while still permitting explicitly generated parameter variants.
        """
        return (
            self.template,
            self.fields,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "template": self.template,
            "fields": list(self.fields),
            "window": self.window,
            "backfill_window": self.backfill_window,
            "direction": self.direction,
            "family": self.family,
            "intuition": self.intuition,
            "repair_target": self.repair_target,
            "source": self.source,
            "source_rank": self.source_rank,
            "repair_reason": self.repair_reason,
        }


class CandidateGenerationError(ValueError):
    """Raised when candidate-generation configuration is invalid."""


class CandidateGenerator:
    """
    Deterministic candidate generator.

    `template_field_counts` must come from FastExprCompiler.
    `live_fields` must contain actual live BRAIN field IDs.
    `allowed_windows` must come from the compiler/validation configuration.

    `blocked_specs` is intentionally explicit. Each blocked item must carry:
        template, fields, window, backfill_window, direction

    Missing historical parameters are NOT guessed.

    For legacy memory records that do not contain these parameters, use
    `blocked_structures` to prevent an already-tested template/field hypothesis
    from becoming the base candidate. Parameter variants can still be generated
    explicitly from that blocked structure, which is useful when repairing
    turnover or robustness failures.
    """

    def __init__(
        self,
        *,
        template_field_counts: Mapping[str, int],
        live_fields: Iterable[str],
        allowed_windows: Iterable[int],
        blocked_specs: Iterable[Any] | None = None,
        blocked_structures: Iterable[Any] | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_per_template: int = DEFAULT_MAX_PER_TEMPLATE,
    ) -> None:
        self.template_field_counts = {
            str(key).strip().upper(): int(value)
            for key, value in template_field_counts.items()
        }

        self.live_fields = {
            str(value).strip()
            for value in live_fields
            if str(value).strip()
        }

        self.allowed_windows = tuple(
            sorted({int(value) for value in allowed_windows})
        )

        if not self.template_field_counts:
            raise CandidateGenerationError(
                "template_field_counts cannot be empty."
            )

        if any(
            count <= 0
            for count in self.template_field_counts.values()
        ):
            raise CandidateGenerationError(
                "All template field counts must be positive."
            )

        if not self.live_fields:
            raise CandidateGenerationError(
                "live_fields cannot be empty."
            )

        if not self.allowed_windows:
            raise CandidateGenerationError(
                "allowed_windows cannot be empty."
            )

        if max_candidates <= 0:
            raise CandidateGenerationError(
                "max_candidates must be positive."
            )

        if max_per_template <= 0:
            raise CandidateGenerationError(
                "max_per_template must be positive."
            )

        self.max_candidates = int(max_candidates)
        self.max_per_template = int(max_per_template)

        self.blocked_signatures = self._normalize_blocked_specs(
            blocked_specs or []
        )

        self.blocked_structures = (
            self._normalize_blocked_structures(
                blocked_structures or []
            )
        )

    # ------------------------------------------------------------------
    # Historical blocking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fields(item: Any) -> tuple[str, ...] | None:
        """Extract a non-empty field tuple without inventing values."""
        if isinstance(item, Mapping):
            raw_fields = item.get("fields", [])
        else:
            raw_fields = getattr(item, "fields", [])

        if isinstance(raw_fields, str):
            value = raw_fields.strip()
            return (value,) if value else None

        try:
            fields = tuple(
                str(value).strip()
                for value in raw_fields
                if str(value).strip()
            )
        except TypeError:
            return None

        return fields or None

    @classmethod
    def _record_to_exact_signature(
        cls,
        item: Any,
    ) -> tuple[Any, ...] | None:
        """
        Convert an object into an exact experiment signature.

        Critically, missing window/backfill/direction values are rejected
        instead of silently defaulting to 60/60/positive.
        """
        if isinstance(item, ResearchSpec):
            return item.signature()

        if isinstance(item, Mapping):
            template = str(
                item.get("template", "")
            ).strip().upper()

            fields = cls._extract_fields(item)
            if not template or not fields:
                return None

            required_keys = (
                "window",
                "backfill_window",
                "direction",
            )

            if any(
                key not in item
                for key in required_keys
            ):
                return None

            try:
                window = int(item["window"])
                backfill_window = int(item["backfill_window"])
            except (TypeError, ValueError):
                return None

            direction = str(
                item["direction"]
            ).strip().lower()

            if direction not in VALID_DIRECTIONS:
                return None

            return (
                template,
                fields,
                window,
                backfill_window,
                direction,
            )

        template = str(
            getattr(item, "template", "")
        ).strip().upper()

        fields = cls._extract_fields(item)

        if not template or not fields:
            return None

        window_value = getattr(
            item,
            "window",
            None,
        )
        backfill_value = getattr(
            item,
            "backfill_window",
            None,
        )
        direction_value = getattr(
            item,
            "direction",
            None,
        )

        if (
            window_value is None
            or backfill_value is None
            or direction_value is None
        ):
            return None

        try:
            window = int(window_value)
            backfill_window = int(backfill_value)
        except (TypeError, ValueError):
            return None

        direction = str(
            direction_value
        ).strip().lower()

        if direction not in VALID_DIRECTIONS:
            return None

        return (
            template,
            fields,
            window,
            backfill_window,
            direction,
        )

    @classmethod
    def _record_to_structure_signature(
        cls,
        item: Any,
    ) -> tuple[Any, ...] | None:
        """Extract only template + ordered fields."""
        if isinstance(item, ResearchSpec):
            return item.structure_signature()

        if isinstance(item, Mapping):
            template = str(
                item.get("template", "")
            ).strip().upper()
            fields = cls._extract_fields(item)

            if not template or not fields:
                return None

            return (
                template,
                fields,
            )

        template = str(
            getattr(item, "template", "")
        ).strip().upper()
        fields = cls._extract_fields(item)

        if not template or not fields:
            return None

        return (
            template,
            fields,
        )

    @classmethod
    def _normalize_blocked_specs(
        cls,
        blocked_specs: Iterable[Any],
    ) -> set[tuple[Any, ...]]:
        """
        Normalize exact historical signatures.

        Invalid/incomplete records are ignored rather than assigned defaults.
        """
        blocked: set[tuple[Any, ...]] = set()

        for item in blocked_specs:
            signature = cls._record_to_exact_signature(item)
            if signature is not None:
                blocked.add(signature)

        return blocked

    @classmethod
    def _normalize_blocked_structures(
        cls,
        blocked_structures: Iterable[Any],
    ) -> set[tuple[Any, ...]]:
        """Normalize historical template/field structures."""
        blocked: set[tuple[Any, ...]] = set()

        for item in blocked_structures:
            signature = cls._record_to_structure_signature(item)
            if signature is not None:
                blocked.add(signature)

        return blocked

    def _is_exactly_blocked(
        self,
        spec: ResearchSpec,
    ) -> bool:
        """Return True only when the complete experiment was tested."""
        return spec.signature() in self.blocked_signatures

    def _base_structure_is_blocked(
        self,
        spec: ResearchSpec,
    ) -> bool:
        """Return True when this template/field structure was tested before."""
        return spec.structure_signature() in self.blocked_structures

    # ------------------------------------------------------------------
    # Validation / normalization
    # ------------------------------------------------------------------

    def _coerce_window(
        self,
        value: Any,
        default: int = 60,
    ) -> int:
        """Snap a proposed window to the nearest legal window."""
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = int(default)

        if candidate in self.allowed_windows:
            return candidate

        return min(
            self.allowed_windows,
            key=lambda allowed: (
                abs(allowed - candidate),
                allowed,
            ),
        )

    def _repair_target_for_template(
        self,
        template: str,
    ) -> str:
        """Provide a deterministic repair target when omitted."""
        mapping = {
            "DECAY": "turnover_and_robustness",
            "INTERACTION": "robustness_and_train_oos_consistency",
            "CONTRAST": "robustness_and_train_oos_consistency",
            "CHANGE": "direction_and_timescale_validation",
            "STABILITY": "predictive_strength_and_turnover",
            "SMOOTHED": "predictive_strength_and_turnover",
            "RATIO": "direction_and_generalization_validation",
            "RATIO_STATE": "direction_and_generalization_validation",
            "RATIO_CHANGE": "direction_and_generalization_validation",
            "CORRELATION": "deprioritized_template_validation",
            "LEVEL": "baseline_validation",
            "HISTORICAL_STATE": "timescale_and_robustness_validation",
        }

        return mapping.get(
            template,
            "validation",
        )

    def _normalize_experiment(
        self,
        item: Mapping[str, Any],
        *,
        source_rank: int,
    ) -> ResearchSpec | None:
        """Validate one structured analyst experiment."""
        template = str(
            item.get("template", "")
        ).strip().upper()

        fields = self._extract_fields(item)

        if template not in self.template_field_counts:
            return None

        if not fields:
            return None

        required = self.template_field_counts[
            template
        ]

        if len(fields) != required:
            return None

        if len(set(fields)) != len(fields):
            return None

        if any(
            field not in self.live_fields
            for field in fields
        ):
            return None

        direction = str(
            item.get("direction", "")
        ).strip().lower()

        if direction not in VALID_DIRECTIONS:
            return None

        repair_target = str(
            item.get("repair_target", "")
            or ""
        ).strip()

        # Repair targets are deterministic candidate-generation metadata.
        # Older analyst schemas may omit them, so infer the target from the
        # template instead of rejecting an otherwise valid research lead.
        if not repair_target:
            repair_target = self._repair_target_for_template(
                template
            )

        window = self._coerce_window(
            item.get("window", 60)
        )

        backfill_window = self._coerce_window(
            item.get("backfill_window", 60)
        )

        return ResearchSpec(
            template=template,
            fields=fields,
            window=window,
            backfill_window=backfill_window,
            direction=direction,
            family=str(
                item.get("family", "")
                or ""
            ).strip(),
            intuition=str(
                item.get("intuition", "")
                or ""
            ).strip(),
            repair_target=repair_target,
            source="analyst",
            source_rank=source_rank,
            repair_reason=str(
                item.get("repair_reason", "")
                or ""
            ).strip(),
        )

    # ------------------------------------------------------------------
    # Repair-aware expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_tokens(
        repair_target: str,
    ) -> set[str]:
        """Normalize repair-target text into coarse deterministic tokens."""
        text = str(
            repair_target
        ).strip().lower().replace("-", "_")

        tokens: set[str] = set()

        aliases = {
            "turnover": "turnover",
            "robustness": "robustness",
            "regional": "robustness",
            "region": "robustness",
            "train": "train_oos",
            "oos": "train_oos",
            "consistency": "train_oos",
            "direction": "direction",
            "timescale": "timescale",
            "window": "timescale",
            "generalization": "generalization",
            "validation": "validation",
        }

        for key, token in aliases.items():
            if key in text:
                tokens.add(token)

        return tokens

    def _preferred_windows(
        self,
        base_window: int,
        tokens: set[str],
    ) -> list[int]:
        """
        Pick alternative windows according to the repair target.

        For turnover issues, shorter windows are tested first.
        For robustness/generalization, nearby windows are tested first.
        """
        available = [
            value
            for value in self.allowed_windows
            if value != base_window
        ]

        if "turnover" in tokens:
            return sorted(
                available,
                key=lambda value: (
                    value > base_window,
                    abs(value - base_window),
                    value,
                ),
            )

        return sorted(
            available,
            key=lambda value: (
                abs(value - base_window),
                value,
            ),
        )

    def _preferred_backfills(
        self,
        base_backfill: int,
        tokens: set[str],
    ) -> list[int]:
        """Pick alternative backfill windows only when useful."""
        available = [
            value
            for value in self.allowed_windows
            if value != base_backfill
        ]

        if "turnover" in tokens:
            return sorted(
                available,
                key=lambda value: (
                    value > base_backfill,
                    abs(value - base_backfill),
                    value,
                ),
            )

        return sorted(
            available,
            key=lambda value: (
                abs(value - base_backfill),
                value,
            ),
        )

    def _expand_spec(
        self,
        base: ResearchSpec,
    ) -> list[ResearchSpec]:
        """
        Produce only variants that address the declared repair target.

        Direction reversal is NOT automatic. It is generated only when the
        repair target explicitly contains a direction question.

        This prevents the common failure mode of generating arbitrary sign flips
        merely because the LLM supplied a direction.
        """
        variants: list[ResearchSpec] = []

        tokens = self._repair_tokens(
            base.repair_target
        )

        preferred_windows = self._preferred_windows(
            base.window,
            tokens,
        )

        # Short-window variants are particularly relevant to low-turnover
        # signals, while nearby windows test temporal stability.
        for window in preferred_windows[:2]:
            variants.append(
                ResearchSpec(
                    template=base.template,
                    fields=base.fields,
                    window=window,
                    backfill_window=base.backfill_window,
                    direction=base.direction,
                    family=base.family,
                    intuition=(
                        f"{base.intuition} "
                        f"Timescale repair variant at window {window}."
                    ).strip(),
                    repair_target=base.repair_target,
                    source="repair_timescale_variant",
                    source_rank=base.source_rank,
                    repair_reason=(
                        f"Test nearby window {window} to address "
                        f"{base.repair_target}."
                    ),
                )
            )

        # Backfill changes are useful for robustness/coverage, and especially
        # useful when the base signal is stale/slow-moving.
        if (
            "robustness" in tokens
            or "generalization" in tokens
        ):
            preferred_backfills = self._preferred_backfills(
                base.backfill_window,
                tokens,
            )

            for backfill in preferred_backfills[:1]:
                variants.append(
                    ResearchSpec(
                        template=base.template,
                        fields=base.fields,
                        window=base.window,
                        backfill_window=backfill,
                        direction=base.direction,
                        family=base.family,
                        intuition=(
                            f"{base.intuition} "
                            f"Backfill variant at {backfill}."
                        ).strip(),
                        repair_target=base.repair_target,
                        source="repair_backfill_variant",
                        source_rank=base.source_rank,
                        repair_reason=(
                            f"Test backfill {backfill} to address "
                            f"{base.repair_target}."
                        ),
                    )
                )

        # Direction reversal only when direction is genuinely part of the
        # unresolved question.
        if "direction" in tokens:
            opposite = (
                "negative"
                if base.direction == "positive"
                else "positive"
            )

            variants.append(
                ResearchSpec(
                    template=base.template,
                    fields=base.fields,
                    window=base.window,
                    backfill_window=base.backfill_window,
                    direction=opposite,
                    family=base.family,
                    intuition=(
                        f"{base.intuition} "
                        "Directional reversal for explicit direction validation."
                    ).strip(),
                    repair_target=base.repair_target,
                    source="repair_direction_variant",
                    source_rank=base.source_rank,
                    repair_reason=(
                        f"Test opposite direction to address "
                        f"{base.repair_target}."
                    ),
                )
            )

        return variants

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_specs(
        self,
        insight: Any,
    ) -> list[ResearchSpec]:
        """
        Fallback only from research-ready evidence.

        OOS-only `research_leads` are deliberately NOT used as fallback bases.
        """
        raw_templates = getattr(
            insight,
            "research_ready_templates",
            getattr(
                insight,
                "recommended_templates",
                [],
            ),
        ) or []

        raw_fields = getattr(
            insight,
            "research_ready_fields",
            getattr(
                insight,
                "recommended_fields",
                [],
            ),
        ) or []

        candidate_templates = list(
            dict.fromkeys(
                str(value).strip().upper()
                for value in raw_templates
                if str(value).strip().upper()
                in self.template_field_counts
            )
        )

        candidate_fields = list(
            dict.fromkeys(
                str(value).strip()
                for value in raw_fields
                if str(value).strip()
                in self.live_fields
            )
        )

        specs: list[ResearchSpec] = []

        for template in candidate_templates:
            required = self.template_field_counts[
                template
            ]

            if len(candidate_fields) < required:
                continue

            if required == 1:
                field_combinations = [
                    (candidate_fields[0],)
                ]
            elif required == 2:
                field_combinations = list(
                    combinations(
                        candidate_fields,
                        2,
                    )
                )[:2]
            else:
                continue

            for fields in field_combinations:
                specs.append(
                    ResearchSpec(
                        template=template,
                        fields=tuple(fields),
                        window=self._coerce_window(60),
                        backfill_window=self._coerce_window(60),
                        direction="positive",
                        intuition=(
                            "Deterministic fallback from research-ready "
                            "analyst evidence."
                        ),
                        repair_target=self._repair_target_for_template(
                            template
                        ),
                        source="fallback",
                        source_rank=0,
                        repair_reason=(
                            "Validate a research-ready hypothesis."
                        ),
                    )
                )

        return specs

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(
        self,
        insight: Any,
    ) -> list[ResearchSpec]:
        """
        Generate a bounded, history-aware candidate set.

        Important behavior:
            - An already-tested exact analyst hypothesis is never emitted.
            - Its repair variants may still be emitted when justified.
            - Exact historical variants are always blocked.
            - No free-form prose is interpreted.
        """
        base_specs: list[ResearchSpec] = []

        next_experiments = getattr(
            insight,
            "next_experiments",
            [],
        ) or []

        for rank, item in enumerate(
            next_experiments,
            start=1,
        ):
            if not isinstance(item, Mapping):
                continue

            normalized = self._normalize_experiment(
                item,
                source_rank=rank,
            )

            if normalized is not None:
                base_specs.append(normalized)

        candidates: list[ResearchSpec] = []

        for base in base_specs:
            # Never emit an exact repeat.
            if not self._is_exactly_blocked(base):
                candidates.append(base)

            # Always evaluate repair variants. This allows a previously tested
            # base structure to produce legitimate parameter repairs.
            candidates.extend(
                self._expand_spec(base)
            )

        if not candidates:
            candidates.extend(
                self._fallback_specs(insight)
            )

        # Exact deduplication first.
        unique: list[ResearchSpec] = []
        seen: set[tuple[Any, ...]] = set()

        for spec in candidates:
            signature = spec.signature()

            if signature in seen:
                continue

            seen.add(signature)
            unique.append(spec)

        # Then block exact historical repeats.
        unique = [
            spec
            for spec in unique
            if not self._is_exactly_blocked(spec)
        ]

        source_priority = {
            "analyst": 0,
            "fallback": 1,
            "repair_timescale_variant": 2,
            "repair_backfill_variant": 3,
            "repair_direction_variant": 4,
        }

        unique.sort(
            key=lambda spec: (
                source_priority.get(
                    spec.source,
                    99,
                ),
                spec.source_rank,
                abs(spec.window - 60),
                abs(spec.backfill_window - 60),
                spec.template,
                spec.fields,
                spec.direction,
            )
        )

        template_counts: dict[str, int] = {}
        final: list[ResearchSpec] = []

        for spec in unique:
            count = template_counts.get(
                spec.template,
                0,
            )

            if count >= self.max_per_template:
                continue

            final.append(spec)
            template_counts[spec.template] = count + 1

            if len(final) >= self.max_candidates:
                break

        return final


def specs_to_dicts(
    specs: Iterable[ResearchSpec],
) -> list[dict[str, Any]]:
    """Convert ResearchSpec objects into JSON-safe dictionaries."""
    return [
        spec.as_dict()
        for spec in specs
    ]


def print_candidates(
    specs: Sequence[ResearchSpec],
) -> None:
    """Print candidates without compiling or simulating them."""
    print("=" * 80)
    print("RESEARCH CANDIDATES")
    print("=" * 80)
    print(f"Candidates: {len(specs)}")

    for index, spec in enumerate(
        specs,
        start=1,
    ):
        print(f"\n{index}. {spec.template}")
        print(f"   Fields:          {list(spec.fields)}")
        print(f"   Window:          {spec.window}")
        print(f"   Backfill:        {spec.backfill_window}")
        print(f"   Direction:       {spec.direction}")
        print(f"   Source:          {spec.source}")
        print(f"   Repair target:   {spec.repair_target}")
        print(f"   Repair reason:   {spec.repair_reason}")
        if spec.intuition:
            print(f"   Intuition:       {spec.intuition}")
