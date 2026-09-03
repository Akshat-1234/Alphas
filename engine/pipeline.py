# ============================================================
# engine/pipeline.py
# ============================================================

"""
High-level orchestration layer for the alpha research engine.

Pipeline
--------

LLM research specification
        |
        v
real BRAIN field IDs
        |
        v
F1/F2/... aliases
        |
        v
FastExprCompiler
        |
        v
alias FASTEXPR
        |
        v
FastExprValidator
        |
        v
alias -> real BRAIN field IDs
        |
        v
ace.generate_alpha()
        |
        v
SimulationJob
        |
        v
SimulationRunner
        |
        v
BRAIN
        |
        v
SimulationJob.raw_result
        |
        v
results.normalize_result()
        |
        v
research result


Design principles
-----------------
1. LLM proposes research specifications.
2. Compiler creates FASTEXPR.
3. Validator checks FASTEXPR.
4. The alias expression is NEVER sent directly to BRAIN.
5. Only real BRAIN field IDs are sent to BRAIN.
6. SimulationJob remains the source of truth for simulation state.
7. results.py remains the source of truth for eligibility.
8. Failed candidates are retained with explicit rejection reasons.
"""


from __future__ import annotations


# ============================================================
# STANDARD LIBRARY
# ============================================================

import re

from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence


# ============================================================
# ENGINE IMPORTS
# ============================================================

import ace_lib as ace

from engine.llm import ResearchLLM

from engine.simulator import SimulationJob
from engine.simulator import SimulationRunner
from engine.simulator import create_job_from_payload


# ============================================================
# DEFAULT BRAIN SETTINGS
# ============================================================

DEFAULT_REGION = "GLB"

DEFAULT_UNIVERSE = "TOPDIV3000"

DEFAULT_DELAY = 1

DEFAULT_DECAY = 2

DEFAULT_NEUTRALIZATION = "INDUSTRY"

DEFAULT_TRUNCATION = 0.08

DEFAULT_PASTEURIZATION = "ON"

DEFAULT_TEST_PERIOD = "P1M"

DEFAULT_UNIT_HANDLING = "VERIFY"

DEFAULT_NAN_HANDLING = "ON"

DEFAULT_MAX_TRADE = "OFF"

DEFAULT_VISUALIZATION = False


# ============================================================
# ALPHA TYPES
# ============================================================

NORMAL = "NORMAL"

POWER_POOL = "POWER_POOL"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CandidateRejection:
    """
    Explicit record of a candidate rejected before BRAIN simulation.
    """

    specification: dict[str, Any]

    stage: str

    reason: str

    error_type: str = ""


@dataclass
class CompiledCandidate:
    """
    Research specification after deterministic compilation.

    compiler_expression:
        Internal alias-based FASTEXPR.

    brain_expression:
        Real-field FASTEXPR sent to BRAIN.

    payload:
        Exact payload passed to ace/simulator.
    """

    specification: dict[str, Any]

    compiler_expression: str

    brain_expression: str | None = None

    payload: dict[str, Any] | None = None

    validation_ok: bool = False

    validation_reason: str = ""


@dataclass
class PipelineCandidate:
    """
    Successfully prepared simulation candidate.
    """

    compiled: CompiledCandidate

    job: SimulationJob


@dataclass
class PipelineRun:
    """
    Complete output of a simulation stage.
    """

    candidates: list[PipelineCandidate]

    jobs: list[SimulationJob]

    raw_results: list[Any]

    normalized_results: list[Any]

    rejected: list[CandidateRejection]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_alpha_type(
    alpha_type: str,
) -> str:
    """
    Normalize NORMAL / POWER_POOL.
    """

    value = str(
        alpha_type
    ).strip().upper()

    if value not in {
        NORMAL,
        POWER_POOL,
    }:

        raise ValueError(
            "alpha_type must be "
            "'NORMAL' or 'POWER_POOL'."
        )

    return value


def normalize_template(
    template: Any,
) -> str:
    """
    Normalize template name.
    """

    return str(
        template or ""
    ).strip().upper()


# ============================================================
# OPERATOR EXTRACTION
# ============================================================

def extract_operator_names(
    expression: str,
    *,
    verified_operators: Iterable[str] = (),
) -> set[str]:
    """
    Extract operator names from FASTEXPR.
    """

    names = set(
        re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            str(expression),
        )
    )

    verified = {
        str(operator).strip()
        for operator
        in verified_operators
    }

    if verified:

        return {
            name
            for name
            in names
            if name in verified
        }

    return names


# ============================================================
# FIELD MAPPING
# ============================================================

def fields_to_aliases(
    field_ids: Sequence[str],
    *,
    id_to_alias: Mapping[str, str],
) -> list[str]:
    """
    Convert real BRAIN field IDs to compiler aliases.
    """

    aliases = []

    for field_id in field_ids:

        normalized = str(
            field_id
        ).strip()

        if not normalized:

            raise ValueError(
                "Field ID cannot be empty."
            )

        alias = id_to_alias.get(
            normalized
        )

        if alias is None:

            raise ValueError(
                "Unknown BRAIN field ID: "
                f"{normalized}"
            )

        aliases.append(
            str(alias).strip()
        )

    return aliases


# ============================================================
# ALIAS -> BRAIN TRANSLATION
# ============================================================

def translate_aliases_to_brain(
    expression: str,
    *,
    alias_to_id: Mapping[str, str],
) -> str:
    """
    Convert compiler aliases into real BRAIN field IDs.

    Example:

        rank(ts_delta(F127,60))

    becomes:

        rank(ts_delta(fnd6_newqus_rectq,60))
    """

    expression = str(
        expression
    ).strip()

    if not expression:

        raise ValueError(
            "expression cannot be empty."
        )

    if not alias_to_id:

        raise ValueError(
            "alias_to_id cannot be empty."
        )

    translated = expression

    aliases = sorted(
        (
            str(alias).strip()
            for alias
            in alias_to_id
        ),
        key=len,
        reverse=True,
    )

    for alias in aliases:

        if not alias:

            continue

        field_id = str(
            alias_to_id[
                alias
            ]
        ).strip()

        if not field_id:

            raise ValueError(
                "Empty field ID for alias: "
                f"{alias}"
            )

        translated = re.sub(
            rf"(?<![A-Za-z0-9_])"
            rf"{re.escape(alias)}"
            rf"(?![A-Za-z0-9_])",
            field_id,
            translated,
        )

    remaining_aliases = sorted(
        set(
            re.findall(
                r"(?<![A-Za-z0-9_])F\d+(?![A-Za-z0-9_])",
                translated,
            )
        )
    )

    if remaining_aliases:

        raise ValueError(
            "Untranslated compiler aliases remain: "
            + ", ".join(
                remaining_aliases
            )
        )

    return translated


# ============================================================
# PIPELINE
# ============================================================

class AlphaPipeline:
    """
    Main alpha research orchestration layer.
    """

    def __init__(
        self,
        *,
        session: Any,
        llm: ResearchLLM,
        compiler: Any,
        validator: Any,
        field_alias_to_id: Mapping[str, str],
        field_alias_to_type: Mapping[str, str],
        verified_operators: Iterable[str],
        allowed_windows: Iterable[int],
        region: str = DEFAULT_REGION,
        universe: str = DEFAULT_UNIVERSE,
        delay: int = DEFAULT_DELAY,
        decay: int = DEFAULT_DECAY,
        neutralization: str = DEFAULT_NEUTRALIZATION,
        truncation: float = DEFAULT_TRUNCATION,
        pasteurization: str = DEFAULT_PASTEURIZATION,
        test_period: str = DEFAULT_TEST_PERIOD,
        unit_handling: str = DEFAULT_UNIT_HANDLING,
        nan_handling: str = DEFAULT_NAN_HANDLING,
        max_trade: str = DEFAULT_MAX_TRADE,
        visualization: bool = DEFAULT_VISUALIZATION,
    ):
        self.session = session

        self.llm = llm

        self.compiler = compiler

        self.validator = validator

        self.field_alias_to_id = {
            str(alias).strip():
            str(field_id).strip()
            for alias, field_id
            in dict(field_alias_to_id).items()
        }

        self.field_alias_to_type = {
            str(alias).strip():
            str(field_type).strip().upper()
            for alias, field_type
            in dict(field_alias_to_type).items()
        }

        self.id_to_alias = {
            field_id: alias
            for alias, field_id
            in self.field_alias_to_id.items()
        }

        self.verified_operators = {
            str(operator).strip()
            for operator
            in verified_operators
        }

        self.allowed_windows = {
            int(window)
            for window
            in allowed_windows
        }

        if not self.allowed_windows:

            raise ValueError(
                "allowed_windows cannot be empty."
            )

        self.region = str(
            region
        ).strip()

        self.universe = str(
            universe
        ).strip()

        self.delay = int(
            delay
        )

        self.decay = int(
            decay
        )

        self.neutralization = str(
            neutralization
        ).strip()

        self.truncation = float(
            truncation
        )

        self.pasteurization = str(
            pasteurization
        ).strip().upper()

        self.test_period = str(
            test_period
        ).strip()

        self.unit_handling = str(
            unit_handling
        ).strip().upper()

        self.nan_handling = str(
            nan_handling
        ).strip().upper()

        self.max_trade = str(
            max_trade
        ).strip().upper()

        self.visualization = bool(
            visualization
        )

    # ========================================================
    # SPECIFICATION VALIDATION
    # ========================================================

    def validate_specification(
        self,
        specification: Mapping[str, Any],
    ) -> tuple[
        bool,
        str,
    ]:
        """
        Validate an LLM specification before compilation.
        """

        template = normalize_template(
            specification.get(
                "template"
            )
        )

        if not template:

            return (
                False,
                "template is missing.",
            )

        if not hasattr(
            self.compiler,
            "TEMPLATE_FIELD_COUNTS",
        ):

            return (
                False,
                "Compiler has no TEMPLATE_FIELD_COUNTS.",
            )

        template_counts = (
            self.compiler.TEMPLATE_FIELD_COUNTS
        )

        if template not in template_counts:

            return (
                False,
                f"Unknown template: {template}",
            )

        fields = specification.get(
            "fields",
            [],
        )

        if isinstance(
            fields,
            str,
        ):

            fields = [
                fields
            ]

        if not isinstance(
            fields,
            list,
        ):

            return (
                False,
                "fields must be a list.",
            )

        fields = [
            str(field).strip()
            for field
            in fields
        ]

        if any(
            not field
            for field
            in fields
        ):

            return (
                False,
                "fields contains an empty field ID.",
            )

        required_count = int(
            template_counts[
                template
            ]
        )

        if len(fields) != required_count:

            return (
                False,
                f"{template} requires "
                f"{required_count} fields, "
                f"got {len(fields)}.",
            )

        unknown_fields = [
            field
            for field
            in fields
            if field not in self.id_to_alias
        ]

        if unknown_fields:

            return (
                False,
                "Unknown BRAIN field IDs: "
                + ", ".join(
                    unknown_fields
                ),
            )

        try:

            window = int(
                specification.get(
                    "window",
                    60,
                )
            )

            backfill_window = int(
                specification.get(
                    "backfill_window",
                    60,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                False,
                "window and backfill_window must be integers.",
            )

        if window not in self.allowed_windows:

            return (
                False,
                f"Invalid window: {window}",
            )

        if (
            backfill_window
            not in self.allowed_windows
        ):

            return (
                False,
                f"Invalid backfill window: {backfill_window}",
            )

        direction = str(
            specification.get(
                "direction",
                "positive",
            )
        ).strip().lower()

        if direction not in {
            "positive",
            "negative",
        }:

            return (
                False,
                "direction must be 'positive' or 'negative'.",
            )

        return (
            True,
            "ok",
        )

    # ========================================================
    # COMPILE ONE
    # ========================================================

    def compile_specification(
        self,
        specification: Mapping[str, Any],
    ) -> CompiledCandidate:
        """
        Compile one research specification.

        Result still uses local compiler aliases.
        """

        specification = dict(
            specification
        )

        valid, reason = (
            self.validate_specification(
                specification
            )
        )

        if not valid:

            raise ValueError(
                reason
            )

        aliases = fields_to_aliases(
            specification[
                "fields"
            ],
            id_to_alias=self.id_to_alias,
        )

        compiled = self.compiler.compile(
            template=normalize_template(
                specification[
                    "template"
                ]
            ),
            fields=aliases,
            window=int(
                specification[
                    "window"
                ]
            ),
            backfill_window=int(
                specification[
                    "backfill_window"
                ]
            ),
            direction=str(
                specification.get(
                    "direction",
                    "positive",
                )
            ).strip().lower(),
        )

        compiler_expression = str(
            compiled.expression
        ).strip()

        if not compiler_expression:

            raise ValueError(
                "Compiler returned an empty expression."
            )

        valid, reason = (
            self.validator.validate(
                compiler_expression
            )
        )

        return CompiledCandidate(
            specification=specification,
            compiler_expression=compiler_expression,
            validation_ok=valid,
            validation_reason=reason,
        )

    # ========================================================
    # COMPILE MANY
    # ========================================================

    def compile_specifications(
        self,
        specifications: Iterable[
            Mapping[str, Any]
        ],
    ) -> tuple[
        list[CompiledCandidate],
        list[CandidateRejection],
    ]:

        compiled = []

        rejected = []

        for specification in specifications:

            specification = dict(
                specification
            )

            try:

                candidate = (
                    self.compile_specification(
                        specification
                    )
                )

            except Exception as exc:

                rejected.append(
                    CandidateRejection(
                        specification=specification,
                        stage="COMPILE",
                        reason=str(
                            exc
                        ),
                        error_type=type(
                            exc
                        ).__name__,
                    )
                )

                continue

            if not candidate.validation_ok:

                rejected.append(
                    CandidateRejection(
                        specification=specification,
                        stage="VALIDATE",
                        reason=(
                            candidate.validation_reason
                        ),
                        error_type="ValidationError",
                    )
                )

                continue

            compiled.append(
                candidate
            )

        return (
            compiled,
            rejected,
        )

    # ========================================================
    # TRANSLATE FOR BRAIN
    # ========================================================

    def prepare_brain_expression(
        self,
        compiler_expression: str,
    ) -> str:
        """
        Translate alias FASTEXPR into real-field FASTEXPR.

        This translation occurs only after validation.
        """

        brain_expression = (
            translate_aliases_to_brain(
                compiler_expression,
                alias_to_id=(
                    self.field_alias_to_id
                ),
            )
        )

        # ----------------------------------------------------
        # Sanity check:
        # no compiler aliases may remain.
        # ----------------------------------------------------

        if re.search(
            r"(?<![A-Za-z0-9_])F\d+(?![A-Za-z0-9_])",
            brain_expression,
        ):

            raise RuntimeError(
                "BRAIN expression still contains compiler aliases."
            )

        return brain_expression

    # ========================================================
    # CREATE BRAIN PAYLOAD
    # ========================================================

    def create_brain_payload(
        self,
        compiler_expression: str,
    ) -> tuple[
        str,
        dict[str, Any],
    ]:
        """
        Build the exact payload sent to BRAIN.
        """

        compiler_expression = str(
            compiler_expression
        ).strip()

        if not compiler_expression:

            raise ValueError(
                "compiler_expression cannot be empty."
            )

        brain_expression = (
            self.prepare_brain_expression(
                compiler_expression
            )
        )

        payload = ace.generate_alpha(
            regular=brain_expression,

            region=self.region,

            universe=self.universe,

            delay=self.delay,

            decay=self.decay,

            neutralization=self.neutralization,

            truncation=self.truncation,

            pasteurization=self.pasteurization,

            test_period=self.test_period,

            unit_handling=self.unit_handling,

            nan_handling=self.nan_handling,

            max_trade=self.max_trade,

            visualization=self.visualization,
        )

        if not isinstance(
            payload,
            dict,
        ):

            raise TypeError(
                "ace.generate_alpha() returned "
                f"{type(payload).__name__}, expected dict."
            )

        returned_expression = payload.get(
            "regular"
        )

        if not returned_expression:

            raise ValueError(
                "BRAIN payload has no regular expression."
            )

        # ----------------------------------------------------
        # Protect against ace_lib somehow returning an alias
        # expression after our explicit translation.
        # ----------------------------------------------------

        if re.search(
            r"(?<![A-Za-z0-9_])F\d+(?![A-Za-z0-9_])",
            str(returned_expression),
        ):

            raise RuntimeError(
                "ace.generate_alpha() returned an expression "
                "containing compiler aliases."
            )

        return (
            brain_expression,
            payload,
        )

    # ========================================================
    # CREATE ONE JOB
    # ========================================================

    def create_job(
        self,
        compiled: CompiledCandidate,
        *,
        alpha_type: str = NORMAL,
    ) -> PipelineCandidate:
        """
        Create a SimulationJob from one validated candidate.
        """

        if not compiled.validation_ok:

            raise ValueError(
                "Cannot create job from invalid expression: "
                + compiled.validation_reason
            )

        alpha_type = (
            normalize_alpha_type(
                alpha_type
            )
        )

        brain_expression, payload = (
            self.create_brain_payload(
                compiled.compiler_expression
            )
        )

        compiled.brain_expression = (
            brain_expression
        )

        compiled.payload = (
            payload
        )

        specification = (
            compiled.specification
        )

        job = create_job_from_payload(
            compiler_expression=(
                compiled.compiler_expression
            ),

            payload=payload,

            alpha_type=alpha_type,

            template=specification.get(
                "template"
            ),

            fields=list(
                specification.get(
                    "fields",
                    [],
                )
            ),

            parameters={
                "window": specification.get(
                    "window"
                ),

                "backfill_window": specification.get(
                    "backfill_window"
                ),

                "direction": specification.get(
                    "direction"
                ),

                "family": specification.get(
                    "family"
                ),

                "intuition": specification.get(
                    "intuition"
                ),

                "compiler_expression": (
                    compiled.compiler_expression
                ),

                "brain_expression": (
                    compiled.brain_expression
                ),
            },
        )

        return PipelineCandidate(
            compiled=compiled,
            job=job,
        )

    # ========================================================
    # CREATE MANY JOBS
    # ========================================================

    def create_jobs(
        self,
        compiled_candidates: Iterable[
            CompiledCandidate
        ],
        *,
        alpha_type: str = NORMAL,
    ) -> tuple[
        list[PipelineCandidate],
        list[CandidateRejection],
    ]:

        candidates = []

        rejected = []

        for compiled in compiled_candidates:

            try:

                candidate = self.create_job(
                    compiled,
                    alpha_type=alpha_type,
                )

            except Exception as exc:

                rejected.append(
                    CandidateRejection(
                        specification=dict(
                            compiled.specification
                        ),
                        stage="PAYLOAD",
                        reason=str(
                            exc
                        ),
                        error_type=type(
                            exc
                        ).__name__,
                    )
                )

                continue

            candidates.append(
                candidate
            )

        return (
            candidates,
            rejected,
        )

    # ========================================================
    # PREPARE
    # ========================================================

    def prepare(
        self,
        specifications: Iterable[
            Mapping[str, Any]
        ],
        *,
        alpha_type: str = NORMAL,
    ) -> tuple[
        list[PipelineCandidate],
        list[CandidateRejection],
    ]:

        compiled, compile_rejected = (
            self.compile_specifications(
                specifications
            )
        )

        candidates, payload_rejected = (
            self.create_jobs(
                compiled,
                alpha_type=alpha_type,
            )
        )

        return (
            candidates,
            [
                *compile_rejected,
                *payload_rejected,
            ],
        )

    # ========================================================
    # SIMULATE
    # ========================================================

    def simulate_jobs(
        self,
        candidates: Sequence[
            PipelineCandidate
        ],
        *,
        alpha_type: str | None = None,
        limit_of_multi_simulations: int = 10,
        num_workers: int | None = None,
        retry_interval_seconds: float = 30.0,
        max_wait_seconds: float | None = None,
        show_progress: bool = True,
        show_batch_progress: bool = False,
        status_file: str | None = "ace_queue_status.json",
    ) -> PipelineRun:
        """
        Submit prepared jobs to BRAIN.

        After SimulationRunner returns, raw_result is read directly
        from SimulationJob, which is the confirmed public result
        location.
        """

        if not candidates:

            return PipelineRun(
                candidates=[],
                jobs=[],
                raw_results=[],
                normalized_results=[],
                rejected=[],
            )

        jobs = [
            candidate.job
            for candidate
            in candidates
        ]

        runner = SimulationRunner(
            self.session
        )

        simulated_jobs = runner.simulate(
            jobs,

            limit_of_multi_simulations=(
                limit_of_multi_simulations
            ),

            num_workers=num_workers,

            retry_interval_seconds=(
                retry_interval_seconds
            ),

            max_wait_seconds=max_wait_seconds,

            show_progress=show_progress,

            show_batch_progress=show_batch_progress,

            status_file=status_file,
        )

        # ----------------------------------------------------
        # raw_result is a real SimulationJob field.
        # ----------------------------------------------------

        raw_results = [
            job.raw_result
            for job
            in simulated_jobs
            if job.raw_result is not None
        ]

        run = PipelineRun(
            candidates=list(
                candidates
            ),

            jobs=list(
                simulated_jobs
            ),

            raw_results=raw_results,

            normalized_results=[],

            rejected=[],
        )

        # ----------------------------------------------------
        # Automatically normalize actual successful BRAIN
        # results when alpha_type is provided.
        # ----------------------------------------------------

        if alpha_type is not None:

            self.normalize_run(
                run,
                alpha_type=alpha_type,
            )

        return run

    # ========================================================
    # RESULT HELPERS
    # ========================================================

    def normalize_record(
        self,
        record: Mapping[str, Any],
        *,
        alpha_type: str = NORMAL,
        specification: Mapping[str, Any] | None = None,
        compiler_expression: str = "",
    ) -> Any:
        """
        Pass one raw BRAIN result to results.py.
        """

        from engine import results

        specification = dict(
            specification
            or {}
        )

        fields = list(
            specification.get(
                "fields",
                [],
            )
        )

        operator_names = (
            extract_operator_names(
                compiler_expression,

                verified_operators=(
                    self.verified_operators
                ),
            )
        )

        return results.normalize_result(
            record,

            alpha_type=normalize_alpha_type(
                alpha_type
            ),

            fields=fields,

            operator_names=operator_names,

            template=specification.get(
                "template"
            ),

            compiler_expression=(
                compiler_expression
            ),
        )

    # ========================================================
    # NORMALIZE RUN
    # ========================================================

    def normalize_run(
        self,
        run: PipelineRun,
        *,
        alpha_type: str = NORMAL,
    ) -> PipelineRun:
        """
        Normalize raw SimulationJob results.

        Matching is performed by alpha_id.

        Failed jobs without alpha_id are not guessed or matched
        positionally.
        """

        alpha_type = normalize_alpha_type(
            alpha_type
        )

        records_by_alpha_id = {}

        for record in run.raw_results:

            if not isinstance(
                record,
                Mapping,
            ):

                continue

            alpha_id = record.get(
                "alpha_id"
            )

            if alpha_id is None:

                continue

            alpha_id = str(
                alpha_id
            ).strip()

            if not alpha_id:

                continue

            records_by_alpha_id[
                alpha_id
            ] = record

        normalized = []

        for candidate in run.candidates:

            job = candidate.job

            alpha_id = job.alpha_id

            if alpha_id is None:

                continue

            alpha_id = str(
                alpha_id
            ).strip()

            if not alpha_id:

                continue

            record = records_by_alpha_id.get(
                alpha_id
            )

            if record is None:

                continue

            try:

                result = (
                    self.normalize_record(
                        record,

                        alpha_type=alpha_type,

                        specification=(
                            candidate
                            .compiled
                            .specification
                        ),

                        compiler_expression=(
                            candidate
                            .compiled
                            .compiler_expression
                        ),
                    )
                )

            except Exception:

                continue

            normalized.append(
                result
            )

        run.normalized_results = (
            normalized
        )

        return run

    # ========================================================
    # GENERATE + PREPARE
    # ========================================================

    def generate_and_prepare(
        self,
        *,
        selected_fields: Sequence[
            Mapping[str, Any]
        ],
        candidate_count: int = 20,
        research_direction: str = "",
        previous_candidates: Sequence[
            Mapping[str, Any]
        ] | None = None,
        alpha_type: str = NORMAL,
    ) -> tuple[
        list[dict[str, Any]],
        list[PipelineCandidate],
        list[CandidateRejection],
    ]:
        """
        Generate LLM specifications and prepare BRAIN jobs.

        No simulation occurs.
        """

        specifications = (
            self.llm.generate_candidates(
                region=self.region,

                universe=self.universe,

                selected_fields=selected_fields,

                candidate_count=candidate_count,

                research_direction=research_direction,

                previous_candidates=(
                    previous_candidates
                ),
            )
        )

        candidates, rejected = (
            self.prepare(
                specifications,

                alpha_type=alpha_type,
            )
        )

        return (
            specifications,
            candidates,
            rejected,
        )

    # ========================================================
    # GENERATE + SIMULATE
    # ========================================================

    def generate_and_run(
        self,
        *,
        selected_fields: Sequence[
            Mapping[str, Any]
        ],
        candidate_count: int = 20,
        research_direction: str = "",
        previous_candidates: Sequence[
            Mapping[str, Any]
        ] | None = None,
        alpha_type: str = NORMAL,
        limit_of_multi_simulations: int = 10,
        num_workers: int | None = None,
        retry_interval_seconds: float = 30.0,
        max_wait_seconds: float | None = None,
        show_progress: bool = True,
        show_batch_progress: bool = False,
        status_file: str | None = "ace_queue_status.json",
    ) -> tuple[
        list[dict[str, Any]],
        PipelineRun,
    ]:
        """
        Full LLM -> compiler -> validator -> BRAIN run.
        """

        specifications, candidates, rejected = (
            self.generate_and_prepare(
                selected_fields=selected_fields,

                candidate_count=candidate_count,

                research_direction=research_direction,

                previous_candidates=(
                    previous_candidates
                ),

                alpha_type=alpha_type,
            )
        )

        run = self.simulate_jobs(
            candidates,

            alpha_type=alpha_type,

            limit_of_multi_simulations=(
                limit_of_multi_simulations
            ),

            num_workers=num_workers,

            retry_interval_seconds=(
                retry_interval_seconds
            ),

            max_wait_seconds=max_wait_seconds,

            show_progress=show_progress,

            show_batch_progress=show_batch_progress,

            status_file=status_file,
        )

        run.rejected.extend(
            rejected
        )

        return (
            specifications,
            run,
        )

    # ========================================================
    # RUN PRE-EXISTING SPECIFICATIONS
    # ========================================================

    def run(
        self,
        specifications: Iterable[
            Mapping[str, Any]
        ],
        *,
        alpha_type: str = NORMAL,
        limit_of_multi_simulations: int = 10,
        num_workers: int | None = None,
        retry_interval_seconds: float = 30.0,
        max_wait_seconds: float | None = None,
        show_progress: bool = True,
        show_batch_progress: bool = False,
        status_file: str | None = "ace_queue_status.json",
    ) -> PipelineRun:
        """
        Compile, validate, submit and normalize results.
        """

        candidates, rejected = (
            self.prepare(
                specifications,

                alpha_type=alpha_type,
            )
        )

        run = self.simulate_jobs(
            candidates,

            alpha_type=alpha_type,

            limit_of_multi_simulations=(
                limit_of_multi_simulations
            ),

            num_workers=num_workers,

            retry_interval_seconds=(
                retry_interval_seconds
            ),

            max_wait_seconds=max_wait_seconds,

            show_progress=show_progress,

            show_batch_progress=show_batch_progress,

            status_file=status_file,
        )

        run.rejected.extend(
            rejected
        )

        return run


# ============================================================
# FACTORY
# ============================================================

def create_pipeline(
    *,
    session: Any,
    llm: ResearchLLM,
    compiler: Any,
    validator: Any,
    field_alias_to_id: Mapping[str, str],
    field_alias_to_type: Mapping[str, str],
    verified_operators: Iterable[str],
    allowed_windows: Iterable[int],
    region: str = DEFAULT_REGION,
    universe: str = DEFAULT_UNIVERSE,
    delay: int = DEFAULT_DELAY,
    decay: int = DEFAULT_DECAY,
    neutralization: str = DEFAULT_NEUTRALIZATION,
    truncation: float = DEFAULT_TRUNCATION,
    pasteurization: str = DEFAULT_PASTEURIZATION,
    test_period: str = DEFAULT_TEST_PERIOD,
    unit_handling: str = DEFAULT_UNIT_HANDLING,
    nan_handling: str = DEFAULT_NAN_HANDLING,
    max_trade: str = DEFAULT_MAX_TRADE,
    visualization: bool = DEFAULT_VISUALIZATION,
) -> AlphaPipeline:

    return AlphaPipeline(
        session=session,

        llm=llm,

        compiler=compiler,

        validator=validator,

        field_alias_to_id=field_alias_to_id,

        field_alias_to_type=field_alias_to_type,

        verified_operators=verified_operators,

        allowed_windows=allowed_windows,

        region=region,

        universe=universe,

        delay=delay,

        decay=decay,

        neutralization=neutralization,

        truncation=truncation,

        pasteurization=pasteurization,

        test_period=test_period,

        unit_handling=unit_handling,

        nan_handling=nan_handling,

        max_trade=max_trade,

        visualization=visualization,
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "NORMAL",
    "POWER_POOL",

    "CandidateRejection",
    "CompiledCandidate",
    "PipelineCandidate",
    "PipelineRun",

    "AlphaPipeline",

    "extract_operator_names",
    "fields_to_aliases",
    "translate_aliases_to_brain",
    "normalize_alpha_type",

    "create_pipeline",
]