# ============================================================
# engine/simulator.py
# ============================================================

"""
WorldQuant BRAIN simulation wrapper.

Responsibilities
----------------
- Use an already-authenticated BRAIN session.
- Track every simulation BEFORE submission.
- Preserve compiler expression -> real BRAIN expression mapping.
- Submit through the existing ace_lib queue.
- Match returned records by exact BRAIN expression.
- Preserve failures and the expression that failed.
- Keep simulation bookkeeping separate from alpha scoring.

This module NEVER calls start_session().
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import pandas as pd

import ace_lib as ace

from engine.simulation_records import (
    FAILED,
    SIMULATED,
    NORMAL,
    POWER_POOL,
    SimulationJob,
    SimulationRegistry,
    attach_ace_result,
    create_job_from_payload,
    extract_brain_expression,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_LIMIT_OF_MULTI_SIMULATIONS = 10
DEFAULT_NUM_WORKERS = None

DEFAULT_RETRY_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = None

DEFAULT_SHOW_PROGRESS = True
DEFAULT_SHOW_BATCH_PROGRESS = False

DEFAULT_STATUS_FILE = "ace_queue_status.json"


# ============================================================
# HELPERS
# ============================================================

def _is_mapping(
    value: Any,
) -> bool:
    return isinstance(
        value,
        Mapping,
    )


def _copy_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not _is_mapping(payload):
        raise TypeError(
            "Simulation payload must be a dict/mapping."
        )

    return dict(payload)


def _result_expression(
    result: Any,
) -> str:
    """
    Extract the final BRAIN expression from an ace_lib result.
    """
    if not _is_mapping(result):
        return ""

    simulate_data = result.get(
        "simulate_data"
    )

    if not _is_mapping(
        simulate_data
    ):
        return ""

    return extract_brain_expression(
        simulate_data
    )


def _result_alpha_id(
    result: Any,
) -> str | None:
    if not _is_mapping(result):
        return None

    value = result.get(
        "alpha_id"
    )

    if value is None:
        return None

    return str(value)


# ============================================================
# RUNNER
# ============================================================

class SimulationRunner:
    """
    High-level wrapper around ace_lib's BRAIN simulation queue.

    The authenticated session is supplied by the caller.
    """

    def __init__(
        self,
        session,
    ):
        if session is None:
            raise ValueError(
                "session cannot be None."
            )

        self.session = session

        self.registry = SimulationRegistry()

    # ========================================================
    # JOB CREATION
    # ========================================================

    def create_jobs(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        alpha_type: str = NORMAL,
    ) -> list[SimulationJob]:
        """
        Create SimulationJob objects from records containing:

            payload
            compiler_expression / expression
            template
            fields
            parameters

        The payload must already contain the final BRAIN expression.
        """

        jobs = []

        for item in items:

            if not _is_mapping(item):
                raise TypeError(
                    "Every simulation item must be a mapping."
                )

            payload = item.get(
                "payload"
            )

            if not _is_mapping(
                payload
            ):
                raise ValueError(
                    "Simulation item must contain "
                    "a dict under 'payload'."
                )

            payload = _copy_payload(
                payload
            )

            compiler_expression = str(
                item.get(
                    "compiler_expression",
                    item.get(
                        "expression",
                        "",
                    ),
                )
                or ""
            ).strip()

            if not compiler_expression:
                raise ValueError(
                    "Simulation item is missing "
                    "'compiler_expression' or 'expression'."
                )

            job = create_job_from_payload(
                compiler_expression=(
                    compiler_expression
                ),
                payload=payload,
                alpha_type=alpha_type,
                template=item.get(
                    "template"
                ),
                fields=item.get(
                    "fields",
                    [],
                ),
                parameters=item.get(
                    "parameters",
                    {},
                ),
            )

            jobs.append(
                job
            )

        return jobs

    # ========================================================
    # REGISTER
    # ========================================================

    def register_jobs(
        self,
        jobs: Iterable[SimulationJob],
    ) -> None:
        """
        Add jobs to the registry before submission.
        """

        for job in jobs:

            self.registry.add(
                job
            )

    # ========================================================
    # MATCH RETURNED RESULTS
    # ========================================================

    def _match_results(
        self,
        jobs: list[SimulationJob],
        raw_results: list[Any],
    ) -> set[str]:
        """
        Match returned records to jobs by exact BRAIN expression.

        There is intentionally NO positional fallback.

        Why:
            A failed multi-simulation may return a different ordering
            or an incomplete set of records. Positional matching could
            therefore assign an error to the wrong alpha.

        For duplicate identical expressions, assignment among those jobs
        is necessarily interchangeable because the BRAIN expression is
        identical.
        """

        jobs_by_expression: dict[
            str,
            list[SimulationJob],
        ] = defaultdict(list)

        for job in jobs:
            jobs_by_expression[
                job.brain_expression
            ].append(
                job
            )

        matched_ids: set[str] = set()

        for raw_result in raw_results:

            expression = _result_expression(
                raw_result
            )

            if not expression:

                continue

            candidates = (
                jobs_by_expression.get(
                    expression,
                    [],
                )
            )

            target_job = None

            for candidate in candidates:

                if (
                    candidate.job_id
                    not in matched_ids
                ):

                    target_job = candidate
                    break

            if target_job is None:
                continue

            attach_ace_result(
                target_job,
                raw_result,
            )

            matched_ids.add(
                target_job.job_id
            )

        return matched_ids

    # ========================================================
    # BATCH-LEVEL FAILURE
    # ========================================================

    def _mark_unmatched_jobs_failed(
        self,
        jobs: list[SimulationJob],
        matched_ids: set[str],
    ) -> None:
        """
        Mark every job for which ace_lib returned no identifiable
        record as failed.

        We do NOT guess which expression failed.
        """

        for job in jobs:

            if job.job_id in matched_ids:
                continue

            if job.status == SIMULATED:
                continue

            job.mark_failure(
                error_type="MissingSimulationResult",
                error_message=(
                    "ace_lib returned no result record "
                    "identifiable by the exact BRAIN expression."
                ),
            )

    # ========================================================
    # DISPLAY FAILURES
    # ========================================================

    @staticmethod
    def _print_failures(
        jobs: list[SimulationJob],
    ) -> None:

        failed = [
            job
            for job in jobs
            if job.status == FAILED
        ]

        if not failed:
            return

        print()
        print(
            "=" * 80
        )
        print(
            "FAILED EXPRESSIONS"
        )
        print(
            "=" * 80
        )

        for index, job in enumerate(
            failed,
            start=1,
        ):

            print()
            print(
                f"[{index}]"
            )

            print(
                f"Compiler expression: "
                f"{job.compiler_expression}"
            )

            print(
                f"BRAIN expression: "
                f"{job.brain_expression}"
            )

            if job.template:
                print(
                    f"Template: {job.template}"
                )

            if job.fields:
                print(
                    "Fields: "
                    + ", ".join(job.fields)
                )

            if job.error_type:
                print(
                    f"Error type: "
                    f"{job.error_type}"
                )

            if job.error_message:
                print(
                    f"Error: "
                    f"{job.error_message}"
                )

            if job.error_response is not None:
                print(
                    f"Server response: "
                    f"{job.error_response}"
                )

    # ========================================================
    # SIMULATE
    # ========================================================

    def simulate(
        self,
        jobs: Iterable[SimulationJob],
        *,
        limit_of_multi_simulations: int = (
            DEFAULT_LIMIT_OF_MULTI_SIMULATIONS
        ),
        num_workers: int | None = (
            DEFAULT_NUM_WORKERS
        ),
        retry_interval_seconds: float = (
            DEFAULT_RETRY_INTERVAL_SECONDS
        ),
        max_wait_seconds: float | None = (
            DEFAULT_MAX_WAIT_SECONDS
        ),
        show_progress: bool = (
            DEFAULT_SHOW_PROGRESS
        ),
        show_batch_progress: bool = (
            DEFAULT_SHOW_BATCH_PROGRESS
        ),
        status_file: str | None = (
            DEFAULT_STATUS_FILE
        ),
    ) -> list[SimulationJob]:
        """
        Submit a batch through ace.simulate_alpha_queue().
        """

        jobs = list(
            jobs
        )

        if not jobs:
            return []

        # ----------------------------------------------------
        # Register before submission.
        # ----------------------------------------------------

        known_job_ids = {
            job.job_id
            for job
            in self.registry.jobs()
        }

        for job in jobs:

            if job.job_id not in known_job_ids:

                self.registry.add(
                    job
                )

        for job in jobs:

            if (
                job.status != SIMULATED
                and job.status != FAILED
            ):
                job.mark_submitted()

        payloads = [
            job.payload
            for job
            in jobs
        ]

        print(
            "=" * 80
        )

        print(
            "BRAIN SIMULATION"
        )

        print(
            "=" * 80
        )

        print(
            "Expressions:",
            len(jobs),
        )

        # ----------------------------------------------------
        # Submit.
        # ----------------------------------------------------

        try:

            raw_results = (
                ace.simulate_alpha_queue(
                    self.session,
                    payloads,
                    limit_of_multi_simulations=(
                        limit_of_multi_simulations
                    ),
                    num_workers=(
                        num_workers
                    ),
                    retry_interval_seconds=(
                        retry_interval_seconds
                    ),
                    max_wait_seconds=(
                        max_wait_seconds
                    ),
                    show_progress=(
                        show_progress
                    ),
                    show_batch_progress=(
                        show_batch_progress
                    ),
                    status_file=status_file,
                )
            )

        except Exception as exc:

            print()
            print(
                "=" * 80
            )
            print(
                "BATCH-LEVEL EXCEPTION"
            )
            print(
                "=" * 80
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )

            for job in jobs:

                job.mark_failure(
                    error=exc
                )

            self._print_failures(
                jobs
            )

            return jobs

        # ----------------------------------------------------
        # Normalize result container.
        # ----------------------------------------------------

        if raw_results is None:
            raw_results = []

        elif not isinstance(
            raw_results,
            list,
        ):
            raw_results = [
                raw_results
            ]

        # ----------------------------------------------------
        # Match ONLY by exact BRAIN expression.
        # ----------------------------------------------------

        matched_ids = (
            self._match_results(
                jobs,
                raw_results,
            )
        )

        # ----------------------------------------------------
        # Any unmatched job is failed, but we never pretend
        # to know a more specific error than we actually have.
        # ----------------------------------------------------

        self._mark_unmatched_jobs_failed(
            jobs,
            matched_ids,
        )

        # ----------------------------------------------------
        # Print exact failed expressions.
        # ----------------------------------------------------

        self._print_failures(
            jobs
        )

        # ----------------------------------------------------
        # Summary.
        # ----------------------------------------------------

        simulated_count = sum(
            job.status == SIMULATED
            for job
            in jobs
        )

        failed_count = sum(
            job.status == FAILED
            for job
            in jobs
        )

        print()
        print(
            "=" * 80
        )
        print(
            "SIMULATION COMPLETE"
        )
        print(
            "=" * 80
        )

        print(
            "Simulated:",
            simulated_count,
        )

        print(
            "Failed:",
            failed_count,
        )

        return jobs

    # ========================================================
    # PAYLOAD CONVENIENCE API
    # ========================================================

    def simulate_payloads(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        compiler_expressions: Iterable[str] | None = None,
        alpha_type: str = NORMAL,
        templates: Iterable[str | None] | None = None,
        fields: Iterable[Iterable[str]] | None = None,
        parameters: Iterable[Mapping[str, Any]] | None = None,
        **simulation_kwargs,
    ) -> list[SimulationJob]:
        """
        Create jobs from payloads and submit them.
        """

        payloads = [
            _copy_payload(
                payload
            )
            for payload
            in payloads
        ]

        count = len(
            payloads
        )

        # ----------------------------------------------------
        # Compiler expressions.
        # ----------------------------------------------------

        if compiler_expressions is None:

            compiler_expressions = [
                extract_brain_expression(
                    payload
                )
                for payload
                in payloads
            ]

        else:

            compiler_expressions = list(
                compiler_expressions
            )

            if len(
                compiler_expressions
            ) != count:

                raise ValueError(
                    "compiler_expressions length "
                    "must match payloads length."
                )

        # ----------------------------------------------------
        # Templates.
        # ----------------------------------------------------

        if templates is None:

            templates = [
                None
                for _ in range(count)
            ]

        else:

            templates = list(
                templates
            )

            if len(
                templates
            ) != count:

                raise ValueError(
                    "templates length "
                    "must match payloads length."
                )

        # ----------------------------------------------------
        # Fields.
        # ----------------------------------------------------

        if fields is None:

            fields = [
                []
                for _ in range(count)
            ]

        else:

            fields = [
                list(value)
                for value
                in fields
            ]

            if len(
                fields
            ) != count:

                raise ValueError(
                    "fields length "
                    "must match payloads length."
                )

        # ----------------------------------------------------
        # Parameters.
        # ----------------------------------------------------

        if parameters is None:

            parameters = [
                {}
                for _ in range(count)
            ]

        else:

            parameters = [
                dict(value)
                for value
                in parameters
            ]

            if len(
                parameters
            ) != count:

                raise ValueError(
                    "parameters length "
                    "must match payloads length."
                )

        # ----------------------------------------------------
        # Build jobs.
        # ----------------------------------------------------

        jobs = []

        for index in range(
            count
        ):

            job = create_job_from_payload(
                compiler_expression=(
                    compiler_expressions[
                        index
                    ]
                ),
                payload=(
                    payloads[
                        index
                    ]
                ),
                alpha_type=alpha_type,
                template=(
                    templates[
                        index
                    ]
                ),
                fields=(
                    fields[
                        index
                    ]
                ),
                parameters=(
                    parameters[
                        index
                    ]
                ),
            )

            jobs.append(
                job
            )

        return self.simulate(
            jobs,
            **simulation_kwargs,
        )

    # ========================================================
    # QUERY METHODS
    # ========================================================

    def successful_jobs(
        self,
    ) -> list[SimulationJob]:

        return self.registry.successful_jobs()

    def failed_jobs(
        self,
    ) -> list[SimulationJob]:

        return self.registry.failed_jobs()

    def summary(
        self,
    ) -> dict[str, int]:

        return self.registry.summary()

    def failure_report(
        self,
    ) -> str:

        return self.registry.failure_report()

    def rows(
        self,
    ) -> list[dict[str, Any]]:

        return self.registry.to_rows()

    def dataframe(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            self.rows()
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    def print_failures(
        self,
    ) -> None:

        self._print_failures(
            self.registry.jobs()
        )

    def print_summary(
        self,
    ) -> None:

        counts = self.summary()

        print(
            "=" * 80
        )

        print(
            "SIMULATION REGISTRY"
        )

        print(
            "=" * 80
        )

        print(
            f"Created:   {counts['CREATED']}"
        )

        print(
            f"Submitted: {counts['SUBMITTED']}"
        )

        print(
            f"Simulated: {counts['SIMULATED']}"
        )

        print(
            f"Failed:    {counts['FAILED']}"
        )


# ============================================================
# FUNCTIONAL API
# ============================================================

def simulate_alpha_queue(
    session,
    jobs: Iterable[SimulationJob],
    **kwargs,
) -> list[SimulationJob]:
    """
    Functional convenience wrapper.
    """

    runner = SimulationRunner(
        session
    )

    return runner.simulate(
        jobs,
        **kwargs,
    )


# ============================================================
# DATAFRAME HELPER
# ============================================================

def jobs_to_dataframe(
    jobs: Iterable[SimulationJob],
) -> pd.DataFrame:

    rows = []

    for job in jobs:

        rows.append({
            "job_id": job.job_id,
            "alpha_type": job.alpha_type,
            "template": job.template,
            "compiler_expression": (
                job.compiler_expression
            ),
            "brain_expression": (
                job.brain_expression
            ),
            "status": job.status,
            "alpha_id": job.alpha_id,
            "error_type": job.error_type,
            "error_message": job.error_message,
            "fields": job.fields,
            "parameters": job.parameters,
        })

    return pd.DataFrame(
        rows
    )