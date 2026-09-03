"""
simulation_records.py

Bookkeeping layer for WorldQuant BRAIN simulations.

Responsibilities:
- Track the exact compiler expression and final BRAIN expression.
- Track the payload submitted for each alpha.
- Preserve per-alpha success/failure information.
- Preserve error information instead of reducing failures to alpha_id=None.
- Support NORMAL and POWER_POOL labels.

This module does not authenticate, submit, or score alphas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import traceback
import uuid


# ============================================================
# STATUS CONSTANTS
# ============================================================

CREATED = "CREATED"
SUBMITTED = "SUBMITTED"
SIMULATED = "SIMULATED"
FAILED = "FAILED"


# ============================================================
# ALPHA TYPE CONSTANTS
# ============================================================

NORMAL = "NORMAL"
POWER_POOL = "POWER_POOL"


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert a mapping to a plain dict."""
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(
        f"Expected a mapping, got {type(value).__name__}."
    )


def extract_brain_expression(
    payload: Mapping[str, Any],
) -> str:
    """
    Extract the final expression from an ace_lib payload.

    REGULAR:
        payload["regular"]

    SUPER:
        payload does not contain one ordinary regular expression,
        so selection/combo are represented together.
    """
    alpha_type = str(
        payload.get("type", "REGULAR")
    ).upper()

    if alpha_type == "REGULAR":
        value = payload.get("regular", "")
        return "" if value is None else str(value).strip()

    if alpha_type == "SUPER":
        parts: list[str] = []

        if payload.get("selection") is not None:
            parts.append(
                f"selection={payload['selection']}"
            )

        if payload.get("combo") is not None:
            parts.append(
                f"combo={payload['combo']}"
            )

        return " | ".join(parts)

    return ""


# ============================================================
# SIMULATION JOB
# ============================================================

@dataclass
class SimulationJob:
    """
    Tracks one alpha from creation through simulation.

    The record is created BEFORE submission so that a later
    BRAIN failure can always be tied back to its expression.
    """

    job_id: str
    compiler_expression: str
    brain_expression: str
    payload: dict[str, Any]

    alpha_type: str = NORMAL
    template: str | None = None
    fields: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    status: str = CREATED
    alpha_id: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    error_response: Any = None
    traceback_text: str | None = None

    raw_result: Any = None

    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def mark_submitted(self) -> None:
        """Mark this job as submitted to BRAIN."""
        self.status = SUBMITTED
        self.updated_at = utc_now_iso()

    def mark_success(
        self,
        raw_result: Any,
        alpha_id: str | None = None,
    ) -> None:
        """Attach a successful simulation result."""
        self.status = SIMULATED
        self.raw_result = raw_result

        if alpha_id is None and isinstance(raw_result, Mapping):
            candidate = raw_result.get("alpha_id")
            if candidate is not None:
                alpha_id = str(candidate)

        self.alpha_id = alpha_id
        self.error_type = None
        self.error_message = None
        self.error_response = None
        self.traceback_text = None
        self.updated_at = utc_now_iso()

    def mark_failure(
        self,
        error: BaseException | None = None,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        error_response: Any = None,
        traceback_text: str | None = None,
        raw_result: Any = None,
    ) -> None:
        """Attach a failure without losing the tracked expression."""
        self.status = FAILED

        if error is not None:
            if error_type is None:
                error_type = type(error).__name__

            if error_message is None:
                error_message = str(error)

            if traceback_text is None:
                traceback_text = traceback.format_exc()

        self.error_type = error_type
        self.error_message = error_message
        self.error_response = error_response
        self.traceback_text = traceback_text
        self.raw_result = raw_result
        self.updated_at = utc_now_iso()

    @property
    def is_success(self) -> bool:
        return self.status == SIMULATED

    @property
    def is_failure(self) -> bool:
        return self.status == FAILED

    def failure_text(self) -> str:
        """
        Produce a diagnostic containing the exact expression.

        This is intentionally expression-first for batch debugging.
        """
        lines = [
            "SIMULATION FAILED",
            f"Job ID: {self.job_id}",
            f"Alpha type: {self.alpha_type}",
            f"Compiler expression: {self.compiler_expression}",
            f"BRAIN expression: {self.brain_expression}",
        ]

        if self.template:
            lines.append(
                f"Template: {self.template}"
            )

        if self.fields:
            lines.append(
                "Fields: " + ", ".join(self.fields)
            )

        if self.error_type:
            lines.append(
                f"Error type: {self.error_type}"
            )

        if self.error_message:
            lines.append(
                f"Error: {self.error_message}"
            )

        if self.error_response is not None:
            lines.append(
                f"Server response: {self.error_response}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the full record as a dictionary."""
        return asdict(self)


# ============================================================
# JOB CONSTRUCTION
# ============================================================

def create_job(
    *,
    compiler_expression: str,
    brain_expression: str,
    payload: Mapping[str, Any],
    alpha_type: str = NORMAL,
    template: str | None = None,
    fields: Iterable[str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> SimulationJob:
    """
    Create a SimulationJob before submitting to BRAIN.
    """
    compiler_expression = str(
        compiler_expression
    ).strip()

    brain_expression = str(
        brain_expression
    ).strip()

    if not compiler_expression:
        raise ValueError(
            "compiler_expression cannot be empty."
        )

    if not brain_expression:
        raise ValueError(
            "brain_expression cannot be empty."
        )

    alpha_type = str(
        alpha_type
    ).strip().upper()

    if alpha_type not in {
        NORMAL,
        POWER_POOL,
    }:
        raise ValueError(
            "alpha_type must be NORMAL or POWER_POOL."
        )

    normalized_fields = [
        str(value).strip()
        for value in (fields or [])
    ]

    normalized_parameters = (
        dict(parameters)
        if parameters is not None
        else {}
    )

    return SimulationJob(
        job_id=job_id or uuid.uuid4().hex,
        compiler_expression=compiler_expression,
        brain_expression=brain_expression,
        payload=_as_dict(payload),
        alpha_type=alpha_type,
        template=template,
        fields=normalized_fields,
        parameters=normalized_parameters,
    )


def create_job_from_payload(
    *,
    compiler_expression: str,
    payload: Mapping[str, Any],
    alpha_type: str = NORMAL,
    template: str | None = None,
    fields: Iterable[str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> SimulationJob:
    """
    Create a job when the final BRAIN expression is already in the payload.
    """
    payload_dict = _as_dict(payload)

    brain_expression = extract_brain_expression(
        payload_dict
    )

    return create_job(
        compiler_expression=compiler_expression,
        brain_expression=brain_expression,
        payload=payload_dict,
        alpha_type=alpha_type,
        template=template,
        fields=fields,
        parameters=parameters,
        job_id=job_id,
    )


# ============================================================
# ACE LIB RESULT ATTACHMENT
# ============================================================

def attach_ace_result(
    job: SimulationJob,
    result: Any,
) -> SimulationJob:
    """
    Attach the result returned by ace_lib.

    Successful ace_lib result:
        {"alpha_id": "...", ...}

    Existing ace_lib failure path:
        {"alpha_id": None, "simulate_data": ...}

    A failure is kept as FAILED and never loses the expression
    already stored in the job.
    """
    if not isinstance(result, Mapping):
        job.mark_failure(
            error_type="UnexpectedResultType",
            error_message=(
                "ace_lib returned "
                f"{type(result).__name__}, expected a mapping."
            ),
            raw_result=result,
        )
        return job

    alpha_id = result.get("alpha_id")

    if alpha_id is not None:
        job.mark_success(
            raw_result=dict(result),
            alpha_id=str(alpha_id),
        )
        return job

    simulate_data = result.get(
        "simulate_data"
    )

    error_type = result.get(
        "error_type"
    )

    error_message = result.get(
        "error_message"
    )

    error_response = result.get(
        "error_response"
    )

    if error_message is None:
        error_message = result.get("error")

    if error_message is None and error_response is None:
        error_message = (
            "BRAIN/ace_lib returned no alpha_id."
        )

    job.mark_failure(
        error_type=error_type or "SimulationError",
        error_message=error_message,
        error_response=error_response,
        raw_result=dict(result),
    )

    if isinstance(simulate_data, Mapping):
        job.payload = dict(simulate_data)

        returned_expression = extract_brain_expression(
            simulate_data
        )

        if returned_expression:
            job.brain_expression = returned_expression

    return job


# ============================================================
# REGISTRY
# ============================================================

class SimulationRegistry:
    """
    In-memory collection of SimulationJob objects.

    This registry is independent of ace_lib and can sit above
    either single-alpha or multi-alpha simulation calls.
    """

    def __init__(
        self,
        jobs: Iterable[SimulationJob] | None = None,
    ):
        self._jobs: dict[str, SimulationJob] = {}

        for job in jobs or []:
            self.add(job)

    def add(
        self,
        job: SimulationJob,
    ) -> None:
        """Add one job."""
        if job.job_id in self._jobs:
            raise ValueError(
                f"Duplicate job_id: {job.job_id}"
            )

        self._jobs[job.job_id] = job

    def get(
        self,
        job_id: str,
    ) -> SimulationJob:
        """Get one job."""
        if job_id not in self._jobs:
            raise KeyError(
                f"Unknown simulation job: {job_id}"
            )

        return self._jobs[job_id]

    def jobs(
        self,
    ) -> list[SimulationJob]:
        """Return jobs in insertion order."""
        return list(self._jobs.values())

    def by_status(
        self,
        status: str,
    ) -> list[SimulationJob]:
        """Return all jobs with the requested status."""
        return [
            job
            for job in self._jobs.values()
            if job.status == status
        ]

    def successful_jobs(
        self,
    ) -> list[SimulationJob]:
        return self.by_status(SIMULATED)

    def failed_jobs(
        self,
    ) -> list[SimulationJob]:
        return self.by_status(FAILED)

    def summary(
        self,
    ) -> dict[str, int]:
        """Return status counts."""
        counts = {
            CREATED: 0,
            SUBMITTED: 0,
            SIMULATED: 0,
            FAILED: 0,
        }

        for job in self._jobs.values():
            if job.status in counts:
                counts[job.status] += 1

        return counts

    def failure_report(
        self,
    ) -> str:
        """Return a detailed report of every failed expression."""
        failures = self.failed_jobs()

        if not failures:
            return "No failed simulations."

        lines = [
            "=" * 80,
            "FAILED SIMULATIONS",
            "=" * 80,
        ]

        for index, job in enumerate(
            failures,
            start=1,
        ):
            lines.extend(
                [
                    "",
                    f"[{index}]",
                    job.failure_text(),
                ]
            )

        return "\n".join(lines)

    def to_rows(
        self,
    ) -> list[dict[str, Any]]:
        """Flatten job bookkeeping into rows."""
        rows = []

        for job in self.jobs():
            rows.append({
                "job_id": job.job_id,
                "alpha_type": job.alpha_type,
                "template": job.template,
                "compiler_expression": job.compiler_expression,
                "brain_expression": job.brain_expression,
                "status": job.status,
                "alpha_id": job.alpha_id,
                "error_type": job.error_type,
                "error_message": job.error_message,
                "fields": list(job.fields),
                "parameters": dict(job.parameters),
                "payload": job.payload,
                "raw_result": job.raw_result,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            })

        return rows


# ============================================================
# PRINT HELPERS
# ============================================================

def print_failed_jobs(
    jobs: Iterable[SimulationJob],
) -> None:
    """Print failed simulations and their exact expressions."""
    failed = [
        job
        for job in jobs
        if job.status == FAILED
    ]

    if not failed:
        print("No failed simulations.")
        return

    print("=" * 80)
    print("FAILED SIMULATIONS")
    print("=" * 80)

    for index, job in enumerate(
        failed,
        start=1,
    ):
        print()
        print(
            f"[{index}] {job.job_id}"
        )
        print(
            f"Compiler expression: {job.compiler_expression}"
        )
        print(
            f"BRAIN expression: {job.brain_expression}"
        )

        if job.template:
            print(
                f"Template: {job.template}"
            )

        if job.error_type:
            print(
                f"Error type: {job.error_type}"
            )

        if job.error_message:
            print(
                f"Error: {job.error_message}"
            )

        if job.error_response is not None:
            print(
                f"Server response: {job.error_response}"
            )


def print_registry_summary(
    registry: SimulationRegistry,
) -> None:
    """Print a compact job-status summary."""
    counts = registry.summary()

    print("=" * 80)
    print("SIMULATION REGISTRY")
    print("=" * 80)
    print(
        f"Created:   {counts[CREATED]}"
    )
    print(
        f"Submitted: {counts[SUBMITTED]}"
    )
    print(
        f"Simulated: {counts[SIMULATED]}"
    )
    print(
        f"Failed:    {counts[FAILED]}"
    )