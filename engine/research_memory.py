"""
Persistent research memory for the alpha-research system.

The memory layer stores experiment outcomes and derived research scores so that
future research iterations can learn from prior experiments.

Storage format:
    JSON Lines (JSONL), one experiment per line.

Design goals:
    - simple
    - human-readable
    - append-friendly
    - restart-safe
    - deterministic
    - no database or vector store dependency
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MEMORY_PATH = Path("research_memory.jsonl")
MEMORY_VERSION = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResearchMemoryRecord:
    """
    One persisted research experiment.

    The record intentionally stores both the alpha metadata and the research
    score. Raw BRAIN output is optional because it can be large.
    """

    memory_version: int = MEMORY_VERSION
    created_at: str = ""

    alpha_id: str | None = None
    expression: str = ""
    compiler_expression: str = ""

    alpha_type: str = "NORMAL"
    simulation_status: str = ""
    eligibility_status: str = ""

    fields: list[str] = field(default_factory=list)
    template: str | None = None

    research_score: float | None = None
    research_class: str | None = None

    oos_score: float | None = None
    consistency_score: float | None = None
    turnover_score: float | None = None
    robustness_score: float | None = None

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    failed_gates: list[str] = field(default_factory=list)
    failed_brain_tests: list[str] = field(default_factory=list)
    warning_brain_tests: list[str] = field(default_factory=list)

    # Optional metrics copied from PerformanceMetrics for easy downstream use.
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

    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Convert a value to float where possible."""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[str]:
    """Convert an iterable of values to a list of strings."""

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    try:
        return [
            str(item)
            for item in value
            if item is not None
        ]
    except TypeError:
        return [str(value)]


def _json_safe(value: Any) -> Any:
    """
    Convert common Python/pandas values into JSON-safe structures.

    Raw BRAIN records can contain DataFrames, NaN values, timestamps, etc.
    Memory deliberately avoids storing those large raw objects by default.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            try:
                if value != value or value in (float("inf"), float("-inf")):
                    return None
            except Exception:
                pass
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return str(value)


def _extract_metric(result: Any, name: str) -> float | None:
    """Read a metric from NormalizedResult.metrics."""

    metrics = getattr(result, "metrics", None)

    if metrics is None:
        return None

    return _safe_float(
        getattr(metrics, name, None)
    )


# ---------------------------------------------------------------------------
# Record creation
# ---------------------------------------------------------------------------

def record_from_result(
    result: Any,
    *,
    research_score: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResearchMemoryRecord:
    """
    Build a persistent memory record from a NormalizedResult.

    `research_score` may be a ResearchScore returned by engine.research.
    """

    score = research_score

    return ResearchMemoryRecord(
        created_at=_now_iso(),
        alpha_id=getattr(result, "alpha_id", None),
        expression=str(getattr(result, "expression", "") or ""),
        compiler_expression=str(
            getattr(result, "compiler_expression", "") or ""
        ),
        alpha_type=str(
            getattr(result, "alpha_type", "NORMAL") or "NORMAL"
        ),
        simulation_status=str(
            getattr(result, "simulation_status", "") or ""
        ),
        eligibility_status=str(
            getattr(result, "eligibility_status", "") or ""
        ),
        fields=_as_list(getattr(result, "fields", [])),
        template=(
            str(getattr(result, "template"))
            if getattr(result, "template", None) is not None
            else None
        ),
        research_score=(
            _safe_float(getattr(score, "score", None))
            if score is not None
            else None
        ),
        research_class=(
            str(getattr(score, "research_class"))
            if score is not None
            and getattr(score, "research_class", None) is not None
            else None
        ),
        oos_score=(
            _safe_float(getattr(score, "oos_score", None))
            if score is not None
            else None
        ),
        consistency_score=(
            _safe_float(getattr(score, "consistency_score", None))
            if score is not None
            else None
        ),
        turnover_score=(
            _safe_float(getattr(score, "turnover_score", None))
            if score is not None
            else None
        ),
        robustness_score=(
            _safe_float(getattr(score, "robustness_score", None))
            if score is not None
            else None
        ),
        strengths=_as_list(
            getattr(score, "strengths", [])
            if score is not None
            else []
        ),
        weaknesses=_as_list(
            getattr(score, "weaknesses", [])
            if score is not None
            else []
        ),
        reasons=_as_list(
            getattr(score, "reasons", [])
            if score is not None
            else []
        ),
        failed_gates=_as_list(
            getattr(score, "failed_gate_names", [])
            if score is not None
            else []
        ),
        failed_brain_tests=_as_list(
            getattr(score, "failed_brain_test_names", [])
            if score is not None
            else getattr(result, "failed_brain_tests", [])
        ),
        warning_brain_tests=_as_list(
            getattr(score, "warning_brain_test_names", [])
            if score is not None
            else getattr(result, "warning_brain_tests", [])
        ),
        train_sharpe=_extract_metric(result, "train_sharpe"),
        test_sharpe=_extract_metric(result, "test_sharpe"),
        train_fitness=_extract_metric(result, "train_fitness"),
        test_fitness=_extract_metric(result, "test_fitness"),
        train_turnover=_extract_metric(result, "train_turnover"),
        test_turnover=_extract_metric(result, "test_turnover"),
        train_returns=_extract_metric(result, "train_returns"),
        test_returns=_extract_metric(result, "test_returns"),
        train_drawdown=_extract_metric(result, "train_drawdown"),
        test_drawdown=_extract_metric(result, "test_drawdown"),
        train_margin=_extract_metric(result, "train_margin"),
        test_margin=_extract_metric(result, "test_margin"),
        metadata=_json_safe(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Persistent store
# ---------------------------------------------------------------------------

class ResearchMemory:
    """
    Append-only JSONL research memory.

    The file is created lazily on first write.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_MEMORY_PATH,
    ) -> None:
        self.path = Path(path)

    def ensure_parent(self) -> None:
        """Create the parent directory when required."""

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        record: ResearchMemoryRecord,
    ) -> ResearchMemoryRecord:
        """Append one record and return it."""

        self.ensure_parent()

        payload = asdict(record)
        payload = _json_safe(payload)

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

        return record

    def append_result(
        self,
        result: Any,
        *,
        research_score: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchMemoryRecord:
        """Create and append a record from a NormalizedResult."""

        record = record_from_result(
            result,
            research_score=research_score,
            metadata=metadata,
        )

        return self.append(record)

    def load(self) -> list[ResearchMemoryRecord]:
        """Load all valid records from memory."""

        if not self.path.exists():
            return []

        records: list[ResearchMemoryRecord] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()

                if not text:
                    continue

                try:
                    payload = json.loads(text)
                    record = ResearchMemoryRecord(
                        **_filter_record_fields(payload)
                    )
                    records.append(record)
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        f"Invalid research memory record at "
                        f"{self.path}:{line_number}"
                    ) from exc

        return records

    def __len__(self) -> int:
        """Return the number of stored records."""

        return len(self.load())

    def clear(self) -> None:
        """Delete the memory file."""

        if self.path.exists():
            self.path.unlink()

    def latest(
        self,
        limit: int = 20,
    ) -> list[ResearchMemoryRecord]:
        """Return the newest records first."""

        if limit <= 0:
            return []

        records = self.load()

        return list(
            reversed(records[-limit:])
        )

    def top(
        self,
        limit: int = 20,
    ) -> list[ResearchMemoryRecord]:
        """Return the highest research-scored records first."""

        if limit <= 0:
            return []

        records = [
            record
            for record in self.load()
            if record.research_score is not None
        ]

        return sorted(
            records,
            key=lambda record: (
                record.research_score
                if record.research_score is not None
                else float("-inf")
            ),
            reverse=True,
        )[:limit]

    def by_class(
        self,
        research_class: str,
    ) -> list[ResearchMemoryRecord]:
        """Return records belonging to one research class."""

        wanted = str(
            research_class
        ).strip().upper()

        return [
            record
            for record in self.load()
            if str(
                record.research_class or ""
            ).strip().upper()
            == wanted
        ]

    def by_template(
        self,
        template: str,
    ) -> list[ResearchMemoryRecord]:
        """Return records for a compiler template."""

        wanted = str(
            template
        ).strip().upper()

        return [
            record
            for record in self.load()
            if str(
                record.template or ""
            ).strip().upper()
            == wanted
        ]


def _filter_record_fields(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Discard unknown fields when reading future-version records."""

    allowed = set(
        ResearchMemoryRecord.__dataclass_fields__.keys()
    )

    return {
        key: value
        for key, value in payload.items()
        if key in allowed
    }


# ---------------------------------------------------------------------------
# Research summaries for the future analyst
# ---------------------------------------------------------------------------

def summarize_memory(
    records: Sequence[ResearchMemoryRecord] | Iterable[ResearchMemoryRecord],
) -> dict[str, Any]:
    """
    Build a compact deterministic summary suitable for an LLM analyst.

    This summary contains aggregate patterns rather than raw BRAIN objects.
    """

    records = list(records)

    classes: dict[str, int] = {}
    templates: dict[str, int] = {}
    failed_tests: dict[str, int] = {}
    successful_patterns: list[dict[str, Any]] = []

    scored_records = []

    for record in records:
        if record.research_class:
            key = record.research_class.upper()
            classes[key] = classes.get(key, 0) + 1

        if record.template:
            key = record.template.upper()
            templates[key] = templates.get(key, 0) + 1

        for test_name in record.failed_brain_tests:
            key = test_name.upper()
            failed_tests[key] = failed_tests.get(key, 0) + 1

        if record.research_score is not None:
            scored_records.append(record)

    top_records = sorted(
        scored_records,
        key=lambda record: record.research_score or float("-inf"),
        reverse=True,
    )[:10]

    for record in top_records:
        successful_patterns.append(
            {
                "alpha_id": record.alpha_id,
                "template": record.template,
                "fields": record.fields,
                "research_class": record.research_class,
                "research_score": record.research_score,
                "test_sharpe": record.test_sharpe,
                "test_fitness": record.test_fitness,
                "test_turnover": record.test_turnover,
                "failed_gates": record.failed_gates,
                "failed_brain_tests": record.failed_brain_tests,
            }
        )

    average_test_sharpe = None

    if scored_records:
        sharpes = [
            record.test_sharpe
            for record in scored_records
            if record.test_sharpe is not None
        ]

        if sharpes:
            average_test_sharpe = sum(sharpes) / len(sharpes)

    return {
        "record_count": len(records),
        "class_counts": dict(
            sorted(
                classes.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "template_counts": dict(
            sorted(
                templates.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "common_failed_brain_tests": [
            {
                "name": name,
                "count": count,
            }
            for name, count in sorted(
                failed_tests.items(),
                key=lambda item: (-item[1], item[0]),
            )[:15]
        ],
        "average_test_sharpe": (
            round(average_test_sharpe, 4)
            if average_test_sharpe is not None
            else None
        ),
        "top_records": successful_patterns,
    }


def memory_for_analyst(
    memory: ResearchMemory,
    *,
    latest_limit: int = 25,
    top_limit: int = 10,
) -> dict[str, Any]:
    """
    Build a deterministic analyst context from persistent memory.

    The result is intentionally plain Python data so it can be serialized into
    the existing local Qwen analyst prompt later.
    """

    records = memory.load()

    return {
        "summary": summarize_memory(records),
        "latest_experiments": [
            asdict(record)
            for record in memory.latest(latest_limit)
        ],
        "top_experiments": [
            asdict(record)
            for record in memory.top(top_limit)
        ],
    }


def print_memory_summary(
    memory: ResearchMemory,
) -> None:
    """Print a compact memory status summary."""

    summary = summarize_memory(
        memory.load()
    )

    print("=" * 80)
    print("RESEARCH MEMORY")
    print("=" * 80)
    print(f"Path: {memory.path}")
    print(f"Records: {summary['record_count']}")
    print(f"Classes: {summary['class_counts']}")
    print(f"Templates: {summary['template_counts']}")
    print(
        "Average test Sharpe: "
        f"{summary['average_test_sharpe']}"
    )

    if summary["common_failed_brain_tests"]:
        print("\nCommon failed BRAIN tests:")

        for item in summary["common_failed_brain_tests"]:
            print(
                f"  {item['name']}: "
                f"{item['count']}"
            )
