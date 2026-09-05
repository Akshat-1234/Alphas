# APP_BUILD: BRAIN-HIGH-THROUGHPUT-2026-09-05-R9
import copy
import inspect
import json
import os
import re
import uuid
from pathlib import Path

from requests.cookies import create_cookie

import pandas as pd
import streamlit as st
import ace_lib as ace

from engine.compiler import FastExprCompiler
from engine.validator import FastExprValidator
from engine.research_candidate_generator import CandidateGenerator, ResearchSpec
from engine.research_memory import ResearchMemory
from engine.research import score_batch
from engine.results import normalize_result
from engine.simulator import SimulationRunner, create_job_from_payload
from engine.llm import LLMConfig, ResearchLLM
import engine.research_analyst as research_analyst


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Alpha Research Lab",
    page_icon="α",
    layout="wide",
)

st.title("Alpha Research Lab")
st.caption(
    "Region → Universe → Dataset → Analyst → BRAIN → Research Memory"
)

# Ollama is intentionally optional for the research loop. If it is down, the
# iteration path uses deterministic fallback research rather than crashing.


# ============================================================
# CONSTANTS
# ============================================================

MEMORY_DIR = Path(__file__).resolve().parent / "research_memory"

TEMPLATES = [
    "LEVEL",
    "HISTORICAL_STATE",
    "CHANGE",
    "SMOOTHED",
    "STABILITY",
    "DECAY",
    "RATIO",
    "RATIO_STATE",
    "RATIO_CHANGE",
    "INTERACTION",
    "CONTRAST",
    "CORRELATION",
]

ALLOWED_WINDOWS = [
    5, 10, 20, 30, 40, 60, 90, 120, 252
]

VERIFIED_OPERATORS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "reverse",
    "rank",
    "ts_backfill",
    "ts_mean",
    "ts_rank",
    "ts_zscore",
    "ts_delta",
    "ts_std_dev",
    "ts_decay_linear",
    "ts_corr",
    "vec_avg",
    "densify",
    "group_rank",
    "group_zscore",
}

DEFAULT_DELAY = 1
FIELD_PAGE_SIZE = 50
DATASET_PAGE_SIZE = 50
MAX_FIELD_PAGES = 500
MAX_DATASET_PAGES = 100

# BRAIN multi-simulation throughput: at most 10 alphas per multi-simulation
# and at most 8 multi-simulations concurrently. We deliberately keep the
# research objective standalone-metric only; no RL/bandit layer is used.
TARGET_SIMULATIONS_PER_ITERATION = 80
MULTI_SIMULATION_SIZE = 10
MAX_CONCURRENT_MULTI_SIMULATIONS = 8
CANDIDATE_POOL_SIZE = 140

# Persist the BRAIN browser/session cookies so a Streamlit process restart
# does not unnecessarily trigger a fresh authentication/persona challenge.
# This file contains authentication material and is intentionally stored in
# the user's home directory with restrictive permissions.
BRAIN_SESSION_CACHE_FILE = (
    Path.home() / ".brain_session_cache.json"
)


# ============================================================
# BRAIN SESSION
# ============================================================

def _serialize_brain_cookies(session):
    cookies = []

    for cookie in session.cookies:
        cookies.append(
            {
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
                "secure": bool(cookie.secure),
                "expires": cookie.expires,
                "rest": dict(cookie._rest),
            }
        )

    return cookies


def _save_brain_session_cache(session):
    payload = {
        "version": 1,
        "api_url": ace.brain_api_url,
        "cookies": _serialize_brain_cookies(session),
    }

    cache_file = BRAIN_SESSION_CACHE_FILE
    tmp_file = cache_file.with_suffix(".tmp")

    try:
        cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tmp_file.open(
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                payload,
                fh,
                separators=(",", ":"),
            )

        # Atomic replacement prevents a Streamlit restart from leaving a
        # half-written credential cache.
        os.replace(
            tmp_file,
            cache_file,
        )

        try:
            os.chmod(
                cache_file,
                0o600,
            )
        except OSError:
            pass

    except Exception:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass


def _load_brain_session_cache(session):
    cache_file = BRAIN_SESSION_CACHE_FILE

    if not cache_file.exists():
        return False

    try:
        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as fh:
            payload = json.load(fh)

        if payload.get("version") != 1:
            return False

        # Never reuse cookies created for a different BRAIN endpoint.
        if payload.get("api_url") != ace.brain_api_url:
            return False

        restored = 0

        for item in payload.get("cookies", []):
            name = item.get("name")
            value = item.get("value")

            if not name or value is None:
                continue

            session.cookies.set_cookie(
                create_cookie(
                    name=name,
                    value=str(value),
                    domain=item.get("domain", ""),
                    path=item.get("path", "/"),
                    secure=bool(item.get("secure", False)),
                    expires=item.get("expires"),
                    rest=item.get("rest") or {},
                )
            )
            restored += 1

        return restored > 0

    except Exception:
        return False


def _brain_session_status(session):
    """Return (status, seconds_remaining).

    Uses ace_lib's own session-timeout implementation rather than assuming a
    particular JSON shape for the /authentication response. A health-check
    failure is treated as unknown so a transient network problem cannot force
    an unnecessary persona/login challenge.

    status is one of:
      valid   -> ace_lib reports positive remaining session time.
      expired -> ace_lib reports zero/non-positive time or an explicit auth
                 rejection.
      unknown -> the health check itself could not be completed reliably.
    """
    try:
        remaining = ace.check_session_timeout(session)
        remaining = float(remaining)
    except Exception:
        # Do one direct request only to distinguish a genuine auth rejection
        # from a transient failure. Do not require a particular JSON schema.
        try:
            response = session.get(
                ace.brain_api_url + "/authentication",
                timeout=15,
            )
        except Exception:
            return "unknown", None

        if response.status_code in (401, 403):
            return "expired", 0

        return "unknown", None

    if remaining > 0:
        return "valid", remaining

    return "expired", 0


def _brain_session_is_valid(session):
    status, _ = _brain_session_status(session)
    return status == "valid"


def _start_fresh_brain_session():
    session = ace.start_session()

    if session is None:
        raise RuntimeError(
            "Could not create WorldQuant BRAIN session."
        )

    _save_brain_session_cache(
        session
    )
    return session


@st.cache_resource(
    show_spinner="Connecting to WorldQuant BRAIN..."
)
def get_brain_session():
    # SingleSession is intentionally used here because ace_lib itself
    # defines it as a process-wide singleton. Rehydrating cookies and reattaching the stored HTTP auth object
    # onto it preserves that behavior while allowing persistence across
    # process restarts without forcing a fresh persona login.
    session = ace.SingleSession()

    if _load_brain_session_cache(session):
        # BRAIN's session check may rely on the HTTP auth object in addition
        # to the persisted session cookies.  Reattach the already-saved
        # credentials from ace_lib's normal credential store; this does NOT
        # authenticate again or trigger the persona flow.
        try:
            session.auth = ace.get_credentials()
        except Exception:
            pass

        status, _ = _brain_session_status(session)

        if status in {"valid", "unknown"}:
            # Unknown means the health-check request itself failed. Reusing
            # the persisted authenticated object is safer than forcing a
            # fresh persona login on a transient network problem.
            _save_brain_session_cache(
                session
            )
            return session

    # The persisted cookies are missing or explicitly expired. Do not repeatedly try
    # them; clear them before the genuine authentication path.
    try:
        session.cookies.clear()
    except Exception:
        pass

    return _start_fresh_brain_session()


def ensure_brain_session(session):
    """Revalidate a cached session before using it for a new app rerun."""
    status, _ = _brain_session_status(session)

    if status in {"valid", "unknown"}:
        _save_brain_session_cache(
            session
        )
        return session

    try:
        session.cookies.clear()
    except Exception:
        pass

    return _start_fresh_brain_session()


# ============================================================
# PAGINATED BRAIN CATALOG LOADERS
# ============================================================

def _check_brain_response(
    response,
    label,
):
    if response.status_code // 100 != 2:
        raise RuntimeError(
            f"{label} request failed: "
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    if hasattr(ace, "_check_rate_limit"):
        ace._check_rate_limit(response)


@st.cache_data(
    show_spinner="Loading datasets..."
)
def load_datasets(
    _session,
    region,
    universe,
    delay=DEFAULT_DELAY,
):
    rows = []
    seen_ids = set()
    seen_pages = set()

    for page in range(MAX_DATASET_PAGES):
        offset = page * DATASET_PAGE_SIZE

        url = (
            ace.brain_api_url
            + "/data-sets?"
            + f"instrumentType=EQUITY"
            + f"&region={region}"
            + f"&delay={delay}"
            + f"&universe={universe}"
            + f"&limit={DATASET_PAGE_SIZE}"
            + f"&offset={offset}"
        )

        response = _session.get(url)
        _check_brain_response(
            response,
            "Dataset",
        )

        page_rows = response.json().get(
            "results",
            [],
        )

        page_signature = tuple(
            str(row.get("id", row.get("dataset_id", "")))
            for row in page_rows
        )

        if page_signature in seen_pages:
            break
        seen_pages.add(page_signature)

        for row in page_rows:
            row_id = str(
                row.get("id", row.get("dataset_id", ""))
            ).strip()
            if row_id and row_id not in seen_ids:
                rows.append(row)
                seen_ids.add(row_id)

        if len(page_rows) < DATASET_PAGE_SIZE:
            break

    if not rows:
        raise RuntimeError(
            f"No datasets returned for "
            f"{region} / {universe} / delay {delay}."
        )

    return pd.DataFrame(rows)


@st.cache_data(
    show_spinner="Loading selected dataset fields..."
)
def load_dataset_fields(
    _session,
    region,
    universe,
    dataset_id,
    delay=DEFAULT_DELAY,
):
    """Load fields only for the selected dataset.

    The BRAIN API supports dataset.id filtering on /data-fields. This avoids
    downloading the entire field catalog for a region/universe, which can be
    extremely large and was the reason the previous UI could sit on
    "Loading fields..." for several minutes.
    """
    if not dataset_id:
        return pd.DataFrame()

    first_url = ace.brain_api_url + "/data-fields"
    first_params = {
        "instrumentType": "EQUITY",
        "region": region,
        "delay": delay,
        "universe": universe,
        "dataset.id": dataset_id,
        "limit": FIELD_PAGE_SIZE,
        "offset": 0,
    }

    response = _session.get(
        first_url,
        params=first_params,
        timeout=30,
    )
    _check_brain_response(response, "Datafield")

    payload = response.json()
    rows = list(payload.get("results", []))
    total_count = int(payload.get("count", len(rows)) or len(rows))

    seen_ids = {
        str(row.get("id", "")).strip()
        for row in rows
        if str(row.get("id", "")).strip()
    }

    # Fetch only the remaining pages for this dataset.
    offset = FIELD_PAGE_SIZE
    pages = 1

    while offset < total_count and pages < MAX_FIELD_PAGES:
        params = {
            **first_params,
            "offset": offset,
        }

        response = _session.get(
            first_url,
            params=params,
            timeout=30,
        )
        _check_brain_response(response, "Datafield")

        page_rows = response.json().get("results", [])
        if not page_rows:
            break

        new_rows = 0
        for row in page_rows:
            row_id = str(row.get("id", "")).strip()
            if not row_id or row_id in seen_ids:
                continue
            rows.append(row)
            seen_ids.add(row_id)
            new_rows += 1

        pages += 1
        offset += FIELD_PAGE_SIZE

        if len(page_rows) < FIELD_PAGE_SIZE:
            break
        if new_rows == 0:
            break

    if not rows:
        raise RuntimeError(
            f"No fields returned for dataset '{dataset_id}' "
            f"in {region} / {universe} / delay {delay}."
        )

    return pd.DataFrame(rows)


# ============================================================
# CONFIG-SPECIFIC MEMORY
# ============================================================

def config_key(
    region,
    universe,
    dataset_id,
    delay,
):
    raw = (
        f"{region}__{universe}__"
        f"{dataset_id}__D{delay}"
    )
    return (
        raw
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def memory_path(
    region,
    universe,
    dataset_id,
    delay,
):
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = config_key(
        region,
        universe,
        dataset_id,
        delay,
    )

    return str(
        MEMORY_DIR / f"{key}.jsonl"
    )


def failure_path(
    region,
    universe,
    dataset_id,
    delay,
):
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = config_key(
        region,
        universe,
        dataset_id,
        delay,
    )

    return str(
        MEMORY_DIR / f"{key}.failures.jsonl"
    )


@st.cache_resource
def get_memory(
    region,
    universe,
    dataset_id,
    delay,
):
    return ResearchMemory(
        memory_path(
            region,
            universe,
            dataset_id,
            delay,
        )
    )


# ============================================================
# FAILURE LEDGER
# ============================================================

def load_failure_records(
    path,
):
    file_path = Path(path)

    if not file_path.exists():
        return []

    records = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as fh:

        for line in fh:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue

    return records


def append_failure_record(
    path,
    record,
):
    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "a",
        encoding="utf-8",
    ) as fh:

        fh.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


# ============================================================
# CATALOG
# ============================================================

def select_seed_fields(
    fields_df,
    limit=40,
):
    matrix = fields_df[
        fields_df["type"]
        .astype(str)
        .str.upper()
        == "MATRIX"
    ].copy()

    if matrix.empty:
        return []

    candidates = matrix.copy()

    coverage = pd.to_numeric(
        candidates.get(
            "coverage",
            pd.Series(
                0,
                index=candidates.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    alpha_count = pd.to_numeric(
        candidates.get(
            "alphaCount",
            pd.Series(
                0,
                index=candidates.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    candidates["_coverage"] = coverage
    candidates["_alpha_count"] = alpha_count

    group_col = None

    for candidate_col in [
        "subcategory_name",
        "category_name",
        "subcategory",
        "category",
    ]:
        if candidate_col in candidates.columns:
            group_col = candidate_col
            break

    selected = []
    used = set()

    if group_col is not None:

        groups = []

        for group_name, group_df in candidates.groupby(
            candidates[group_col].astype(str),
            dropna=False,
        ):
            groups.append(
                (
                    group_name,
                    group_df.sort_values(
                        [
                            "_coverage",
                            "_alpha_count",
                        ],
                        ascending=[
                            False,
                            True,
                        ],
                    ),
                )
            )

        groups.sort(
            key=lambda x: (
                -len(x[1]),
                str(x[0]),
            )
        )

        for _, group_df in groups:

            if len(selected) >= limit:
                break

            for _, row in group_df.iterrows():

                field_id = str(
                    row["id"]
                )

                if field_id in used:
                    continue

                selected.append(
                    field_id
                )

                used.add(field_id)
                break

    remaining = candidates.sort_values(
        [
            "_coverage",
            "_alpha_count",
        ],
        ascending=[
            False,
            True,
        ],
    )

    for _, row in remaining.iterrows():

        if len(selected) >= limit:
            break

        field_id = str(
            row["id"]
        )

        if field_id in used:
            continue

        selected.append(
            field_id
        )

        used.add(field_id)

    return selected[:limit]


def build_catalog(
    fields_df,
):
    required_columns = {"id", "type"}
    missing = required_columns - set(fields_df.columns)
    if missing:
        raise RuntimeError(
            "BRAIN field response is missing required columns: "
            + ", ".join(sorted(missing))
        )

    matrix_fields = fields_df[
        fields_df["type"]
        .astype(str)
        .str.upper()
        == "MATRIX"
    ].copy()

    matrix_fields = matrix_fields.reset_index(
        drop=True
    )

    if matrix_fields.empty:
        raise RuntimeError(
            "The selected dataset contains no MATRIX "
            "fields. The current research generator "
            "requires MATRIX fields."
        )

    field_ids = [
        str(x)
        for x in matrix_fields[
            "id"
        ].tolist()
    ]

    field_types = [
        str(x).upper()
        for x in matrix_fields[
            "type"
        ].tolist()
    ]

    alias_to_id = {
        f"F{i + 1}": field_id
        for i, field_id
        in enumerate(field_ids)
    }

    alias_to_type = {
        f"F{i + 1}": field_type
        for i, field_type
        in enumerate(field_types)
    }

    id_to_alias = {
        field_id: alias
        for alias, field_id
        in alias_to_id.items()
    }

    return {
        "matrix_fields_df":
            matrix_fields,
        "field_alias_to_id":
            alias_to_id,
        "field_alias_to_type":
            alias_to_type,
        "id_to_field_alias":
            id_to_alias,
        "seed_fields":
            select_seed_fields(
                matrix_fields
            ),
    }


# ============================================================
# COMPILER / VALIDATOR
# ============================================================

def build_engine(
    catalog,
):
    compiler = FastExprCompiler(
        field_alias_to_id=catalog[
            "field_alias_to_id"
        ],
        field_alias_to_type=catalog[
            "field_alias_to_type"
        ],
        verified_operators=VERIFIED_OPERATORS,
        allowed_windows=ALLOWED_WINDOWS,
    )

    validator = FastExprValidator(
        field_alias_to_type=catalog[
            "field_alias_to_type"
        ],
        verified_operators=VERIFIED_OPERATORS,
        allowed_windows=ALLOWED_WINDOWS,
        max_length=700,
        max_depth=9,
    )

    return compiler, validator


# ============================================================
# ANALYST
# ============================================================

class AnalystLLMAdapter:

    def __init__(
        self,
        research_llm,
    ):
        self.research_llm = research_llm

    def generate_json(
        self,
        prompt,
    ):
        return self.research_llm.analyze_json(
            prompt
        )


class DashboardResearchAnalyst(
    research_analyst.ResearchAnalyst
):

    def __init__(
        self,
        *args,
        seed_fields=(),
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.seed_fields = tuple(
            seed_fields
        )

    def build_prompt(
        self,
        context,
    ):
        evidence = context.get(
            "evidence",
            {},
        )

        tiers = evidence.get(
            "tiers",
            {},
        )

        experiments = list(
            evidence.get(
                "experiments",
                [],
            )
        )

        experiments.sort(
            key=lambda x: (
                x.get(
                    "research_score"
                )
                if x.get(
                    "research_score"
                ) is not None
                else float("-inf")
            ),
            reverse=True,
        )

        compact_experiments = []

        for item in experiments[:8]:
            compact_experiments.append({
                "alpha":
                    item.get(
                        "alpha_id"
                    ),
                "template":
                    item.get(
                        "template"
                    ),
                "fields":
                    item.get(
                        "fields",
                        [],
                    ),
                "class":
                    item.get(
                        "research_class"
                    ),
                "score":
                    item.get(
                        "research_score"
                    ),
                "test_sharpe":
                    item.get(
                        "test_sharpe"
                    ),
                "test_fitness":
                    item.get(
                        "test_fitness"
                    ),
                "test_turnover":
                    item.get(
                        "test_turnover"
                    ),
                "robustness":
                    item.get(
                        "robustness_score"
                    ),
                "failed":
                    item.get(
                        "failed_brain_tests",
                        [],
                    ),
            })

        signal_templates = tiers.get(
            "signal_templates",
            [],
        )

        signal_fields = tiers.get(
            "signal_fields",
            [],
        )

        failed_templates = tiers.get(
            "failure_templates",
            [],
        )

        failed_fields = tiers.get(
            "failure_fields",
            [],
        )

        initial_mode = (
            len(experiments) == 0
        )

        if initial_mode:
            available_fields = list(
                self.seed_fields
            )
        else:
            available_fields = list(
                signal_fields
            )

        compact_context = {
            "initial_mode":
                initial_mode,
            "available_fields":
                available_fields,
            "signal_templates":
                signal_templates,
            "failed_templates":
                failed_templates,
            "failed_fields":
                failed_fields,
            "experiments":
                compact_experiments,
        }

        serialized = json.dumps(
            compact_context,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        initial_instructions = """
INITIAL DATASET EXPLORATION MODE:
There are no historical experiments yet.

Choose 1–2 small baseline experiments using ONLY
available_fields.

Prefer simple templates:
LEVEL, CHANGE, SMOOTHED, STABILITY, INTERACTION.

Do not claim that any field or template is validated.
""" if initial_mode else """
RESEARCH MODE:
OOS_SIGNAL templates may only be selected as
explicit repair experiments.
Prefer repairing existing OOS_SIGNAL evidence.
"""

        return f"""
You are an alpha research experiment planner.

Your ONLY task is to select the next experiments.

DO NOT:
- generate FASTEXPR
- generate BRAIN expressions
- invent fields
- claim validation
- revive failed templates or fields

HARD RULES:

AVAILABLE FIELDS:
{available_fields}

FAILED TEMPLATES:
{failed_templates}

FAILED FIELDS:
{failed_fields}

OOS SIGNAL TEMPLATES:
{signal_templates}

{initial_instructions}

Repair targets:

DECAY -> turnover_and_robustness
INTERACTION -> robustness_and_train_oos_consistency
CONTRAST -> robustness_and_train_oos_consistency
CHANGE -> direction_and_timescale_validation
SMOOTHED -> predictive_strength_and_turnover
STABILITY -> predictive_strength_and_turnover
RATIO -> direction_and_generalization_validation
RATIO_STATE -> direction_and_generalization_validation
RATIO_CHANGE -> direction_and_generalization_validation
CORRELATION -> deprioritized_template_validation
LEVEL -> baseline_validation
HISTORICAL_STATE -> timescale_and_robustness_validation

Return exactly:

{{
  "summary": "...",
  "promising_patterns": [],
  "failure_patterns": [],
  "fixable_patterns": [],
  "recommended_templates": [],
  "recommended_fields": [],
  "recommended_directions": [],
  "avoid_templates": [],
  "avoid_fields": [],
  "next_experiments": [
    {{
      "template": "...",
      "fields": ["..."],
      "direction": "positive",
      "reason": "...",
      "repair_target": "..."
    }}
  ],
  "confidence": "LOW | MEDIUM | HIGH"
}}

CRITICAL:
- at most 2 next experiments
- every field must be in AVAILABLE FIELDS
- never use FAILED FIELDS
- never use FAILED TEMPLATES
- every non-initial experiment must repair an observed weakness
- do not output FASTEXPR

EMPIRICAL EVIDENCE:
{serialized}
""".strip()


@st.cache_resource
def create_analyst(
    dataset_key,
    live_fields,
    seed_fields,
):
    config = LLMConfig(
        model="qwen3:8b",
        analyst_model="qwen3:8b",
        timeout=300,
        max_retries=0,
    )

    llm = ResearchLLM(
        config
    )

    return DashboardResearchAnalyst(
        AnalystLLMAdapter(llm),
        max_memory_records=25,
        live_templates=TEMPLATES,
        live_fields=tuple(
            live_fields
        ),
        seed_fields=tuple(
            seed_fields
        ),
    )


# ============================================================
# FAILURE BLOCKING
# ============================================================

def build_blocked_specs(
    records,
    failures,
):
    blocked = []

    # Successful historical experiments.
    for r in records:

        metadata = getattr(
            r,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        window = metadata.get(
            "window"
        )

        backfill_window = metadata.get(
            "backfill_window"
        )

        direction = metadata.get(
            "direction"
        )

        if (
            window is None
            or backfill_window is None
            or direction is None
        ):
            continue

        blocked.append({
            "template":
                r.template,
            "fields":
                list(r.fields),
            "window":
                int(window),
            "backfill_window":
                int(backfill_window),
            "direction":
                str(direction).lower(),
        })

    # Historical BRAIN failures.
    for failure in failures:

        window = failure.get(
            "window"
        )

        backfill_window = failure.get(
            "backfill_window"
        )

        direction = failure.get(
            "direction"
        )

        template = failure.get(
            "template"
        )

        fields = failure.get(
            "fields"
        )

        if (
            window is None
            or backfill_window is None
            or direction is None
            or not template
            or not fields
        ):
            continue

        blocked.append({
            "template":
                template,
            "fields":
                list(fields),
            "window":
                int(window),
            "backfill_window":
                int(backfill_window),
            "direction":
                str(direction).lower(),
        })

    return blocked


# ============================================================
# GENERATOR
# ============================================================

def build_generator(
    catalog,
    records,
    failures,
):
    kwargs = {
        "template_field_counts": {
            "LEVEL": 1,
            "HISTORICAL_STATE": 1,
            "CHANGE": 1,
            "SMOOTHED": 1,
            "STABILITY": 1,
            "DECAY": 1,
            "RATIO": 2,
            "RATIO_STATE": 2,
            "RATIO_CHANGE": 2,
            "INTERACTION": 2,
            "CONTRAST": 2,
            "CORRELATION": 2,
        },
        "live_fields": catalog[
            "field_alias_to_id"
        ].values(),
        "allowed_windows": ALLOWED_WINDOWS,
        "max_candidates": 80,
        "max_per_template": 10,
    }

    blocked = build_blocked_specs(
        records,
        failures,
    )

    supported = inspect.signature(
        CandidateGenerator.__init__
    ).parameters

    if "blocked_specs" in supported:
        kwargs["blocked_specs"] = blocked

    return CandidateGenerator(**kwargs)


# ============================================================
# BRAIN EXPRESSION TRANSLATION
# ============================================================

def aliases_to_brain_expression(
    expression,
    alias_to_id,
):
    def replace(
        match,
    ):
        alias = match.group(0)

        if alias not in alias_to_id:
            raise RuntimeError(
                f"Unknown field alias: {alias}"
            )

        return alias_to_id[
            alias
        ]

    return re.sub(
        r"\bF\d+\b",
        replace,
        expression,
    )


def build_brain_payload(
    region,
    universe,
    delay,
):
    return {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "delay": delay,
            "decay": 2,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "testPeriod": "P1M",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "maxTrade": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
        },
        "regular": "",
    }


# ============================================================
# COLD-START / RECOVERY CANDIDATES
# ============================================================

def _make_research_spec(**kwargs):
    supported = inspect.signature(ResearchSpec).parameters
    return ResearchSpec(
        **{
            key: value
            for key, value in kwargs.items()
            if key in supported
        }
    )


def _spec_key(spec):
    return (
        str(spec.template).upper(),
        tuple(str(x) for x in spec.fields),
        int(spec.window),
        int(spec.backfill_window),
        str(spec.direction).lower(),
    )


def _blocked_spec_keys(records, failures):
    blocked = set()

    def add(value):
        template = fields = window = backfill = direction = None
        if isinstance(value, dict):
            template = value.get("template")
            fields = value.get("fields")
            window = value.get("window")
            backfill = value.get("backfill_window")
            direction = value.get("direction")
        else:
            template = getattr(value, "template", None)
            fields = getattr(value, "fields", None)
            window = getattr(value, "window", None)
            backfill = getattr(value, "backfill_window", None)
            direction = getattr(value, "direction", None)
            metadata = getattr(value, "metadata", {})
            if isinstance(metadata, dict):
                template = template or metadata.get("template")
                fields = fields or metadata.get("fields")
                window = window if window is not None else metadata.get("window")
                backfill = (
                    backfill
                    if backfill is not None
                    else metadata.get("backfill_window")
                )
                direction = direction or metadata.get("direction")

        if not template or not fields:
            return
        try:
            key = (
                str(template).upper(),
                tuple(str(x) for x in fields),
                int(window if window is not None else 60),
                int(backfill if backfill is not None else 60),
                str(direction or "positive").lower(),
            )
        except (TypeError, ValueError):
            return
        blocked.add(key)

    for item in records or []:
        add(item)
    for item in failures or []:
        add(item)

    return blocked


def _research_template_priority(records, failures):
    """Deterministically rank templates from observed standalone results."""
    stats = {
        template: {"scores": [], "good": 0, "bad": 0}
        for template in TEMPLATES
    }

    for record in records or []:
        template = getattr(record, "template", None)
        metadata = getattr(record, "metadata", {})
        if not template and isinstance(metadata, dict):
            template = metadata.get("template")
        if not template:
            continue
        template = str(template).upper()
        stats.setdefault(template, {"scores": [], "good": 0, "bad": 0})

        score_obj = getattr(record, "research_score", None)
        score = getattr(score_obj, "score", None)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        if score is not None:
            stats[template]["scores"].append(score)
            if score >= 50:
                stats[template]["good"] += 1
            if score < 20:
                stats[template]["bad"] += 1

    for failure in failures or []:
        template = failure.get("template") if isinstance(failure, dict) else getattr(failure, "template", None)
        if not template and isinstance(failure, dict):
            template = failure.get("metadata", {}).get("template")
        if template:
            template = str(template).upper()
            stats.setdefault(template, {"scores": [], "good": 0, "bad": 0})
            stats[template]["bad"] += 1

    def rank(item):
        template, value = item
        scores = value["scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        good_rate = value["good"] / len(scores) if scores else 0.0
        bad_rate = value["bad"] / max(1, len(scores) + value["bad"])
        return (avg + 15.0 * good_rate - 12.0 * bad_rate, template)

    return [template for template, _ in sorted(stats.items(), key=rank, reverse=True)]


def _make_spec_from_parts(
    *,
    template,
    fields,
    window,
    backfill_window,
    direction,
    family,
    intuition,
    repair_target,
    source,
    source_rank,
    repair_reason,
):
    return _make_research_spec(
        template=template,
        fields=tuple(fields),
        window=int(window),
        backfill_window=int(backfill_window),
        direction=str(direction).lower(),
        family=family,
        intuition=intuition,
        repair_target=repair_target,
        source=source,
        source_rank=int(source_rank),
        repair_reason=repair_reason,
    )


def expand_specs_to_target(
    catalog,
    records,
    failures,
    base_specs,
    target=80,
):
    """Fill a research batch deterministically up to the target size.

    Priority order:
      1) analyst ideas;
      2) local parameter variants around observed successful experiments;
      3) field substitutions around those experiments;
      4) deterministic seed-field exploration across templates.

    This is not reinforcement learning. No policy is learned; all ordering is
    derived from the persisted standalone research evidence and fixed rules.
    """
    target = max(1, int(target))
    blocked = _blocked_spec_keys(records, failures)
    seen = set(blocked)
    final = []

    def add(spec):
        if spec is None:
            return False
        key = _spec_key(spec)
        if key in seen:
            return False
        seen.add(key)
        final.append(spec)
        return True

    # 1. Analyst output is always first.
    for spec in base_specs or []:
        if len(final) >= target:
            break
        add(spec)

    # Seed fields are already curated from the live dataset.
    seeds = [str(x) for x in catalog.get("seed_fields", []) if str(x)]
    if not seeds:
        seeds = [str(x) for x in catalog.get("field_alias_to_id", {}).values() if str(x)]
    seeds = list(dict.fromkeys(seeds))

    # 2. Strong historical experiments get local neighborhoods first.
    ranked_records = []
    for record in records or []:
        score_obj = getattr(record, "research_score", None)
        score = getattr(score_obj, "score", None)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = float("-inf")
        template = getattr(record, "template", None)
        fields = tuple(getattr(record, "fields", ()) or ())
        metadata = getattr(record, "metadata", {})
        if isinstance(metadata, dict):
            template = template or metadata.get("template")
            if not fields:
                fields = tuple(metadata.get("fields", ()) or ())
        if template and fields:
            ranked_records.append((score, record, str(template).upper(), fields, metadata if isinstance(metadata, dict) else {}))
    ranked_records.sort(key=lambda x: x[0], reverse=True)

    for score, record, template, fields, metadata in ranked_records[:20]:
        if len(final) >= target:
            break
        base_window = metadata.get("window", 60)
        base_backfill = metadata.get("backfill_window", 60)
        direction = str(metadata.get("direction", "positive")).lower()
        try:
            base_window = int(base_window)
            base_backfill = int(base_backfill)
        except (TypeError, ValueError):
            base_window, base_backfill = 60, 60

        windows = sorted(ALLOWED_WINDOWS, key=lambda x: (abs(x - base_window), x))
        windows = list(dict.fromkeys([base_window] + windows))
        # Favor close timescales and only expand further when capacity remains.
        for window in windows[:6]:
            if len(final) >= target:
                break
            for backfill in (base_backfill, window, 60):
                for dir_value in (direction, "negative" if direction == "positive" else "positive"):
                    spec = _make_spec_from_parts(
                        template=template,
                        fields=fields,
                        window=window,
                        backfill_window=backfill,
                        direction=dir_value,
                        family="historical_neighborhood",
                        intuition="Controlled neighborhood around an observed standalone experiment.",
                        repair_target="parameter_robustness",
                        source="historical_neighborhood",
                        source_rank=0,
                        repair_reason=f"Historical score={score:.2f}; deterministic local search.",
                    )
                    add(spec)
                    if len(final) >= target:
                        break
                if len(final) >= target:
                    break
            if len(final) >= target:
                break

        # Controlled field substitutions around promising structures.
        if len(final) < target and seeds:
            required = len(fields)
            if required == 1:
                for field in seeds:
                    if field == fields[0]:
                        continue
                    spec = _make_spec_from_parts(
                        template=template,
                        fields=(field,),
                        window=base_window,
                        backfill_window=base_backfill,
                        direction=direction,
                        family="historical_field_substitution",
                        intuition="Substitute a curated live field into a promising structure.",
                        repair_target="field_generalization",
                        source="historical_field_substitution",
                        source_rank=1,
                        repair_reason="Test whether the successful structure generalizes to another dataset field.",
                    )
                    add(spec)
                    if len(final) >= target:
                        break
            elif required == 2:
                for field in seeds:
                    if field in fields:
                        continue
                    for pos in (0, 1):
                        pair = list(fields)
                        pair[pos] = field
                        if pair[0] == pair[1]:
                            continue
                        spec = _make_spec_from_parts(
                            template=template,
                            fields=pair,
                            window=base_window,
                            backfill_window=base_backfill,
                            direction=direction,
                            family="historical_field_substitution",
                            intuition="Replace one leg of a promising two-field structure with a curated live field.",
                            repair_target="field_generalization",
                            source="historical_field_substitution",
                            source_rank=1,
                            repair_reason="Test field-level generalization without changing the successful structure.",
                        )
                        add(spec)
                        if len(final) >= target:
                            break
                    if len(final) >= target:
                        break

    # 3. Deterministic template exploration fills any remaining capacity.
    template_order = _research_template_priority(records, failures)
    # Put templates from the current compiler first when there is no evidence.
    if not records:
        default_order = [
            "LEVEL", "SMOOTHED", "CHANGE", "STABILITY", "DECAY",
            "HISTORICAL_STATE", "INTERACTION", "CONTRAST",
            "RATIO_STATE", "RATIO_CHANGE", "RATIO", "CORRELATION",
        ]
        template_order = default_order
    for template in template_order:
        if template not in TEMPLATES:
            continue
        if len(final) >= target:
            break
        required = 1 if template in {"LEVEL", "HISTORICAL_STATE", "CHANGE", "SMOOTHED", "STABILITY", "DECAY"} else 2
        if len(seeds) < required:
            continue

        local_field_pairs = []
        if required == 1:
            local_fields = [(field,) for field in seeds[: min(len(seeds), 40)]]
        else:
            # Limit pair construction to a structured subset so the search does
            # not waste the simulation budget on arbitrary combinatorics.
            n = min(len(seeds), 16)
            local_fields = []
            for i in range(n):
                for j in range(i + 1, n):
                    local_fields.append((seeds[i], seeds[j]))
                    if len(local_fields) >= 60:
                        break
                if len(local_fields) >= 60:
                    break

        # Keep the throughput pool diversified: no single template can consume
        # the entire 140-candidate pool before other templates get explored.
        template_pool_cap = max(6, min(16, (target + len(template_order) - 1) // max(1, len(template_order)) + 4))
        template_pool_count = 0

        for fields in local_fields:
            if len(final) >= target or template_pool_count >= template_pool_cap:
                break
            # Two timescales and both directions provide controlled coverage.
            for window, backfill, direction in (
                (60, 60, "positive"),
                (20, 20, "positive"),
                (40, 60, "positive"),
                (60, 60, "negative"),
            ):
                if window not in ALLOWED_WINDOWS or backfill not in ALLOWED_WINDOWS:
                    continue
                spec = _make_spec_from_parts(
                    template=template,
                    fields=fields,
                    window=window,
                    backfill_window=backfill,
                    direction=direction,
                    family="deterministic_exploration",
                    intuition="Curated standalone alpha exploration for simulation throughput.",
                    repair_target="standalone_metric_search",
                    source="deterministic_exploration",
                    source_rank=2,
                    repair_reason="Fill the simulation budget with valid, typed, deterministic hypotheses.",
                )
                if add(spec):
                    template_pool_count += 1
                if len(final) >= target or template_pool_count >= template_pool_cap:
                    break

    return final[:target]


def build_cold_start_specs(seed_fields, limit=80):
    # Cold start uses the same deterministic high-throughput expansion rules,
    # but creates a compact set of safe baseline specs without LLM evidence.
    base_specs = []
    for field in list(seed_fields)[:8]:
        base_specs.append(_make_research_spec(
            template="LEVEL",
            fields=(str(field),),
            window=60,
            backfill_window=60,
            direction="positive",
            family="cold_start",
            intuition="Baseline level signal validation.",
            repair_target="baseline_validation",
            source="cold_start",
            source_rank=1,
            repair_reason="Initial baseline; no historical evidence exists yet.",
        ))
        base_specs.append(_make_research_spec(
            template="CHANGE",
            fields=(str(field),),
            window=20,
            backfill_window=20,
            direction="positive",
            family="cold_start",
            intuition="Baseline change signal and short-timescale validation.",
            repair_target="direction_and_timescale_validation",
            source="cold_start",
            source_rank=1,
            repair_reason="Initial baseline; test short-horizon change behavior.",
        ))

    # This function is only called without the full catalog, so preserve the
    # original simple behavior here. The caller can pass the resulting specs to
    # expand_specs_to_target for the full 80-job batch.
    return base_specs[:limit]


def build_ollama_fallback_specs(catalog, records, failures, limit=80):
    """Build bounded deterministic mutations when the local LLM is unavailable.

    This is deliberately not reinforcement learning: it is a deterministic
    recovery path so an Ollama outage cannot crash an otherwise healthy BRAIN
    research loop.
    """
    specs = []
    seen = set()
    supported = inspect.signature(ResearchSpec).parameters

    def add_spec(**kwargs):
        spec = ResearchSpec(
            **{
                key: value
                for key, value in kwargs.items()
                if key in supported
            }
        )
        key = (
            spec.template,
            tuple(spec.fields),
            int(spec.window),
            int(spec.backfill_window),
            str(spec.direction).lower(),
        )
        if key in seen:
            return
        seen.add(key)
        specs.append(spec)

    # Prefer successful historical research because it provides a grounded
    # deterministic neighborhood to explore without asking the LLM for help.
    ranked = []
    for record in records:
        score_obj = getattr(record, "research_score", None)
        score_value = getattr(score_obj, "score", None)
        try:
            score_value = float(score_value)
        except (TypeError, ValueError):
            score_value = float("-inf")
        metadata = getattr(record, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        ranked.append((score_value, record, metadata))

    ranked.sort(key=lambda x: x[0], reverse=True)

    window_variants = (20, 30, 40, 60, 90)
    for _, record, metadata in ranked[:6]:
        template = getattr(record, "template", None) or metadata.get("template")
        fields = tuple(getattr(record, "fields", ()) or metadata.get("fields", ()))
        if not template or not fields:
            continue

        base_window = metadata.get("window", 60)
        base_backfill = metadata.get("backfill_window", 60)
        base_direction = str(metadata.get("direction", "positive")).lower()

        try:
            base_window = int(base_window)
            base_backfill = int(base_backfill)
        except (TypeError, ValueError):
            base_window, base_backfill = 60, 60

        for window in window_variants:
            if len(specs) >= limit:
                break
            if window == base_window:
                continue
            add_spec(
                template=str(template),
                fields=fields,
                window=window,
                backfill_window=base_backfill,
                direction=base_direction,
                family="ollama_fallback",
                intuition="Deterministic neighborhood search around a historical experiment.",
                repair_target="parameter_neighborhood",
                source="ollama_fallback",
                source_rank=1,
                repair_reason="Ollama unavailable; explore nearby timing parameters without LLM reasoning.",
            )

    # Always have a deterministic escape hatch even when historical metadata
    # is incomplete. This uses the normal seed-field baseline, then the normal
    # compiler/validator and failure blocking prevent unsafe submissions.
    if len(specs) < limit:
        for spec in build_cold_start_specs(
            catalog.get("seed_fields", []),
            limit=limit - len(specs),
        ):
            add_spec(
                template=spec.template,
                fields=tuple(spec.fields),
                window=spec.window,
                backfill_window=spec.backfill_window,
                direction=spec.direction,
                family="ollama_fallback",
                intuition=spec.intuition,
                repair_target=spec.repair_target,
                source="ollama_fallback",
                source_rank=2,
                repair_reason="Ollama unavailable; deterministic fallback experiment.",
            )

    return specs[:limit]


# ============================================================
# COMPLETE ITERATION
# ============================================================

def run_iteration(
    session,
    region,
    universe,
    dataset_id,
    delay,
    catalog,
    compiler,
    validator,
    analyst,
    memory,
    failure_file,
):
    log = []

    records = memory.load()
    failures = load_failure_records(
        failure_file
    )

    log.append(
        f"Configuration: {region} / "
        f"{universe} / {dataset_id} / "
        f"delay={delay}"
    )

    log.append(
        f"Research memory: {len(records)}"
    )

    log.append(
        f"Failure ledger: {len(failures)}"
    )

    # --------------------------------------------------------
    # Analyst / cold start
    # --------------------------------------------------------

    if not records:
        insight = None
        specs = build_cold_start_specs(
            catalog["seed_fields"],
            limit=8,
        )
        log.append(
            "Cold start: no research memory; generated deterministic baseline specs."
        )
    else:
        try:
            insight = analyst.analyze(
                memory
            )

            log.append(
                f"Analyst confidence: "
                f"{insight.confidence}"
            )

            # ----------------------------------------------------
            # Generate
            # ----------------------------------------------------

            generator = build_generator(
                catalog,
                records,
                failures,
            )

            specs = generator.generate(
                insight
            )
            specs = expand_specs_to_target(
                catalog,
                records,
                failures,
                specs,
                target=CANDIDATE_POOL_SIZE,
            )

            log.append(
                f"Generated research specs: "
                f"{len(specs)}"
            )

        except Exception as exc:
            # Do not let a local Ollama outage kill the BRAIN research loop.
            # In particular, WinError 10061 means the local Ollama server is
            # not listening. Fall back to deterministic candidate mutations.
            insight = None
            specs = build_ollama_fallback_specs(
                catalog,
                records,
                failures,
                limit=80,
            )
            specs = expand_specs_to_target(
                catalog, records, failures, specs, target=CANDIDATE_POOL_SIZE
            )
            log.append(
                "Analyst unavailable; deterministic fallback enabled."
            )
            log.append(
                f"Analyst error: {type(exc).__name__}: {exc}"
            )
            log.append(
                f"Fallback research specs: {len(specs)}"
            )

    if records and not specs:
        fallback_seed = catalog.get("seed_fields", [])
        specs = build_cold_start_specs(
            fallback_seed,
            limit=80,
        )
        specs = expand_specs_to_target(
            catalog, records, failures, specs, target=CANDIDATE_POOL_SIZE
        )
        log.append(
            "Analyst produced no usable candidates; used deterministic recovery specs."
        )

    specs = expand_specs_to_target(
        catalog,
        records,
        failures,
        specs,
        target=CANDIDATE_POOL_SIZE,
    )

    existing_expressions = {
        r.compiler_expression
        for r in records
        if getattr(
            r,
            "compiler_expression",
            None,
        )
    }

    prepared = []

    for spec in specs:

        aliases = []

        for field_id in spec.fields:

            alias = catalog[
                "id_to_field_alias"
            ].get(
                field_id
            )

            if alias is None:
                log.append(
                    f"SKIP unknown field: "
                    f"{field_id}"
                )
                aliases = []
                break

            aliases.append(
                alias
            )

        if not aliases:
            continue

        # ----------------------------------------------------
        # Compile / validate / package one candidate
        # ----------------------------------------------------

        # One malformed candidate must never abort the whole 80-alpha batch.
        try:
            compile_result = compiler.compile(
                spec.template,
                aliases,
                window=spec.window,
                backfill_window=spec.backfill_window,
                direction=spec.direction,
            )

            compiler_expression = compile_result.expression

            valid, message = validator.validate(
                compiler_expression
            )

            if not valid:
                log.append(
                    f"INVALID: {compiler_expression} :: {message}"
                )
                continue

            if compiler_expression in existing_expressions:
                log.append(
                    f"DUPLICATE: {compiler_expression}"
                )
                continue

            brain_expression = aliases_to_brain_expression(
                compiler_expression,
                catalog["field_alias_to_id"],
            )

            payload = build_brain_payload(
                region,
                universe,
                delay,
            )
            payload["regular"] = brain_expression

            job = create_job_from_payload(
                compiler_expression=compiler_expression,
                payload=payload,
                alpha_type="NORMAL",
                template=spec.template,
                fields=list(spec.fields),
                parameters={
                    "window": spec.window,
                    "backfill_window": spec.backfill_window,
                    "direction": spec.direction,
                    "repair_target": spec.repair_target,
                    "reason": getattr(spec, "repair_reason", ""),
                    "intuition": spec.intuition,
                    "region": region,
                    "universe": universe,
                    "delay": delay,
                    "dataset": dataset_id,
                },
                job_id=uuid.uuid4().hex,
            )
        except Exception as exc:
            log.append(
                f"SKIP candidate after compile/package error: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        prepared.append((spec, job))

        if len(prepared) >= TARGET_SIMULATIONS_PER_ITERATION:
            break

    log.append(
        f"Candidate pool: {len(specs)}; valid simulation jobs: {len(prepared)}"
    )

    if not prepared:

        return {
            "insight":
                insight,
            "specs":
                specs,
            "jobs":
                [],
            "results":
                [],
            "scores":
                [],
            "failed_jobs":
                [],
            "log":
                log,
        }

    # --------------------------------------------------------
    # BRAIN
    # --------------------------------------------------------

    runner = SimulationRunner(
        session
    )

    jobs = [
        job
        for _, job
        in prepared
    ]

    # Hard cap: never submit more than the 8x10 throughput budget.
    jobs = jobs[:TARGET_SIMULATIONS_PER_ITERATION]
    expected_multi_batches = (len(jobs) + MULTI_SIMULATION_SIZE - 1) // MULTI_SIMULATION_SIZE
    concurrent_batches = min(
        MAX_CONCURRENT_MULTI_SIMULATIONS,
        max(1, expected_multi_batches),
    )
    log.append(
        f"Submitting {len(jobs)} alpha jobs as "
        f"{expected_multi_batches} multi-simulations "
        f"({MULTI_SIMULATION_SIZE} alphas each; "
        f"up to {MAX_CONCURRENT_MULTI_SIMULATIONS} concurrent batches)..."
    )

    simulated = runner.simulate(
        jobs,
        limit_of_multi_simulations=MULTI_SIMULATION_SIZE,
        num_workers=concurrent_batches,
        retry_interval_seconds=30.0,
        max_wait_seconds=7200,
        show_progress=False,
        show_batch_progress=False,
    )

    succeeded = [
        job
        for job in simulated
        if (
            job.status
            == "SIMULATED"
            and job.raw_result
            is not None
        )
    ]

    failed = [
        job
        for job in simulated
        if (
            job.status
            != "SIMULATED"
            or job.raw_result
            is None
        )
    ]

    log.append(
        f"BRAIN succeeded: "
        f"{len(succeeded)}"
    )

    log.append(
        f"BRAIN failed: "
        f"{len(failed)}"
    )

    if len(jobs) == TARGET_SIMULATIONS_PER_ITERATION:
        log.append(
            "Throughput target reached: 8 concurrent multi-simulations x 10 alphas."
        )

    # --------------------------------------------------------
    # Persist returned BRAIN failures
    # --------------------------------------------------------

    for job in failed:

        params = dict(
            job.parameters or {}
        )

        append_failure_record(
            failure_file,
            {
                "created_at":
                    getattr(
                        job,
                        "updated_at",
                        None,
                    ),
                "job_id":
                    job.job_id,
                "template":
                    job.template,
                "fields":
                    list(job.fields),
                "window":
                    params.get(
                        "window"
                    ),
                "backfill_window":
                    params.get(
                        "backfill_window"
                    ),
                "direction":
                    params.get(
                        "direction"
                    ),
                "repair_target":
                    params.get(
                        "repair_target"
                    ),
                "compiler_expression":
                    job.compiler_expression,
                "brain_expression":
                    job.brain_expression,
                "status":
                    job.status,
                "error_type":
                    job.error_type,
                "error_message":
                    job.error_message,
            },
        )

    # --------------------------------------------------------
    # Normalize successful jobs ONLY
    # --------------------------------------------------------

    normalized = []

    for job in succeeded:

        normalized.append(
            normalize_result(
                job.raw_result,
                alpha_type=job.alpha_type,
                fields=job.fields,
                template=job.template,
                compiler_expression=job.compiler_expression,
            )
        )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scores = score_batch(
        normalized
    )

    scores_by_alpha = {
        score.alpha_id:
            score
        for score in scores
    }

    prepared_by_alpha = {
        job.alpha_id:
            (
                spec,
                job,
            )
        for spec, job
        in prepared
        if job.alpha_id
        is not None
    }

    # --------------------------------------------------------
    # Persist successful research results
    # --------------------------------------------------------

    persisted = 0

    for result in normalized:

        alpha_id = result.alpha_id

        if (
            alpha_id
            not in prepared_by_alpha
        ):
            continue

        spec, job = (
            prepared_by_alpha[
                alpha_id
            ]
        )

        score = scores_by_alpha.get(
            alpha_id
        )

        if score is None:
            continue

        memory.append_result(
            result,
            research_score=score,
            metadata={
                "source":
                    "dashboard_research_loop",
                "region":
                    region,
                "universe":
                    universe,
                "delay":
                    delay,
                "dataset":
                    dataset_id,
                "template":
                    spec.template,
                "fields":
                    list(spec.fields),
                "window":
                    spec.window,
                "backfill_window":
                    spec.backfill_window,
                "direction":
                    spec.direction,
                "repair_target":
                    spec.repair_target,
                "repair_reason":
                    getattr(spec, "repair_reason", ""),
                "intuition":
                    spec.intuition,
            },
        )

        persisted += 1

    log.append(
        f"Successful results persisted: "
        f"{persisted}"
    )

    return {
        "insight":
            insight,
        "specs":
            specs,
        "jobs":
            simulated,
        "results":
            normalized,
        "scores":
            scores,
        "failed_jobs":
            failed,
        "log":
            log,
    }


# ============================================================
# INITIAL CONNECTION
# ============================================================

try:

    session = ensure_brain_session(
    get_brain_session()
)

except Exception as exc:

    st.error(
        f"BRAIN connection failed: "
        f"{exc}"
    )

    st.stop()


# ============================================================
# SIDEBAR CONFIG
# ============================================================

with st.sidebar:

    st.header(
        "BRAIN Configuration"
    )

    region_input = st.text_input(
        "Region",
        value=st.session_state.get(
            "region",
            "GLB",
        ),
        placeholder="GLB / USA / IND / EUR",
    ).strip().upper()

    universe_input = st.text_input(
        "Universe",
        value=st.session_state.get(
            "universe",
            "TOPDIV3000",
        ),
        placeholder="TOP3000 / TOPDIV3000",
    ).strip().upper()

    delay = st.selectbox(
        "Delay",
        options=[1, 0],
        index=0,
        help=(
            "The current research engine is primarily "
            "validated on Delay 1."
        ),
    )

    load_config = st.button(
        "Load Configuration",
        type="primary",
        use_container_width=True,
    )

    if load_config:

        if not region_input:
            st.error(
                "Enter a region."
            )

        elif not universe_input:
            st.error(
                "Enter a universe."
            )

        else:

            changed = (
                region_input
                != st.session_state.get(
                    "region"
                )
                or universe_input
                != st.session_state.get(
                    "universe"
                )
                or delay
                != st.session_state.get(
                    "delay",
                    DEFAULT_DELAY,
                )
            )

            st.session_state[
                "region"
            ] = region_input

            st.session_state[
                "universe"
            ] = universe_input

            st.session_state[
                "delay"
            ] = delay

            if changed:
                st.session_state.pop(
                    "dataset_id",
                    None,
                )
                st.session_state.pop(
                    "last_iteration",
                    None,
                )

            st.rerun()


# ============================================================
# ACTIVE CONFIG
# ============================================================

region = st.session_state.get(
    "region",
    "GLB",
)

universe = st.session_state.get(
    "universe",
    "TOPDIV3000",
)

delay = st.session_state.get(
    "delay",
    DEFAULT_DELAY,
)


# ============================================================
# DATASETS
# ============================================================

try:

    datasets_df = load_datasets(
        session,
        region,
        universe,
        delay,
    )

except Exception as exc:

    st.error(
        f"Dataset loading failed: "
        f"{exc}"
    )

    st.stop()


# ============================================================
# DATASET OPTIONS
# ============================================================

dataset_options = []

for _, row in datasets_df.iterrows():

    dataset_id = str(
        row.get(
            "id",
            row.get(
                "dataset_id",
                "",
            ),
        )
    )

    dataset_name = str(
        row.get(
            "name",
            row.get(
                "dataset_name",
                dataset_id,
            ),
        )
    )

    if dataset_id and dataset_id.lower() != "nan":
        dataset_options.append(
            (
                dataset_id,
                dataset_name if dataset_name.lower() != "nan" else dataset_id,
            )
        )


# Remove accidental duplicate IDs.
dataset_options = list(
    dict.fromkeys(
        dataset_options
    )
)

if not dataset_options:

    st.error(
        "BRAIN returned no selectable datasets."
    )

    st.stop()


dataset_ids = [
    item[0]
    for item in dataset_options
]


def format_dataset(
    dataset_id,
):
    for did, name in dataset_options:
        if did == dataset_id:
            return (
                f"{did} — {name}"
            )

    return dataset_id


# ============================================================
# DATASET SELECTOR
# ============================================================

with st.sidebar:

    st.divider()

    st.header(
        "Dataset"
    )

    saved_dataset = st.session_state.get(
        "dataset_id"
    )

    dataset_select_options = [
        "__SELECT_DATASET__"
    ] + dataset_ids

    if saved_dataset in dataset_ids:
        select_index = dataset_select_options.index(
            saved_dataset
        )
    else:
        select_index = 0

    selected_dataset = st.selectbox(
        "Choose dataset",
        options=dataset_select_options,
        index=select_index,
        format_func=(
            lambda value: (
                "Select a dataset..."
                if value == "__SELECT_DATASET__"
                else format_dataset(value)
            )
        ),
    )

    if selected_dataset == "__SELECT_DATASET__":
        # Crucially, do not start any datafield request until the user has
        # selected an actual dataset.
        st.session_state.pop(
            "dataset_id",
            None,
        )
        st.info(
            "Select a dataset to load its fields and research metadata."
        )
        dataset_id = None
    else:
        if selected_dataset != st.session_state.get(
            "dataset_id"
        ):
            st.session_state[
                "dataset_id"
            ] = selected_dataset

            st.session_state.pop(
                "last_iteration",
                None,
            )

        dataset_id = selected_dataset


# ============================================================
# LOAD SELECTED DATASET FIELDS
# ============================================================

if dataset_id is None:
    # No dataset has been chosen yet. Do not touch /data-fields.
    fields_df = pd.DataFrame()
    catalog = None
    compiler = None
    validator = None
    memory = None
    analyst = None
    failure_file = None
    records = []

else:
    try:

        fields_df = load_dataset_fields(
            session,
            region,
            universe,
            dataset_id,
            delay,
        )

        catalog = build_catalog(
            fields_df
        )

        compiler, validator = build_engine(
            catalog
        )

        memory = get_memory(
            region,
            universe,
            dataset_id,
            delay,
        )

        analyst = create_analyst(
            config_key(
                region,
                universe,
                dataset_id,
                delay,
            ),
            tuple(
                catalog[
                    "field_alias_to_id"
                ].values()
            ),
            tuple(
                catalog[
                    "seed_fields"
                ]
            ),
        )

        failure_file = failure_path(
            region,
            universe,
            dataset_id,
            delay,
        )

        # Load research memory once per Streamlit rerun and reuse it
        # throughout the dashboard.
        records = memory.load()

    except Exception as exc:

        st.error(
            f"Selected dataset initialization failed: "
            f"{exc}"
        )

        st.stop()


# ============================================================
# DATASET METADATA
# ============================================================

def dataset_record(dataset_id):
    if not dataset_id:
        return None

    matches = datasets_df[
        datasets_df["id"].astype(str) == str(dataset_id)
    ] if "id" in datasets_df.columns else pd.DataFrame()

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def _display_value(value):
    if value is None:
        return "—"
    if isinstance(value, float) and pd.isna(value):
        return "—"
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return str(value)
    return str(value)

selected_dataset_row = dataset_record(dataset_id)


# ============================================================
# CURRENT CONFIG HEADER
# ============================================================

if dataset_id is None:

    st.subheader(
        f"{region} / {universe} / Delay {delay}"
    )

    st.info(
        "Choose a dataset in the sidebar. No dataset fields are loaded until a dataset is selected."
    )

else:

    st.subheader(
        f"{region} / {universe} / "
        f"{dataset_id} / Delay {delay}"
    )

    if selected_dataset_row:
        dataset_name = _display_value(
            selected_dataset_row.get("name", dataset_id)
        )
        dataset_description = _display_value(
            selected_dataset_row.get("description", "")
        )

        st.markdown(
            f"### {dataset_name}"
        )

        if dataset_description not in {"", "—"}:
            st.write(
                dataset_description
            )

        metric_values = [
            ("Value score", selected_dataset_row.get("valueScore")),
            ("Pyramid multiplier", selected_dataset_row.get("pyramidMultiplier")),
            ("Coverage", selected_dataset_row.get("coverage")),
            ("Date coverage", selected_dataset_row.get("dateCoverage")),
            ("Users", selected_dataset_row.get("userCount")),
            ("Alphas", selected_dataset_row.get("alphaCount")),
            ("Fields", selected_dataset_row.get("fieldCount", len(fields_df))),
            ("Updated", selected_dataset_row.get("dateUpdated")),
        ]

        metric_cols = st.columns(4)

        for i, (label, value) in enumerate(metric_values):
            with metric_cols[i % 4]:
                st.metric(
                    label,
                    _display_value(value),
                )

        with st.expander("Full dataset metadata", expanded=False):
            metadata_rows = []
            for key, value in selected_dataset_row.items():
                metadata_rows.append({
                    "Property": key,
                    "Value": _display_value(value),
                })

            st.dataframe(
                pd.DataFrame(metadata_rows),
                use_container_width=True,
                hide_index=True,
            )

            papers = selected_dataset_row.get("researchPapers")
            if papers:
                st.markdown("**Research papers / resources**")
                for paper in papers if isinstance(papers, list) else [papers]:
                    if isinstance(paper, dict):
                        title = _display_value(paper.get("title", "Resource"))
                        url = paper.get("url")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.write(f"- {title}")
                    else:
                        st.write(f"- {_display_value(paper)}")

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Datasets",
            len(dataset_options),
        )

    with b:
        st.metric(
            "Dataset fields",
            len(fields_df),
        )

    with c:
        st.metric(
            "MATRIX fields",
            len(
                catalog[
                    "field_alias_to_id"
                ]
            ),
        )

    with d:
        st.metric(
            "Research records",
            len(records),
        )


# ============================================================
# SIDEBAR ACTIONS
# ============================================================

with st.sidebar:

    st.divider()

    run_now = st.button(
        "Run Next Iteration",
        type="primary",
        use_container_width=True,
        disabled=(dataset_id is None),
    )

    refresh_catalog = st.button(
        "Refresh BRAIN Data",
        use_container_width=True,
    )

    if refresh_catalog:

        load_datasets.clear()
        load_dataset_fields.clear()
        # Keep the authenticated BRAIN session alive. Refreshing catalog data
        # must never force a new login.
        get_memory.clear()
        create_analyst.clear()

        st.session_state.pop(
            "last_iteration",
            None,
        )

        st.rerun()


# ============================================================
# RUN ITERATION
# ============================================================

if run_now and dataset_id is not None:

    with st.status(
        "Running research iteration...",
        expanded=True,
    ) as status:

        try:

            iteration = run_iteration(
                session=session,
                region=region,
                universe=universe,
                dataset_id=dataset_id,
                delay=delay,
                catalog=catalog,
                compiler=compiler,
                validator=validator,
                analyst=analyst,
                memory=memory,
                failure_file=failure_file,
            )

            st.session_state[
                "last_iteration"
            ] = iteration

            status.update(
                label="Iteration complete",
                state="complete",
            )

        except Exception as exc:

            status.update(
                label="Iteration failed",
                state="error",
            )

            st.exception(
                exc
            )


# ============================================================
# TABS
# ============================================================

failure_records = (
    load_failure_records(failure_file)
    if failure_file is not None
    else []
)

tab_overview, tab_iteration, tab_memory, tab_fields = st.tabs(
    [
        "Overview",
        "Latest Iteration",
        "Research Memory",
        "Dataset Fields",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    if dataset_id is None:
        st.info(
            "Select a dataset to view research analytics and dataset details."
        )

    else:
        promising = sum(
            1
            for r in records
            if r.research_class == "PROMISING"
        )

        failures = sum(
            1
            for r in records
            if r.research_class == "FAILURE"
        )

        ready = sum(
            1
            for r in records
            if r.research_class == "RESEARCH_READY"
        )

        a, b, c, d = st.columns(4)

        with a:
            st.metric("Experiments", len(records))

        with b:
            st.metric("Promising", promising)

        with c:
            st.metric("Research failures", failures)

        with d:
            st.metric("BRAIN failures", len(failure_records))

        if records:
            ranked = sorted(
                records,
                key=lambda r: (
                    r.research_score
                    if r.research_score is not None
                    else float("-inf")
                ),
                reverse=True,
            )

            rows = []
            for r in ranked[:15]:
                rows.append({
                    "Alpha": r.alpha_id,
                    "Template": r.template,
                    "Class": r.research_class,
                    "Score": r.research_score,
                    "OOS": r.oos_score,
                    "Test Sharpe": r.test_sharpe,
                    "Test Fitness": r.test_fitness,
                    "Turnover": r.test_turnover,
                })

            st.subheader("Top research leads")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        if failure_records:
            with st.expander("BRAIN failure ledger"):
                st.dataframe(
                    pd.DataFrame(failure_records),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# LATEST ITERATION
# ============================================================

with tab_iteration:

    if dataset_id is None:
        st.info("Select a dataset before running an iteration.")
    else:

        iteration = st.session_state.get(
            "last_iteration"
        )

        if iteration is None:

            st.info(
                "No iteration has been run for this configuration."
            )

        else:

            insight = iteration[
                "insight"
            ]

            st.subheader(
                "Analyst"
            )

            if insight is None:
                st.info(
                    "Cold-start iteration: deterministic baseline experiments were generated before using the LLM analyst."
                )
            else:
                st.write(
                    insight.summary
                )

                x, y = st.columns(2)

                with x:

                    st.write(
                        "**Fixable patterns**"
                    )

                    st.write(
                        insight.fixable_patterns
                    )

                with y:

                    st.write(
                        "**Avoid templates**"
                    )

                    st.write(
                        insight.avoid_templates
                    )

            st.subheader(
                "Generated research specs"
            )

            for i, spec in enumerate(
                iteration["specs"],
                1,
            ):

                st.write(
                    f"{i}. "
                    f"`{spec.template}` | "
                    f"{list(spec.fields)} | "
                    f"{spec.window}/"
                    f"{spec.backfill_window} | "
                    f"{spec.direction}"
                )

            if iteration[
                "scores"
            ]:

                rows = []

                for score in iteration[
                    "scores"
                ]:

                    rows.append({
                        "Alpha":
                            score.alpha_id,
                        "Score":
                            score.score,
                        "Class":
                            score.research_class,
                        "OOS":
                            score.oos_score,
                        "Consistency":
                            score.consistency_score,
                        "Turnover":
                            score.turnover_score,
                        "Robustness":
                            score.robustness_score,
                    })

                st.subheader(
                    "Successful BRAIN results"
                )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

            if iteration.get(
                "failed_jobs"
            ):

                failed_rows = []

                for job in iteration[
                    "failed_jobs"
                ]:

                    failed_rows.append({
                        "Status":
                            job.status,
                        "Job ID":
                            job.job_id,
                        "Template":
                            job.template,
                        "Error type":
                            job.error_type,
                        "Error":
                            job.error_message,
                        "Compiler":
                            job.compiler_expression,
                    })

                st.subheader(
                    "Failed BRAIN jobs"
                )

                st.dataframe(
                    pd.DataFrame(
                        failed_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader(
                "Execution log"
            )

            for line in iteration[
                "log"
            ]:

                st.code(
                    line
                )


# ============================================================
# RESEARCH MEMORY
# ============================================================

with tab_memory:

    if dataset_id is None:
        st.info("Select a dataset to view research memory.")
    else:

        if records:

            rows = []

            for r in reversed(
                records
            ):

                rows.append({
                    "Alpha":
                        r.alpha_id,
                    "Template":
                        r.template,
                    "Class":
                        r.research_class,
                    "Score":
                        r.research_score,
                    "OOS":
                        r.oos_score,
                    "Test Sharpe":
                        r.test_sharpe,
                    "Test Fitness":
                        r.test_fitness,
                    "Turnover":
                        r.test_turnover,
                    "Robustness":
                        r.robustness_score,
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No research memory exists for this configuration yet."
            )


# ============================================================
# DATASET FIELDS
# ============================================================

with tab_fields:

    if dataset_id is None:
        st.info(
            "Select a dataset to load and inspect its fields."
        )

    else:
        st.subheader(
            f"{dataset_id} — field explorer"
        )

        search_value = st.text_input(
            "Search fields",
            placeholder="Field ID, description, category, subcategory...",
        )

        display = fields_df.copy()

        display["alias"] = display["id"].map(
            catalog["id_to_field_alias"]
        )

        if search_value.strip():
            q = search_value.strip().lower()
            searchable = display.astype(str).agg(" ".join, axis=1).str.lower()
            display = display[searchable.str.contains(q, na=False)].copy()

        preferred = [
            "alias",
            "id",
            "description",
            "dataset_name",
            "type",
            "category_name",
            "subcategory_name",
            "dateCoverage",
            "coverage",
            "valueScore",
            "userCount",
            "alphaCount",
            "pyramidMultiplier",
            "dateCreated",
            "themes",
        ]

        available = [
            col
            for col in preferred
            if col in display.columns
        ]

        st.caption(
            f"Showing {len(display):,} of {len(fields_df):,} fields."
        )

        st.dataframe(
            display[available],
            use_container_width=True,
            hide_index=True,
            height=620,
        )

        st.subheader("Selected field details")

        field_ids = display["id"].astype(str).tolist() if not display.empty else []

        if field_ids:
            selected_field_id = st.selectbox(
                "Choose a field",
                field_ids,
                format_func=lambda fid: (
                    f"{catalog['id_to_field_alias'].get(fid, '—')} — {fid}"
                ),
            )

            selected_matches = fields_df[
                fields_df["id"].astype(str) == str(selected_field_id)
            ]

            if not selected_matches.empty:
                field_row = selected_matches.iloc[0].to_dict()

                field_name = _display_value(
                    field_row.get("id", selected_field_id)
                )
                st.markdown(
                    f"### {field_name}"
                )

                field_metric_values = [
                    ("Type", field_row.get("type")),
                    ("Coverage", field_row.get("coverage")),
                    ("Date coverage", field_row.get("dateCoverage")),
                    ("Value score", field_row.get("valueScore")),
                    ("Users", field_row.get("userCount")),
                    ("Alphas", field_row.get("alphaCount")),
                    ("Pyramid multiplier", field_row.get("pyramidMultiplier")),
                    ("Created", field_row.get("dateCreated")),
                ]

                cols = st.columns(4)
                for i, (label, value) in enumerate(field_metric_values):
                    with cols[i % 4]:
                        st.metric(label, _display_value(value))

                description = _display_value(
                    field_row.get("description", "")
                )
                if description not in {"", "—"}:
                    st.write(description)

                with st.expander("Full field metadata", expanded=False):
                    metadata_rows = [
                        {
                            "Property": key,
                            "Value": _display_value(value),
                        }
                        for key, value in field_row.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(metadata_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        elif search_value.strip():
            st.info("No fields match the current search.")

