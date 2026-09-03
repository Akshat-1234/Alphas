# ============================================================
# engine/llm.py
# ============================================================

"""
Local LLM research layer.

Architecture
------------

                    ┌─────────────────────┐
                    │      Ollama         │
                    │      Qwen3 8B       │
                    └──────────┬──────────┘
                               │
                               ▼
                     structured research
                         specifications
                               │
                               ▼
                    deterministic compiler
                               │
                               ▼
                         FASTEXPR
                               │
                               ▼
                         validator
                               │
                               ▼
                         WorldQuant
                            BRAIN
                               │
                               ▼
                        empirical results


IMPORTANT
---------
The LLM does NOT generate final FASTEXPR directly.

The LLM generates:
    - template
    - real BRAIN field IDs
    - window
    - backfill window
    - direction
    - research intuition

The deterministic compiler creates the FASTEXPR.

BRAIN remains the empirical judge.
"""


from __future__ import annotations


# ============================================================
# STANDARD LIBRARY
# ============================================================

import json
import re
import time

from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from urllib.error import HTTPError
from urllib.error import URLError

from urllib.request import Request
from urllib.request import urlopen


# ============================================================
# DEFAULT OLLAMA CONFIGURATION
# ============================================================

DEFAULT_OLLAMA_URL = (
    "http://127.0.0.1:11434"
)

DEFAULT_MODEL = (
    "qwen3:8b"
)

DEFAULT_ANALYST_MODEL = (
    "qwen3:8b"
)

DEFAULT_TIMEOUT = 90

DEFAULT_MAX_RETRIES = 1

DEFAULT_RETRY_DELAY = 1.5


# ============================================================
# FIELD SELECTION CONFIGURATION
# ============================================================

DEFAULT_FIELD_BATCH_SIZE = 25

DEFAULT_BATCH_TOP_K = 4

DEFAULT_FIELD_SHORTLIST_SIZE = 10


# ============================================================
# ALPHA GENERATION CONFIGURATION
# ============================================================

DEFAULT_CANDIDATE_COUNT = 20

DEFAULT_MAX_RESULT_CONTEXT = 100


# ============================================================
# TEXT LIMITS
# ============================================================

DEFAULT_MAX_FIELD_DESCRIPTION = 180

DEFAULT_MAX_OPERATOR_DEFINITION = 180

DEFAULT_MAX_OPERATOR_DESCRIPTION = 220


# ============================================================
# RESEARCH TEMPLATES
# ============================================================

SUPPORTED_TEMPLATES = (
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
)


TWO_FIELD_TEMPLATES = (
    "RATIO",
    "RATIO_STATE",
    "RATIO_CHANGE",
    "INTERACTION",
    "CONTRAST",
    "CORRELATION",
)


ONE_FIELD_TEMPLATES = (
    "LEVEL",
    "HISTORICAL_STATE",
    "CHANGE",
    "SMOOTHED",
    "STABILITY",
    "DECAY",
)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the local Ollama model.
    """

    base_url: str = (
        DEFAULT_OLLAMA_URL
    )

    model: str = (
        DEFAULT_MODEL
    )

    analyst_model: str = (
        DEFAULT_ANALYST_MODEL
    )

    timeout: int = (
        DEFAULT_TIMEOUT
    )

    max_retries: int = (
        DEFAULT_MAX_RETRIES
    )

    retry_delay: float = (
        DEFAULT_RETRY_DELAY
    )

    generation_temperature: float = 0.80

    analysis_temperature: float = 0.40

    field_temperature: float = 0.20

    critique_temperature: float = 0.25

    field_batch_size: int = (
        DEFAULT_FIELD_BATCH_SIZE
    )

    batch_top_k: int = (
        DEFAULT_BATCH_TOP_K
    )

    field_shortlist_size: int = (
        DEFAULT_FIELD_SHORTLIST_SIZE
    )


# ============================================================
# JSON CLEANING
# ============================================================

def strip_thinking(
    text: str,
) -> str:
    """
    Remove <think>...</think> sections.
    """

    if not text:
        return ""

    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    ).strip()


def strip_code_fences(
    text: str,
) -> str:
    """
    Remove Markdown code fences.
    """

    text = strip_thinking(
        text
    )

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def extract_json_array(
    text: str,
) -> list[Any]:
    """
    Extract a genuine JSON array.

    Nested arrays such as:
        {"fields": ["F1", "F2"]}

    are intentionally NOT mistaken for candidate arrays.
    """

    text = strip_code_fences(
        text
    )

    if not text:
        return []

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"\[",
        text,
    ):

        start = match.start()

        before = text[
            :start
        ].strip()

        # Avoid nested field/lookback arrays.
        if (
            before.endswith(
                '"fields":'
            )
            or before.endswith(
                '"lookbacks":'
            )
            or before.endswith(
                '"operators":'
            )
        ):
            continue

        try:

            value, _ = (
                decoder.raw_decode(
                    text[start:]
                )
            )

        except json.JSONDecodeError:

            continue

        if isinstance(
            value,
            list,
        ):

            return value

    return []


def extract_json_object(
    text: str,
) -> dict[str, Any]:
    """
    Extract the first valid JSON object.
    """

    text = strip_code_fences(
        text
    )

    if not text:
        return {}

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"\{",
        text,
    ):

        try:

            value, _ = (
                decoder.raw_decode(
                    text[
                        match.start():
                    ]
                )
            )

        except json.JSONDecodeError:

            continue

        if isinstance(
            value,
            dict,
        ):

            return value

    return {}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_template(
    value: Any,
) -> str:
    """
    Normalize an LLM template.
    """

    template = str(
        value or ""
    ).strip().upper()

    aliases = {
        "RAW": "LEVEL",
        "LEVEL_SIGNAL": "LEVEL",

        "STATE": "HISTORICAL_STATE",
        "HISTORICAL": "HISTORICAL_STATE",

        "DELTA": "CHANGE",
        "CHANGE_SIGNAL": "CHANGE",

        "MEAN": "SMOOTHED",
        "SMOOTH": "SMOOTHED",

        "VOLATILITY": "STABILITY",
        "STD": "STABILITY",

        "DECAY_LINEAR": "DECAY",

        "RELATIVE": "RATIO",
        "RATIO_CHANGE_SIGNAL": "RATIO_CHANGE",

        "INTERACT": "INTERACTION",
        "DIFFERENCE": "CONTRAST",

        "CORR": "CORRELATION",
    }

    return aliases.get(
        template,
        template,
    )


def normalize_direction(
    value: Any,
) -> str:
    """
    Normalize signal direction.
    """

    direction = str(
        value or ""
    ).strip().lower()

    if direction in {
        "negative",
        "short",
        "inverse",
        "reversed",
    }:

        return "negative"

    return "positive"


def normalize_family(
    family: Any,
) -> str:

    value = str(
        family or ""
    ).strip().lower()

    aliases = {
        "value": "valuation",
        "valuation": "valuation",

        "profit": "profitability",
        "profitability": "profitability",

        "quality": "quality",

        "growth": "growth",

        "debt": "leverage",
        "leverage": "leverage",

        "cash": "liquidity",
        "liquidity": "liquidity",

        "risk": "risk",
        "volatility": "risk",

        "sentiment": "sentiment",

        "change": "change",

        "other": "other",
        "unknown": "other",
    }

    return aliases.get(
        value,
        "other",
    )


def normalize_window(
    value: Any,
    *,
    default: int = 60,
) -> int:
    """
    Convert a model window into an integer.

    The deterministic compiler/validator remains authoritative
    over whether the final value is allowed.
    """

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return int(
            default
        )


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:
    """
    Minimal native HTTP client for Ollama.

    No OpenAI SDK.
    No external HTTP library.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
    ):

        self.config = (
            config
            or LLMConfig()
        )

        self.base_url = (
            self.config.base_url
            .rstrip("/")
        )

    # ========================================================
    # HTTP
    # ========================================================

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        url = (
            self.base_url
            + "/"
            + path.lstrip("/")
        )

        body = None

        if payload is not None:

            body = json.dumps(
                payload
            ).encode(
                "utf-8"
            )

        request = Request(
            url=url,
            data=body,
            method=method.upper(),
            headers={
                "Content-Type": (
                    "application/json"
                ),
                "Accept": (
                    "application/json"
                ),
            },
        )

        last_error: Exception | None = None

        for attempt in range(
            self.config.max_retries + 1
        ):

            try:

                with urlopen(
                    request,
                    timeout=self.config.timeout,
                ) as response:

                    raw = response.read()

                if not raw:

                    raise RuntimeError(
                        "Ollama returned an empty response."
                    )

                decoded = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                if not isinstance(
                    decoded,
                    dict,
                ):

                    raise RuntimeError(
                        "Ollama returned invalid JSON."
                    )

                return decoded

            except HTTPError as exc:

                try:

                    message = (
                        exc.read()
                        .decode(
                            "utf-8",
                            errors="replace",
                        )
                    )

                except Exception:

                    message = ""

                last_error = RuntimeError(
                    f"Ollama HTTP {exc.code}: "
                    f"{message or exc.reason}"
                )

            except URLError as exc:

                last_error = RuntimeError(
                    "Could not connect to Ollama: "
                    f"{exc.reason}"
                )

            except TimeoutError:

                last_error = RuntimeError(
                    "Ollama request timed out."
                )

            except json.JSONDecodeError as exc:

                last_error = RuntimeError(
                    "Ollama returned malformed JSON: "
                    f"{exc}"
                )

            except Exception as exc:

                last_error = exc

            if (
                attempt
                < self.config.max_retries
            ):

                time.sleep(
                    self.config.retry_delay
                )

        raise (
            last_error
            or RuntimeError(
                "Ollama request failed."
            )
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def is_running(
        self,
    ) -> bool:

        try:

            self._request_json(
                "GET",
                "/api/tags",
            )

            return True

        except Exception:

            return False

    # ========================================================
    # MODELS
    # ========================================================

    def list_models(
        self,
    ) -> list[str]:

        response = self._request_json(
            "GET",
            "/api/tags",
        )

        models = response.get(
            "models",
            [],
        )

        names = []

        for item in models:

            if not isinstance(
                item,
                Mapping,
            ):

                continue

            name = item.get(
                "name"
            )

            if name:

                names.append(
                    str(name)
                )

        return names

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        messages: Sequence[
            Mapping[str, str]
        ],
        *,
        model: str | None = None,
        temperature: float = 0.70,
        response_format: str | None = None,
    ) -> str:

        selected_model = (
            model
            or self.config.model
        )

        payload: dict[str, Any] = {
            "model": selected_model,

            "messages": [
                {
                    "role": str(
                        message.get(
                            "role",
                            "user",
                        )
                    ),

                    "content": str(
                        message.get(
                            "content",
                            "",
                        )
                    ),
                }
                for message
                in messages
            ],

            "stream": False,

            "options": {
                "temperature": float(
                    temperature
                ),
            },
        }

        if response_format == "json":

            payload["format"] = "json"

        response = self._request_json(
            "POST",
            "/api/chat",
            payload,
        )

        message = response.get(
            "message"
        )

        if not isinstance(
            message,
            Mapping,
        ):

            raise RuntimeError(
                "Ollama response has no message."
            )

        content = message.get(
            "content"
        )

        if content is None:

            raise RuntimeError(
                "Ollama returned empty content."
            )

        return str(
            content
        )


# ============================================================
# RESEARCH LLM
# ============================================================

class ResearchLLM:
    """
    High-level local research interface.
    """

    SYSTEM_PROMPT = """
You are a quantitative equity research assistant.

You are operating inside a deterministic alpha research engine.

Rules:

1. Use only supplied WorldQuant BRAIN field IDs.
2. Never invent a field ID.
3. Never invent an operator.
4. Never write arbitrary FASTEXPR when the task asks for a
   structured research specification.
5. Templates must come from the supplied template list.
6. Separate research hypotheses from empirical evidence.
7. Never claim an alpha is profitable before BRAIN simulation.
8. Prefer economically distinct hypotheses.
9. Return the requested JSON structure exactly.
""".strip()

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: OllamaClient | None = None,
    ):

        self.config = (
            config
            or LLMConfig()
        )

        self.client = (
            client
            or OllamaClient(
                self.config
            )
        )

    # ========================================================
    # BASIC REQUEST
    # ========================================================

    def ask(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.70,
        json_output: bool = False,
    ) -> str:

        if not prompt.strip():

            raise ValueError(
                "prompt cannot be empty."
            )

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self.client.chat(
            messages,
            model=model,
            temperature=temperature,
            response_format=(
                "json"
                if json_output
                else None
            ),
        )

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> str:

        if temperature is None:

            temperature = (
                self.config.generation_temperature
            )

        return self.ask(
            prompt,
            model=self.config.model,
            temperature=temperature,
            json_output=False,
        )

    # ========================================================
    # ANALYSIS
    # ========================================================

    def analyze(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> str:

        if temperature is None:

            temperature = (
                self.config.analysis_temperature
            )

        return self.ask(
            prompt,
            model=self.config.analyst_model,
            temperature=temperature,
            json_output=False,
        )

    # ========================================================
    # JSON GENERATION
    # ========================================================

    def generate_json_object(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:

        raw = self.ask(
            prompt,
            model=self.config.model,
            temperature=(
                temperature
                if temperature is not None
                else self.config.generation_temperature
            ),
            json_output=True,
        )

        return extract_json_object(
            raw
        )

    def generate_json_array(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> list[Any]:

        raw = self.ask(
            prompt,
            model=self.config.model,
            temperature=(
                temperature
                if temperature is not None
                else self.config.generation_temperature
            ),
            json_output=True,
        )

        parsed_array = (
            extract_json_array(
                raw
            )
        )

        if parsed_array:

            return parsed_array

        parsed_object = (
            extract_json_object(
                raw
            )
        )

        # ----------------------------------------------------
        # Candidate wrapper.
        # ----------------------------------------------------

        candidates = parsed_object.get(
            "candidates"
        )

        if isinstance(
            candidates,
            list,
        ):

            return candidates

        # ----------------------------------------------------
        # Single research candidate.
        # ----------------------------------------------------

        if "template" in parsed_object:

            return [
                parsed_object
            ]

        # ----------------------------------------------------
        # Single generated expression.
        #
        # Kept only for backwards compatibility with malformed
        # model responses. The main architecture does not ask
        # the model for expressions.
        # ----------------------------------------------------

        if "expression" in parsed_object:

            return [
                parsed_object
            ]

        return []

    def analyze_json(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:

        raw = self.ask(
            prompt,
            model=self.config.analyst_model,
            temperature=(
                temperature
                if temperature is not None
                else self.config.analysis_temperature
            ),
            json_output=True,
        )

        return extract_json_object(
            raw
        )


# ============================================================
# FIELD CATALOG
# ============================================================

def build_field_catalog(
    datafields: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:

    if datafields is None:

        raise ValueError(
            "datafields cannot be None."
        )

    if not hasattr(
        datafields,
        "iterrows",
    ):

        raise TypeError(
            "datafields must be DataFrame-like."
        )

    frame = datafields.copy()

    if "id" not in frame.columns:

        raise ValueError(
            "Datafield catalog must contain 'id'."
        )

    if "alphaCount" in frame.columns:

        frame = frame.sort_values(
            "alphaCount",
            ascending=False,
            na_position="last",
        )

    if limit is not None:

        frame = frame.head(
            int(limit)
        )

    result = []

    for _, row in frame.iterrows():

        field_id = str(
            row.get(
                "id",
                "",
            )
        ).strip()

        if not field_id:

            continue

        result.append({
            "id": field_id,

            "name": str(
                row.get(
                    "name",
                    "",
                )
            ).strip(),

            "description": str(
                row.get(
                    "description",
                    "",
                )
            ).strip()[
                :DEFAULT_MAX_FIELD_DESCRIPTION
            ],

            "type": str(
                row.get(
                    "type",
                    "",
                )
            ).strip(),

            "coverage": row.get(
                "coverage",
                None,
            ),

            "alphaCount": row.get(
                "alphaCount",
                None,
            ),
        })

    return result


# ============================================================
# OPERATOR CATALOG
# ============================================================

def build_operator_catalog(
    operators: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:

    if operators is None:

        raise ValueError(
            "operators cannot be None."
        )

    if not hasattr(
        operators,
        "iterrows",
    ):

        raise TypeError(
            "operators must be DataFrame-like."
        )

    frame = operators.copy()

    required = [
        "name",
        "category",
        "definition",
    ]

    missing = [
        column
        for column
        in required
        if column not in frame.columns
    ]

    if missing:

        raise ValueError(
            "Operator catalog missing columns: "
            + ", ".join(missing)
        )

    result = []

    for _, row in frame.iterrows():

        name = str(
            row.get(
                "name",
                "",
            )
        ).strip()

        if not name:

            continue

        result.append({
            "name": name,

            "category": str(
                row.get(
                    "category",
                    "",
                )
            ).strip(),

            "definition": str(
                row.get(
                    "definition",
                    "",
                )
            ).strip()[
                :DEFAULT_MAX_OPERATOR_DEFINITION
            ],

            "description": str(
                row.get(
                    "description",
                    "",
                )
            ).strip()[
                :DEFAULT_MAX_OPERATOR_DESCRIPTION
            ],
        })

    if limit is not None:

        result = result[
            :int(limit)
        ]

    return result


# ============================================================
# FAMILY INFERENCE
# ============================================================

def infer_field_family(
    field: Mapping[str, Any],
) -> str:
    """
    Deterministic family inference used to improve diversity.
    """

    text = " ".join([
        str(
            field.get(
                "id",
                "",
            )
        ),
        str(
            field.get(
                "name",
                "",
            )
        ),
        str(
            field.get(
                "description",
                "",
            )
        ),
    ]).lower()

    keyword_groups = {
        "liquidity": [
            "cash",
            "liquidity",
            "current asset",
            "current liability",
            "short-term investment",
            "working capital",
            "receivable",
        ],

        "leverage": [
            "debt",
            "leverage",
            "liabilities",
            "borrow",
            "capital structure",
            "interest coverage",
        ],

        "growth": [
            "growth",
            "increase",
            "sales growth",
            "revenue growth",
            "earnings growth",
        ],

        "profitability": [
            "income",
            "profit",
            "earnings",
            "operating income",
            "ebit",
            "ebitda",
            "eps",
            "margin",
        ],

        "valuation": [
            "valuation",
            "book value",
            "enterprise value",
            "market value",
            "multiple",
            "yield",
        ],

        "risk": [
            "volatility",
            "risk",
            "uncertainty",
            "standard deviation",
            "beta",
        ],

        "sentiment": [
            "sentiment",
            "estimate",
            "analyst",
            "forecast",
            "surprise",
            "revision",
        ],

        "quality": [
            "return on",
            "quality",
            "accrual",
            "efficiency",
            "turnover",
            "consistency",
        ],

        "change": [
            "delta",
            "change",
            "difference",
            "revision",
        ],
    }

    scores = {
        family: 0
        for family
        in (
            "valuation",
            "profitability",
            "quality",
            "growth",
            "leverage",
            "liquidity",
            "risk",
            "sentiment",
            "change",
            "other",
        )
    }

    for family, keywords in (
        keyword_groups.items()
    ):

        for keyword in keywords:

            if keyword in text:

                scores[
                    family
                ] += 1

    best = max(
        scores,
        key=scores.get,
    )

    if scores[
        best
    ] == 0:

        return "other"

    return best


# ============================================================
# FIELD SELECTION BATCH PROMPT
# ============================================================

def build_field_batch_prompt(
    *,
    region: str,
    universe: str,
    fields: Sequence[
        Mapping[str, Any]
    ],
    top_k: int,
) -> str:

    return f"""
Rank the datafields in this batch for quantitative equity
alpha research.

Region:
{region}

Universe:
{universe}

Choose up to {top_k} strongest fields.

The "field" MUST exactly equal an "id" shown below.

Never invent a field.
Never output lookback values as fields.
Never output numbers as field IDs.

Evaluate:

- economic usefulness
- distinct information
- potential cross-sectional usefulness
- potential for fundamental signal construction
- diversity versus other likely fundamental mechanisms

Possible families:

valuation
profitability
quality
growth
leverage
liquidity
risk
sentiment
change
other

Return EXACTLY:

{{
  "selections": [
    {{
      "field": "EXACT_FIELD_ID",
      "family": "quality",
      "score": 5,
      "reason": "one concise sentence"
    }}
  ]
}}

Score:

1 = weak
2 = below average
3 = useful
4 = strong
5 = very strong

Fields:

{json.dumps(
    list(fields),
    indent=2,
    default=str,
)}
""".strip()


# ============================================================
# BATCH FIELD SELECTION
# ============================================================

def _select_field_batch(
    llm: ResearchLLM,
    *,
    region: str,
    universe: str,
    fields: Sequence[
        Mapping[str, Any]
    ],
    top_k: int,
) -> list[dict[str, Any]]:

    prompt = build_field_batch_prompt(
        region=region,
        universe=universe,
        fields=fields,
        top_k=top_k,
    )

    raw = llm.ask(
        prompt,
        model=llm.config.model,
        temperature=llm.config.field_temperature,
        json_output=True,
    )

    parsed = extract_json_object(
        raw
    )

    selections = parsed.get(
        "selections",
        [],
    )

    if not isinstance(
        selections,
        list,
    ):

        return []

    allowed = {
        str(
            item.get(
                "id",
                "",
            )
        ).strip()
        for item
        in fields
    }

    result = []

    seen = set()

    for item in selections:

        if not isinstance(
            item,
            Mapping,
        ):

            continue

        field_id = str(
            item.get(
                "field",
                "",
            )
        ).strip()

        if field_id not in allowed:

            continue

        if field_id in seen:

            continue

        try:

            score = float(
                item.get(
                    "score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        result.append({
            "field": field_id,

            "family": normalize_family(
                item.get(
                    "family",
                    "other",
                )
            ),

            "score": score,

            "reason": str(
                item.get(
                    "reason",
                    "",
                )
            ).strip(),
        })

        seen.add(
            field_id
        )

    result.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    return result[
        :top_k
    ]


# ============================================================
# DIVERSITY SELECTOR
# ============================================================

def select_diverse_fields(
    candidates: Sequence[
        Mapping[str, Any]
    ],
    *,
    target_size: int,
) -> list[dict[str, Any]]:
    """
    Deterministic family-diversity selection.

    Raw model score alone is not trusted because small local
    models often saturate scores at 5.
    """

    target_size = max(
        1,
        int(target_size),
    )

    best_by_field = {}

    for candidate in candidates:

        field_id = str(
            candidate.get(
                "field",
                "",
            )
        ).strip()

        if not field_id:

            continue

        family = normalize_family(
            candidate.get(
                "family",
                "other",
            )
        )

        try:

            score = float(
                candidate.get(
                    "score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        candidate = {
            **dict(candidate),

            "field": field_id,

            "family": family,

            "score": score,
        }

        existing = best_by_field.get(
            field_id
        )

        if (
            existing is None
            or score
            > existing["score"]
        ):

            best_by_field[
                field_id
            ] = candidate

    candidates = list(
        best_by_field.values()
    )

    candidates.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    families = {}

    for candidate in candidates:

        family = candidate[
            "family"
        ]

        families.setdefault(
            family,
            [],
        ).append(
            candidate
        )

    selected = []

    used = set()

    family_priority = [
        "valuation",
        "profitability",
        "quality",
        "growth",
        "leverage",
        "liquidity",
        "risk",
        "sentiment",
        "change",
        "other",
    ]

    # --------------------------------------------------------
    # One candidate per available family.
    # --------------------------------------------------------

    for family in family_priority:

        if len(
            selected
        ) >= target_size:

            break

        members = families.get(
            family,
            [],
        )

        if not members:

            continue

        candidate = members[0]

        if candidate[
            "field"
        ] in used:

            continue

        selected.append(
            candidate
        )

        used.add(
            candidate[
                "field"
            ]
        )

    # --------------------------------------------------------
    # Fill remaining slots.
    # --------------------------------------------------------

    remaining = [
        candidate
        for candidate
        in candidates
        if candidate[
            "field"
        ] not in used
    ]

    remaining.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    family_counts = {}

    for candidate in selected:

        family = candidate[
            "family"
        ]

        family_counts[
            family
        ] = (
            family_counts.get(
                family,
                0,
            )
            + 1
        )

    max_per_family = max(
        2,
        (
            target_size
            + 2
        )
        // 4,
    )

    for candidate in remaining:

        if len(
            selected
        ) >= target_size:

            break

        family = candidate[
            "family"
        ]

        if (
            family_counts.get(
                family,
                0,
            )
            >= max_per_family
        ):

            continue

        selected.append(
            candidate
        )

        used.add(
            candidate[
                "field"
            ]
        )

        family_counts[
            family
        ] = (
            family_counts.get(
                family,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # Absolute fallback.
    # --------------------------------------------------------

    for candidate in remaining:

        if len(
            selected
        ) >= target_size:

            break

        if candidate[
            "field"
        ] in used:

            continue

        selected.append(
            candidate
        )

        used.add(
            candidate[
                "field"
            ]
        )

    return selected[
        :target_size
    ]


# ============================================================
# PUBLIC FIELD SELECTION
# ============================================================

def select_fields_batched(
    llm: ResearchLLM,
    *,
    region: str,
    universe: str,
    datafields: Any,
    shortlist_size: int = DEFAULT_FIELD_SHORTLIST_SIZE,
    batch_size: int | None = None,
    batch_top_k: int | None = None,
) -> list[dict[str, Any]]:

    catalog = build_field_catalog(
        datafields
    )

    if not catalog:

        return []

    batch_size = (
        int(batch_size)
        if batch_size is not None
        else llm.config.field_batch_size
    )

    batch_top_k = (
        int(batch_top_k)
        if batch_top_k is not None
        else llm.config.batch_top_k
    )

    shortlist_size = max(
        1,
        int(shortlist_size),
    )

    batch_size = max(
        1,
        batch_size,
    )

    batch_top_k = max(
        1,
        batch_top_k,
    )

    batch_count = (
        len(catalog)
        + batch_size
        - 1
    ) // batch_size

    print(
        f"FIELD SELECTION: "
        f"{len(catalog)} fields "
        f"→ {batch_count} batches"
    )

    all_candidates = []

    for start in range(
        0,
        len(catalog),
        batch_size,
    ):

        batch = catalog[
            start:
            start + batch_size
        ]

        batch_number = (
            start
            // batch_size
            + 1
        )

        print(
            f"  Batch {batch_number}/"
            f"{batch_count}: "
            f"{len(batch)} fields"
        )

        try:

            selected = _select_field_batch(
                llm,
                region=region,
                universe=universe,
                fields=batch,
                top_k=batch_top_k,
            )

        except Exception as exc:

            print(
                f"  Batch {batch_number} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            selected = []

        metadata = {
            str(
                field["id"]
            ).strip(): field
            for field
            in batch
        }

        for candidate in selected:

            field_id = candidate[
                "field"
            ]

            field = metadata.get(
                field_id
            )

            if field is None:

                continue

            family = normalize_family(
                candidate.get(
                    "family",
                    "other",
                )
            )

            if family == "other":

                family = infer_field_family(
                    field
                )

            all_candidates.append({
                "field": field_id,

                "family": family,

                "score": candidate[
                    "score"
                ],

                "reason": candidate[
                    "reason"
                ],

                "name": str(
                    field.get(
                        "name",
                        "",
                    )
                ).strip(),

                "description": str(
                    field.get(
                        "description",
                        "",
                    )
                ).strip(),

                "type": str(
                    field.get(
                        "type",
                        "",
                    )
                ).strip(),

                "coverage": field.get(
                    "coverage",
                    None,
                ),

                "alphaCount": field.get(
                    "alphaCount",
                    None,
                ),
            })

    if not all_candidates:

        return []

    selected = select_diverse_fields(
        all_candidates,
        target_size=shortlist_size,
    )

    result = [
        {
            "field": item[
                "field"
            ],

            "family": item[
                "family"
            ],

            "direction": "unknown",

            "batch_score": item[
                "score"
            ],

            "reason": item[
                "reason"
            ],
        }
        for item
        in selected
    ]

    print(
        "FINAL FIELD SHORTLIST:",
        len(result),
    )

    return result


def select_fields(
    llm: ResearchLLM,
    *,
    region: str,
    universe: str,
    datafields: Any,
    shortlist_size: int = DEFAULT_FIELD_SHORTLIST_SIZE,
) -> list[dict[str, Any]]:

    return select_fields_batched(
        llm,
        region=region,
        universe=universe,
        datafields=datafields,
        shortlist_size=shortlist_size,
    )


# ============================================================
# RESEARCH SPECIFICATION PROMPT
# ============================================================

def build_generation_prompt(
    *,
    region: str,
    universe: str,
    selected_fields: Sequence[
        Mapping[str, Any]
    ],
    candidate_count: int,
    research_direction: str = "",
    previous_candidates: Sequence[
        Mapping[str, Any]
    ] | None = None,
) -> str:
    """
    Ask the LLM for structured research specifications.

    No FASTEXPR is requested.
    """

    previous = list(
        previous_candidates
        or []
    )[
        :DEFAULT_MAX_RESULT_CONTEXT
    ]

    field_ids = [
        str(
            field.get(
                "field",
                "",
            )
        ).strip()
        for field
        in selected_fields
        if field.get(
            "field"
        )
    ]

    return f"""
Generate exactly {candidate_count} DIFFERENT quantitative
alpha research specifications for WorldQuant BRAIN.

Region:
{region}

Universe:
{universe}

IMPORTANT ARCHITECTURE:

You are NOT generating FASTEXPR directly.

You are proposing a STRUCTURED RESEARCH SPECIFICATION.

The deterministic compiler will convert your specification
into FASTEXPR later.

Allowed templates ONLY:

{json.dumps(
    list(SUPPORTED_TEMPLATES),
    indent=2,
)}

Template field requirements:

One-field templates:
{json.dumps(
    list(ONE_FIELD_TEMPLATES),
    indent=2,
)}

Two-field templates:
{json.dumps(
    list(TWO_FIELD_TEMPLATES),
    indent=2,
)}

Field rules:

- Use ONLY the exact field IDs listed below.
- Never invent field IDs.
- Never use aliases such as F1.
- Never write conceptual variables.
- A one-field template must use exactly one field.
- A two-field template must use exactly two fields.
- Fields may repeat across candidates, but the hypotheses
  should be meaningfully different.

Window rules:

Use one of:
20
30
60
90
120
180
252

Backfill window must also use one of those values.

Direction:

"positive" or "negative"

Research direction:

{research_direction}

Selected fields:

{json.dumps(
    field_ids,
    indent=2,
)}

Previous candidates:

{json.dumps(
    previous,
    indent=2,
    default=str,
)}

Return EXACTLY ONE JSON OBJECT:

{{
  "candidates": [
    {{
      "template": "RATIO",
      "fields": [
        "exact_field_id_1",
        "exact_field_id_2"
      ],
      "window": 60,
      "backfill_window": 60,
      "direction": "positive",
      "family": "valuation",
      "intuition": "concise economic hypothesis"
    }}
  ]
}}

The candidates array MUST contain
{candidate_count} objects.

Do NOT include:
- expression
- FASTEXPR
- operator names
- invented field IDs
- commentary outside the JSON
""".strip()


# ============================================================
# SPECIFICATION NORMALIZATION
# ============================================================

def normalize_research_specification(
    candidate: Mapping[str, Any],
    *,
    allowed_fields: set[str],
) -> dict[str, Any] | None:
    """
    Normalize and structurally check one LLM research specification.

    This does NOT replace the deterministic compiler/validator.
    """

    template = normalize_template(
        candidate.get(
            "template"
        )
    )

    if template not in SUPPORTED_TEMPLATES:

        return None

    raw_fields = candidate.get(
        "fields",
        [],
    )

    if isinstance(
        raw_fields,
        str,
    ):

        raw_fields = [
            raw_fields
        ]

    if not isinstance(
        raw_fields,
        list,
    ):

        return None

    fields = [
        str(
            field
        ).strip()
        for field
        in raw_fields
        if str(
            field
        ).strip()
    ]

    required_count = (
        1
        if template in ONE_FIELD_TEMPLATES
        else 2
    )

    if len(fields) != required_count:

        return None

    if any(
        field not in allowed_fields
        for field
        in fields
    ):

        return None

    window = normalize_window(
        candidate.get(
            "window",
            60,
        )
    )

    backfill_window = normalize_window(
        candidate.get(
            "backfill_window",
            60,
        )
    )

    direction = normalize_direction(
        candidate.get(
            "direction",
            "positive",
        )
    )

    family = normalize_family(
        candidate.get(
            "family",
            "other",
        )
    )

    intuition = str(
        candidate.get(
            "intuition",
            "",
        )
    ).strip()

    return {
        "template": template,

        "fields": fields,

        "window": window,

        "backfill_window": backfill_window,

        "direction": direction,

        "family": family,

        "intuition": intuition,
    }


# ============================================================
# GENERATE RESEARCH SPECIFICATIONS
# ============================================================

def generate_candidates(
    llm: ResearchLLM,
    *,
    region: str,
    universe: str,
    selected_fields: Sequence[
        Mapping[str, Any]
    ],
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    research_direction: str = "",
    previous_candidates: Sequence[
        Mapping[str, Any]
    ] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate structured research specifications.

    IMPORTANT:
        The returned objects are NOT FASTEXPR.
    """

    candidate_count = max(
        1,
        int(candidate_count),
    )

    field_ids = {
        str(
            field.get(
                "field",
                "",
            )
        ).strip()
        for field
        in selected_fields
        if field.get(
            "field"
        )
    }

    if not field_ids:

        raise ValueError(
            "selected_fields contains no valid field IDs."
        )

    prompt = build_generation_prompt(
        region=region,
        universe=universe,
        selected_fields=selected_fields,
        candidate_count=candidate_count,
        research_direction=research_direction,
        previous_candidates=previous_candidates,
    )

    raw_items = llm.generate_json_array(
        prompt,
        temperature=llm.config.generation_temperature,
    )

    normalized = []

    seen = set()

    for item in raw_items:

        if not isinstance(
            item,
            Mapping,
        ):

            continue

        spec = normalize_research_specification(
            item,
            allowed_fields=field_ids,
        )

        if spec is None:

            continue

        # Exact structural deduplication.
        key = (
            spec["template"],
            tuple(spec["fields"]),
            spec["window"],
            spec["backfill_window"],
            spec["direction"],
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        normalized.append(
            spec
        )

        if len(
            normalized
        ) >= candidate_count:

            break

    return normalized


# ============================================================
# SPECIFICATION DEDUPLICATION
# ============================================================

def deduplicate_candidates(
    candidates: Iterable[
        Mapping[str, Any]
    ],
    *,
    max_candidates: int = DEFAULT_CANDIDATE_COUNT,
) -> list[dict[str, Any]]:
    """
    Deduplicate structured research specifications.
    """

    seen = set()

    result = []

    for candidate in candidates:

        if not isinstance(
            candidate,
            Mapping,
        ):

            continue

        template = normalize_template(
            candidate.get(
                "template"
            )
        )

        fields = candidate.get(
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

        fields = tuple(
            str(
                field
            ).strip()
            for field
            in fields
        )

        key = (
            template,
            fields,
            normalize_window(
                candidate.get(
                    "window",
                    60,
                )
            ),
            normalize_window(
                candidate.get(
                    "backfill_window",
                    60,
                )
            ),
            normalize_direction(
                candidate.get(
                    "direction",
                    "positive",
                )
            ),
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        result.append(
            dict(candidate)
        )

        if len(
            result
        ) >= max_candidates:

            break

    return result


# ============================================================
# CRITIQUE PROMPT
# ============================================================

def build_critique_prompt(
    *,
    candidates: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> str:

    return f"""
Critically review these quantitative alpha research
specifications.

This is a research critique, NOT a profitability judgment.

Assess:

1. economic plausibility
2. redundancy
3. overfitting risk
4. whether the fields support the proposed mechanism
5. whether the template fits the hypothesis
6. likely failure modes
7. whether candidates are genuinely different

Do not claim any candidate is profitable.

Allowed fields:

{json.dumps(
    list(allowed_fields),
    indent=2,
)}

Allowed templates:

{json.dumps(
    list(SUPPORTED_TEMPLATES),
    indent=2,
)}

Research specifications:

{json.dumps(
    list(candidates),
    indent=2,
    default=str,
)}

Return JSON ONLY:

{{
  "overall_assessment": "...",
  "strong_candidates": [
    {{
      "index": 1,
      "reason": "..."
    }}
  ],
  "weak_candidates": [
    {{
      "index": 1,
      "reason": "..."
    }}
  ],
  "redundancy_patterns": ["..."],
  "research_warnings": ["..."],
  "recommended_changes": ["..."]
}}
""".strip()


# ============================================================
# CRITIQUE
# ============================================================

def critique_candidates(
    llm: ResearchLLM,
    *,
    candidates: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> dict[str, Any]:

    prompt = build_critique_prompt(
        candidates=candidates,
        allowed_fields=allowed_fields,
    )

    return llm.generate_json_object(
        prompt,
        temperature=llm.config.critique_temperature,
    )


# ============================================================
# RESULT ANALYSIS PROMPT
# ============================================================

def build_result_analysis_prompt(
    *,
    simulation_results: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> str:

    results = list(
        simulation_results
    )[
        :DEFAULT_MAX_RESULT_CONTEXT
    ]

    return f"""
Analyze these empirical WorldQuant BRAIN simulation results.

BRAIN is the empirical judge.

Do not declare an alpha profitable merely from positive
performance in a small sample.

Focus on:

1. strongest research families
2. strongest template structures
3. fields recurring in stronger candidates
4. redundancy
5. regional fragility
6. turnover problems
7. repeated BRAIN test failures
8. research directions worth exploring next
9. fields that deserve more attention

Allowed fields:

{json.dumps(
    list(allowed_fields),
    indent=2,
)}

Allowed templates:

{json.dumps(
    list(SUPPORTED_TEMPLATES),
    indent=2,
)}

Results:

{json.dumps(
    results,
    indent=2,
    default=str,
)}

Return JSON ONLY:

{{
  "summary": "...",
  "promising_templates": ["..."],
  "strong_families": ["..."],
  "failure_patterns": ["..."],
  "redundancy_patterns": ["..."],
  "priority_fields": ["exact_field_id"],
  "priority_families": ["..."],
  "next_directions": ["..."]
}}
""".strip()


# ============================================================
# RESULT ANALYSIS
# ============================================================

def analyze_results(
    llm: ResearchLLM,
    *,
    simulation_results: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> dict[str, Any]:

    prompt = build_result_analysis_prompt(
        simulation_results=simulation_results,
        allowed_fields=allowed_fields,
    )

    return llm.analyze_json(
        prompt,
        temperature=llm.config.analysis_temperature,
    )


# ============================================================
# NEXT GENERATION PROMPT
# ============================================================

def build_next_generation_prompt(
    *,
    research_analysis: Mapping[str, Any],
    previous_results: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
    candidate_count: int,
) -> str:

    return f"""
Generate exactly {candidate_count} NEW structured alpha
research specifications.

Base them on the empirical research analysis below.

Do NOT generate FASTEXPR.

Use only:
- exact supplied field IDs
- supplied templates

Allowed templates:

{json.dumps(
    list(SUPPORTED_TEMPLATES),
    indent=2,
)}

Allowed fields:

{json.dumps(
    list(allowed_fields),
    indent=2,
)}

Previous research analysis:

{json.dumps(
    dict(research_analysis),
    indent=2,
    default=str,
)}

Previous simulation results:

{json.dumps(
    list(previous_results)[
        :DEFAULT_MAX_RESULT_CONTEXT
    ],
    indent=2,
    default=str,
)}

Use windows only from:

20, 30, 60, 90, 120, 180, 252

Return EXACTLY:

{{
  "candidates": [
    {{
      "template": "RATIO",
      "fields": [
        "exact_field_id_1",
        "exact_field_id_2"
      ],
      "window": 60,
      "backfill_window": 60,
      "direction": "positive",
      "family": "valuation",
      "intuition": "concise economic hypothesis"
    }}
  ]
}}

Do not include FASTEXPR.
""".strip()


# ============================================================
# NEXT GENERATION
# ============================================================

def generate_next_generation(
    llm: ResearchLLM,
    *,
    research_analysis: Mapping[str, Any],
    previous_results: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> list[dict[str, Any]]:

    prompt = build_next_generation_prompt(
        research_analysis=research_analysis,
        previous_results=previous_results,
        allowed_fields=allowed_fields,
        candidate_count=candidate_count,
    )

    raw = llm.generate_json_array(
        prompt,
        temperature=llm.config.generation_temperature,
    )

    allowed_set = {
        str(
            field
        ).strip()
        for field
        in allowed_fields
    }

    result = []

    seen = set()

    for item in raw:

        if not isinstance(
            item,
            Mapping,
        ):

            continue

        spec = normalize_research_specification(
            item,
            allowed_fields=allowed_set,
        )

        if spec is None:

            continue

        key = (
            spec["template"],
            tuple(spec["fields"]),
            spec["window"],
            spec["backfill_window"],
            spec["direction"],
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        result.append(
            spec
        )

        if len(
            result
        ) >= candidate_count:

            break

    return result


# ============================================================
# REPAIR PROMPT
# ============================================================

def build_repair_prompt(
    *,
    rejected_candidates: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> str:

    return f"""
Repair these invalid structured alpha research
specifications.

Do NOT generate FASTEXPR.

Allowed templates:

{json.dumps(
    list(SUPPORTED_TEMPLATES),
    indent=2,
)}

Allowed fields:

{json.dumps(
    list(allowed_fields),
    indent=2,
)}

Allowed windows:

20
30
60
90
120
180
252

Rejected candidates:

{json.dumps(
    list(rejected_candidates),
    indent=2,
    default=str,
)}

Return JSON ONLY:

{{
  "candidates": [
    {{
      "template": "LEVEL",
      "fields": ["exact_field_id"],
      "window": 60,
      "backfill_window": 60,
      "direction": "positive",
      "family": "quality",
      "intuition": "concise hypothesis"
    }}
  ]
}}
""".strip()


# ============================================================
# REPAIR
# ============================================================

def repair_candidates(
    llm: ResearchLLM,
    *,
    rejected_candidates: Sequence[
        Mapping[str, Any]
    ],
    allowed_fields: Sequence[str],
) -> list[dict[str, Any]]:

    if not rejected_candidates:

        return []

    prompt = build_repair_prompt(
        rejected_candidates=rejected_candidates,
        allowed_fields=allowed_fields,
    )

    raw = llm.generate_json_array(
        prompt,
        temperature=0.20,
    )

    allowed_set = {
        str(
            field
        ).strip()
        for field
        in allowed_fields
    }

    result = []

    for item in raw:

        if not isinstance(
            item,
            Mapping,
        ):

            continue

        spec = normalize_research_specification(
            item,
            allowed_fields=allowed_set,
        )

        if spec is None:

            continue

        result.append(
            spec
        )

    return deduplicate_candidates(
        result
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def check_ollama(
    llm: ResearchLLM,
) -> tuple[bool, str]:

    try:

        if not llm.client.is_running():

            return (
                False,
                "Ollama is not reachable.",
            )

        models = (
            llm.client.list_models()
        )

        model = (
            llm.config.model
        )

        if model not in models:

            return (
                False,
                (
                    f"Model '{model}' "
                    "is not installed. "
                    f"Available: {models}"
                ),
            )

        return (
            True,
            "ok",
        )

    except Exception as exc:

        return (
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# TINY MODEL TEST
# ============================================================

def test_llm(
    llm: ResearchLLM,
) -> tuple[bool, str]:

    try:

        response = llm.ask(
            "Reply with exactly: LLM_OK",
            model=llm.config.model,
            temperature=0.0,
            json_output=False,
        )

        if "LLM_OK" in response.upper():

            return (
                True,
                "ok",
            )

        return (
            False,
            f"Unexpected response: {response}",
        )

    except Exception as exc:

        return (
            False,
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# FACTORIES
# ============================================================

def create_default_llm(
    *,
    model: str = DEFAULT_MODEL,
) -> ResearchLLM:

    return ResearchLLM(
        LLMConfig(
            model=model,
            analyst_model=model,
        )
    )


def create_deep_analyst_llm() -> ResearchLLM:
    """
    Optional slower analyst.

    Generator:
        Qwen3 8B

    Analyst:
        DeepSeek-R1 8B
    """

    return ResearchLLM(
        LLMConfig(
            model="qwen3:8b",
            analyst_model="deepseek-r1:8b",
        )
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "LLMConfig",

    "OllamaClient",
    "ResearchLLM",

    "SUPPORTED_TEMPLATES",
    "ONE_FIELD_TEMPLATES",
    "TWO_FIELD_TEMPLATES",

    "strip_thinking",
    "strip_code_fences",
    "extract_json_array",
    "extract_json_object",

    "normalize_template",
    "normalize_direction",
    "normalize_family",
    "normalize_window",

    "build_field_catalog",
    "build_operator_catalog",

    "infer_field_family",

    "build_field_batch_prompt",
    "select_diverse_fields",
    "select_fields_batched",
    "select_fields",

    "build_generation_prompt",
    "normalize_research_specification",
    "generate_candidates",
    "deduplicate_candidates",

    "build_critique_prompt",
    "critique_candidates",

    "build_repair_prompt",
    "repair_candidates",

    "build_result_analysis_prompt",
    "analyze_results",

    "build_next_generation_prompt",
    "generate_next_generation",

    "check_ollama",
    "test_llm",

    "create_default_llm",
    "create_deep_analyst_llm",
]