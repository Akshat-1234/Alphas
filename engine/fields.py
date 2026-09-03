# ============================================================
# engine/fields.py
# ============================================================

import pandas as pd


# ============================================================
# WORLDQUANT BRAIN FIELD TYPES
# ============================================================
#
# These are the field types actually observed in the live BRAIN
# catalog from your diagnostic:
#
#     MATRIX
#     VECTOR
#     GROUP
#     UNIVERSE
#     SYMBOL
#
# Do not add EVENT here unless BRAIN actually reports it.
# ============================================================

BRAIN_FIELD_TYPES = {
    "MATRIX",
    "VECTOR",
    "GROUP",
    "UNIVERSE",
    "SYMBOL",
}


# ============================================================
# TYPE NORMALIZATION
# ============================================================

def classify_field_type(
    raw_type,
) -> str:
    """
    Normalize the explicit BRAIN `type` value.

    We deliberately do not infer a field's type from its
    description or field name.

    Known BRAIN types:
        MATRIX
        VECTOR
        GROUP
        UNIVERSE
        SYMBOL

    Unknown / blank values are returned as UNKNOWN rather than
    silently being treated as MATRIX.
    """

    value = (
        str(raw_type)
        .strip()
        .upper()
    )

    if value in BRAIN_FIELD_TYPES:
        return value

    return "UNKNOWN"


# ============================================================
# TYPE PREDICATES
# ============================================================

def is_matrix_field(
    field_type: str,
) -> bool:
    """
    True only for BRAIN MATRIX fields.
    """

    return (
        classify_field_type(
            field_type
        )
        == "MATRIX"
    )


def is_vector_field(
    field_type: str,
) -> bool:
    """
    True only for BRAIN VECTOR fields.
    """

    return (
        classify_field_type(
            field_type
        )
        == "VECTOR"
    )


def is_group_field(
    field_type: str,
) -> bool:
    """
    True only for BRAIN GROUP fields.
    """

    return (
        classify_field_type(
            field_type
        )
        == "GROUP"
    )


def is_universe_field(
    field_type: str,
) -> bool:
    """
    True only for BRAIN UNIVERSE fields.
    """

    return (
        classify_field_type(
            field_type
        )
        == "UNIVERSE"
    )


def is_symbol_field(
    field_type: str,
) -> bool:
    """
    True only for BRAIN SYMBOL fields.
    """

    return (
        classify_field_type(
            field_type
        )
        == "SYMBOL"
    )


def is_numeric_field(
    field_type: str,
) -> bool:
    """
    Return whether the field is a numeric MATRIX field.

    MATRIX is the ordinary numeric field type observed in
    fundamental6.

    VECTOR is intentionally excluded because it requires
    a vector operator before being used as a matrix-valued
    expression.

    GROUP / UNIVERSE / SYMBOL are metadata/grouping types and
    are not treated as ordinary numeric fields.
    """

    return is_matrix_field(
        field_type
    )


def requires_vector_operator(
    field_type: str,
) -> bool:
    """
    True when a field is VECTOR and therefore needs an
    appropriate vector operator before normal matrix/time-series
    processing.

    We do not claim a particular reduction operator here.
    Operator compatibility is handled by the operator layer.
    """

    return is_vector_field(
        field_type
    )


def is_group_input(
    field_type: str,
) -> bool:
    """
    True when a field is a BRAIN GROUP field.
    """

    return is_group_field(
        field_type
    )


def is_metadata_field(
    field_type: str,
) -> bool:
    """
    True for SYMBOL or UNIVERSE fields.

    These should not automatically be used in ordinary
    arithmetic/financial templates.
    """

    normalized = classify_field_type(
        field_type
    )

    return normalized in {
        "SYMBOL",
        "UNIVERSE",
    }


# ============================================================
# BUILD CANONICAL FIELD CATALOG
# ============================================================

def build_field_catalog(
    filtered_datafields_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical field catalog used by the research engine.

    IMPORTANT:
        This function preserves the explicit BRAIN field type.

    It does not convert:
        VECTOR -> MATRIX
        GROUP  -> MATRIX
        SYMBOL -> MATRIX
        UNIVERSE -> MATRIX

    Unknown types remain UNKNOWN.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not isinstance(
        filtered_datafields_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "filtered_datafields_df must be a pandas DataFrame."
        )

    if "id" not in (
        filtered_datafields_df.columns
    ):
        raise KeyError(
            "Field catalog must contain an 'id' column."
        )

    result = (
        filtered_datafields_df
        .copy()
    )

    # --------------------------------------------------------
    # Required/common columns
    # --------------------------------------------------------

    result["id"] = (
        result["id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for column in [
        "name",
        "description",
        "type",
    ]:

        if column not in (
            result.columns
        ):

            result[column] = ""

    result["name"] = (
        result["name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["description"] = (
        result["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["type"] = (
        result["type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Canonical BRAIN type
    # --------------------------------------------------------

    result["field_type"] = (
        result["type"]
        .map(
            classify_field_type
        )
    )

    # --------------------------------------------------------
    # Remove empty field IDs
    # --------------------------------------------------------

    result = result[
        result["id"].ne("")
    ].copy()

    # --------------------------------------------------------
    # One row per field ID
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["id"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# FIELD TYPE COUNTS
# ============================================================

def field_type_counts(
    field_catalog_df: pd.DataFrame,
) -> dict:
    """
    Return counts of the explicit BRAIN field types.
    """

    if (
        "field_type"
        not in field_catalog_df.columns
    ):
        raise KeyError(
            "field_catalog_df must contain "
            "'field_type'."
        )

    counts = (
        field_catalog_df[
            "field_type"
        ]
        .value_counts()
        .to_dict()
    )

    # Include zeroes for the known BRAIN taxonomy so the output
    # is stable across datasets.
    return {
        field_type: int(
            counts.get(
                field_type,
                0,
            )
        )
        for field_type in sorted(
            BRAIN_FIELD_TYPES
        )
    } | {
        "UNKNOWN": int(
            counts.get(
                "UNKNOWN",
                0,
            )
        )
    }


# ============================================================
# FILTER BY FIELD TYPE
# ============================================================

def filter_field_types(
    field_catalog_df: pd.DataFrame,
    allowed_types,
) -> pd.DataFrame:
    """
    Return only fields whose canonical BRAIN type is in
    `allowed_types`.
    """

    if (
        "field_type"
        not in field_catalog_df.columns
    ):
        raise KeyError(
            "field_catalog_df must contain "
            "'field_type'."
        )

    allowed = {
        classify_field_type(
            field_type
        )
        for field_type
        in allowed_types
    }

    return (
        field_catalog_df[
            field_catalog_df[
                "field_type"
            ].isin(
                allowed
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# NUMERIC RESEARCH FIELDS
# ============================================================

def numeric_research_fields(
    field_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return fields that can be treated as ordinary numeric
    MATRIX research inputs.

    VECTOR, GROUP, UNIVERSE and SYMBOL fields are excluded.
    """

    return filter_field_types(
        field_catalog_df,
        {"MATRIX"},
    )


# ============================================================
# VECTOR RESEARCH FIELDS
# ============================================================

def vector_research_fields(
    field_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only VECTOR fields.

    These require an appropriate vector operator before they
    can enter ordinary matrix/time-series expressions.
    """

    return filter_field_types(
        field_catalog_df,
        {"VECTOR"},
    )


# ============================================================
# GROUP FIELDS
# ============================================================

def group_fields(
    field_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return GROUP fields.

    These are intended to be supplied to group operators rather
    than ordinary arithmetic/time-series operators.
    """

    return filter_field_types(
        field_catalog_df,
        {"GROUP"},
    )


# ============================================================
# METADATA FIELDS
# ============================================================

def metadata_fields(
    field_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return SYMBOL and UNIVERSE fields.
    """

    return filter_field_types(
        field_catalog_df,
        {
            "SYMBOL",
            "UNIVERSE",
        },
    )


# ============================================================
# FIELD ALIASES
# ============================================================

def make_field_aliases(
    field_catalog_df: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    """
    Create deterministic aliases:

        F1 -> actual BRAIN field ID
        F2 -> actual BRAIN field ID
        ...

    Returns:

        alias_to_id
        alias_to_type

    The alias order is exactly the current dataframe order.

    No alias is generated by an LLM.
    """

    required_columns = {
        "id",
        "field_type",
    }

    missing = (
        required_columns
        - set(
            field_catalog_df.columns
        )
    )

    if missing:

        raise KeyError(
            "Field catalog missing columns: "
            f"{sorted(missing)}"
        )

    alias_to_id = {}
    alias_to_type = {}

    for index, row in (
        field_catalog_df.iterrows()
    ):

        alias = (
            f"F{index + 1}"
        )

        field_id = str(
            row["id"]
        ).strip()

        field_type = (
            classify_field_type(
                row["field_type"]
            )
        )

        if not field_id:

            raise ValueError(
                f"Field at row {index} "
                "has an empty ID."
            )

        alias_to_id[
            alias
        ] = field_id

        alias_to_type[
            alias
        ] = field_type

    return (
        alias_to_id,
        alias_to_type,
    )


# ============================================================
# FIELD DESCRIPTION MAP
# ============================================================

def make_field_descriptions(
    field_catalog_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Create:

        actual_field_id -> description

    This is useful for LLM field-selection prompts while keeping
    the actual ID authoritative.
    """

    if "id" not in (
        field_catalog_df.columns
    ):
        raise KeyError(
            "Field catalog must contain 'id'."
        )

    if "description" not in (
        field_catalog_df.columns
    ):
        raise KeyError(
            "Field catalog must contain "
            "'description'."
        )

    return {
        str(
            row["id"]
        ).strip(): str(
            row["description"]
        ).strip()
        for _, row
        in field_catalog_df.iterrows()
    }