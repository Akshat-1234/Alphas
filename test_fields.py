# ============================================================
# test_fields.py
# ============================================================

import ace_lib as ace

from engine.brain import BrainContext

from engine.fields import (
    build_field_catalog,
    field_type_counts,
    numeric_research_fields,
    vector_research_fields,
    group_fields,
    metadata_fields,
)


# ============================================================
# BRAIN CONFIG
# ============================================================

Region = "GLB"
Univ = "TOPDIV3000"
dataset_id = "fundamental6"


# ============================================================
# SESSION
# ============================================================

print("=" * 80)
print("1. START SESSION")
print("=" * 80)

session = ace.start_session()

print(
    "Session created:",
    session is not None,
)


# ============================================================
# CONTEXT
# ============================================================

ctx = BrainContext(
    session=session,
    region=Region,
    universe=Univ,
    dataset_id=dataset_id,
    delay=1,
)


# ============================================================
# LOAD DATAFIELDS
# ============================================================

print()
print("=" * 80)
print("2. LOAD FUNDAMENTAL6")
print("=" * 80)

raw_fields = ctx.get_dataset_fields()

print(
    "Raw fundamental6 fields:",
    len(raw_fields),
)


# ============================================================
# BUILD ENGINE CATALOG
# ============================================================

print()
print("=" * 80)
print("3. BUILD ENGINE FIELD CATALOG")
print("=" * 80)

catalog = build_field_catalog(
    raw_fields
)

print(
    "Canonical fields:",
    len(catalog),
)


# ============================================================
# TYPE COUNTS
# ============================================================

print()
print("=" * 80)
print("4. FIELD TYPE COUNTS")
print("=" * 80)

counts = field_type_counts(
    catalog
)

print(
    counts
)


# ============================================================
# EXPECT FUNDAMENTAL6 TO BE MATRIX
# ============================================================

expected_matrix_count = len(
    catalog
)

actual_matrix_count = counts.get(
    "MATRIX",
    0,
)

if actual_matrix_count != expected_matrix_count:

    raise RuntimeError(
        "Unexpected fundamental6 type distribution.\n"
        f"Expected MATRIX={expected_matrix_count}\n"
        f"Actual MATRIX={actual_matrix_count}\n"
        f"Counts={counts}"
    )


if counts.get(
    "VECTOR",
    0,
) != 0:

    raise RuntimeError(
        "fundamental6 unexpectedly contains VECTOR fields."
    )


if counts.get(
    "GROUP",
    0,
) != 0:

    raise RuntimeError(
        "fundamental6 unexpectedly contains GROUP fields."
    )


if counts.get(
    "UNIVERSE",
    0,
) != 0:

    raise RuntimeError(
        "fundamental6 unexpectedly contains UNIVERSE fields."
    )


if counts.get(
    "SYMBOL",
    0,
) != 0:

    raise RuntimeError(
        "fundamental6 unexpectedly contains SYMBOL fields."
    )


# ============================================================
# TYPE-SPECIFIC FILTERS
# ============================================================

print()
print("=" * 80)
print("5. TYPE FILTERS")
print("=" * 80)

matrix_fields = numeric_research_fields(
    catalog
)

vector_fields = vector_research_fields(
    catalog
)

grouped_fields = group_fields(
    catalog
)

meta_fields = metadata_fields(
    catalog
)

print(
    "MATRIX research fields:",
    len(matrix_fields),
)

print(
    "VECTOR fields:",
    len(vector_fields),
)

print(
    "GROUP fields:",
    len(grouped_fields),
)

print(
    "SYMBOL + UNIVERSE fields:",
    len(meta_fields),
)


# ============================================================
# FILTER CONSISTENCY
# ============================================================

if len(matrix_fields) != len(
    catalog
):

    raise RuntimeError(
        "numeric_research_fields() "
        "did not return all MATRIX fields."
    )


if not vector_fields.empty:

    raise RuntimeError(
        "Unexpected VECTOR fields."
    )


if not grouped_fields.empty:

    raise RuntimeError(
        "Unexpected GROUP fields."
    )


if not meta_fields.empty:

    raise RuntimeError(
        "Unexpected SYMBOL/UNIVERSE fields."
    )


# ============================================================
# SHOW SAMPLE FIELDS
# ============================================================

print()
print("=" * 80)
print("6. SAMPLE MATRIX FIELDS")
print("=" * 80)

sample_columns = [
    column
    for column in [
        "id",
        "description",
        "type",
        "field_type",
    ]
    if column in catalog.columns
]

print(
    matrix_fields[
        sample_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("FIELD ENGINE TEST: PASS")
print("=" * 80)