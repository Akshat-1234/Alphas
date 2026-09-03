# ============================================================
# inspect_brain_types.py
# ============================================================

import ace_lib as ace

from engine.brain import BrainContext


# ============================================================
# 1. BRAIN SESSION
# ============================================================

session = ace.start_session()

Region = "GLB"
Univ = "TOPDIV3000"
dataset_id = "fundamental6"

ctx = BrainContext(
    session=session,
    region=Region,
    universe=Univ,
    dataset_id=dataset_id,
    delay=1,
)


# ============================================================
# 2. LOAD LIVE FIELDS
# ============================================================

print("=" * 80)
print("FIELD TYPES")
print("=" * 80)

fields = ctx.get_datafields().copy()

print(
    "Total datafields:",
    len(fields),
)

print()

print(
    "Columns:"
)

print(
    list(fields.columns)
)

print()


# ============================================================
# 3. UNIQUE TYPE VALUES
# ============================================================

if "type" not in fields.columns:

    raise RuntimeError(
        "BRAIN datafield catalog has no 'type' column."
    )

type_counts = (
    fields["type"]
    .fillna("<NULL>")
    .astype(str)
    .str.strip()
    .value_counts()
)

print(
    "Unique field types:"
)

print(
    type_counts.to_string()
)


# ============================================================
# 4. FUNDAMENTAL6 FIELDS
# ============================================================

dataset_fields = (
    fields[
        fields["dataset_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(dataset_id)
    ]
    .copy()
)

print()
print("=" * 80)
print("FUNDAMENTAL6 TYPE DISTRIBUTION")
print("=" * 80)

print(
    "fundamental6 fields:",
    len(dataset_fields),
)

print()

print(
    dataset_fields[
        "type"
    ]
    .fillna("<NULL>")
    .astype(str)
    .str.strip()
    .value_counts()
    .to_string()
)


# ============================================================
# 5. SHOW EXAMPLES OF EACH TYPE
# ============================================================

print()
print("=" * 80)
print("EXAMPLES BY FIELD TYPE")
print("=" * 80)

for field_type in sorted(
    dataset_fields[
        "type"
    ]
    .fillna("<NULL>")
    .astype(str)
    .str.strip()
    .unique()
):

    print()
    print(
        f"TYPE: {field_type}"
    )

    subset = dataset_fields[
        dataset_fields[
            "type"
        ]
        .fillna("<NULL>")
        .astype(str)
        .str.strip()
        .eq(field_type)
    ].copy()

    columns = [
        column
        for column in [
            "id",
            "name",
            "description",
            "type",
            "dataset_id",
            "coverage",
            "alphaCount",
        ]
        if column in subset.columns
    ]

    print(
        subset[
            columns
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# 6. LOAD OPERATORS
# ============================================================

print()
print("=" * 80)
print("OPERATOR CATALOG")
print("=" * 80)

operators = (
    ctx.get_operators()
    .copy()
)

print(
    "Total operators:",
    len(operators),
)

print()

print(
    "Operator columns:"
)

print(
    list(operators.columns)
)


# ============================================================
# 7. PRINT OPERATORS RELEVANT TO FIELD TYPES
# ============================================================

keywords = [
    "vector",
    "matrix",
    "group",
    "universe",
    "symbol",
    "vec_",
    "group_",
]

print()
print("=" * 80)
print("TYPE-RELATED OPERATORS")
print("=" * 80)

operator_text = (
    operators
    .fillna("")
    .astype(str)
    .agg(
        " | ".join,
        axis=1,
    )
)

mask = operator_text.str.lower().apply(
    lambda text: any(
        keyword in text
        for keyword in keywords
    )
)

relevant = operators[
    mask
].copy()

if relevant.empty:

    print(
        "No operator definition/metadata "
        "contains the requested type keywords."
    )

else:

    for _, row in relevant.iterrows():

        print(
            "\nNAME:",
            row.get(
                "name",
                "",
            ),
        )

        print(
            "DEFINITION:",
            row.get(
                "definition",
                "",
            ),
        )

        for column in [
            "category",
            "description",
            "type",
        ]:

            if column in operators.columns:

                print(
                    f"{column.upper()}:",
                    row.get(
                        column,
                        "",
                    ),
                )


# ============================================================
# 8. EXACT SAMPLE FIELDS
# ============================================================

print()
print("=" * 80)
print("FUNDAMENTAL6 SAMPLE FIELD METADATA")
print("=" * 80)

sample_ids = [
    "fnd6_int_accdq",
    "fnd6_int_eqrtq",
    "fnd6_newqint_revtq",
]

sample = dataset_fields[
    dataset_fields[
        "id"
    ].isin(
        sample_ids
    )
].copy()

if sample.empty:

    print(
        "Requested sample IDs were not found."
    )

else:

    print(
        sample.to_string(
            index=False
        )
    )


# ============================================================
# 9. COMPLETE
# ============================================================

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)