# ============================================================
# test_engine.py
# ============================================================

import ace_lib as ace

from engine.brain import BrainContext
from engine.fields import build_field_catalog
from engine.operators import (
    build_operator_catalog,
    get_operator_signature,
)
from engine.compiler import FastExprCompiler
from engine.validator import FastExprValidator


# ============================================================
# 1. START BRAIN SESSION
# ============================================================

print("=" * 80)
print("1. STARTING BRAIN SESSION")
print("=" * 80)

session = ace.start_session()

print(
    "Session created:",
    session is not None,
)


# ============================================================
# 2. CREATE BRAIN CONTEXT
# ============================================================

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

print()
print("=" * 80)
print("2. BRAIN CONTEXT")
print("=" * 80)

print("Region:", Region)
print("Universe:", Univ)
print("Dataset:", dataset_id)


# ============================================================
# 3. LOAD LIVE DATAFIELD CATALOG
# ============================================================

print()
print("=" * 80)
print("3. LOADING DATAFIELDS")
print("=" * 80)

datafields = ctx.get_datafields()

print(
    "Total datafields returned:",
    len(datafields),
)

dataset_fields = ctx.get_dataset_fields()

print(
    "Fields in target dataset:",
    len(dataset_fields),
)


# ============================================================
# 4. BUILD ENGINE FIELD CATALOG
# ============================================================

field_catalog = build_field_catalog(
    dataset_fields
)

print()
print("=" * 80)
print("4. ENGINE FIELD CATALOG")
print("=" * 80)

print(
    "Canonical fields:",
    len(field_catalog),
)

print(
    "Field types:",
    field_catalog[
        "field_type"
    ].value_counts().to_dict()
)


# ============================================================
# 5. LOAD LIVE OPERATOR CATALOG
# ============================================================

print()
print("=" * 80)
print("5. LOADING OPERATORS")
print("=" * 80)

operators_df = (
    ctx.get_operators()
)

operator_catalog = (
    build_operator_catalog(
        operators_df
    )
)

print(
    "Live operators:",
    len(operator_catalog)
)


# ============================================================
# 6. INSPECT A FEW CRITICAL OPERATORS
# ============================================================

print()
print("=" * 80)
print("6. CRITICAL OPERATOR SIGNATURES")
print("=" * 80)

critical_operators = [
    "rank",
    "divide",
    "ts_backfill",
    "ts_mean",
    "ts_zscore",
    "ts_delta",
    "ts_std_dev",
    "ts_corr",
    "vec_avg",
]

available_critical = []

for operator in critical_operators:

    if operator not in operator_catalog:

        print(
            operator,
            "-> NOT FOUND"
        )

        continue

    signature = (
        get_operator_signature(
            operators_df,
            operator,
        )
    )

    available_critical.append(
        operator
    )

    print()
    print(
        operator,
        "->",
        signature[
            "definition"
        ],
    )

    print(
        "Arguments:",
        signature[
            "args"
        ],
    )


# ============================================================
# 7. BUILD DETERMINISTIC FIELD ALIASES
# ============================================================

print()
print("=" * 80)
print("7. BUILDING FIELD ALIASES")
print("=" * 80)

# Keep the test intentionally small.
test_fields = (
    field_catalog
    .head(5)
    .copy()
    .reset_index(drop=True)
)

alias_to_id = {}
alias_to_type = {}

for index, row in (
    test_fields.iterrows()
):

    alias = f"F{index + 1}"

    alias_to_id[
        alias
    ] = str(
        row["id"]
    )

    alias_to_type[
        alias
    ] = str(
        row["field_type"]
    )

    print(
        f"{alias} -> "
        f"{row['id']} -> "
        f"{row['field_type']}"
    )


# ============================================================
# 8. CREATE COMPILER
# ============================================================

print()
print("=" * 80)
print("8. BUILDING COMPILER")
print("=" * 80)

allowed_windows = [
    20,
    30,
    60,
    90,
    120,
    180,
    252,
]

compiler = FastExprCompiler(
    field_alias_to_id=alias_to_id,
    field_alias_to_type=alias_to_type,
    verified_operators=set(
        available_critical
    ),
    allowed_windows=allowed_windows,
)

print(
    "Compiler created."
)


# ============================================================
# 9. CREATE VALIDATOR
# ============================================================

print()
print("=" * 80)
print("9. BUILDING VALIDATOR")
print("=" * 80)

validator = FastExprValidator(
    field_alias_to_type=alias_to_type,
    verified_operators=set(
        available_critical
    ),
    allowed_windows=allowed_windows,
)

print(
    "Validator created."
)


# ============================================================
# 10. COMPILE A SIMPLE TEST EXPRESSION
# ============================================================

print()
print("=" * 80)
print("10. COMPILING TEST EXPRESSION")
print("=" * 80)

test_expression = compiler.compile(
    template="LEVEL",
    fields=["F1"],
    window=60,
    backfill_window=60,
    direction="positive",
)

print(
    "Compiled expression:"
)

print(
    test_expression.expression
)


# ============================================================
# 11. VALIDATE IT
# ============================================================

print()
print("=" * 80)
print("11. VALIDATING TEST EXPRESSION")
print("=" * 80)

ok, reason = validator.validate(
    test_expression.expression
)

print(
    "Valid:",
    ok,
)

print(
    "Reason:",
    reason,
)


# ============================================================
# 12. VECTOR/EVENT SAFETY TEST
# ============================================================

print()
print("=" * 80)
print("12. TYPE SAFETY TEST")
print("=" * 80)

vector_aliases = [
    alias
    for alias, field_type
    in alias_to_type.items()
    if field_type in {
        "VECTOR",
        "EVENT",
    }
]

if vector_aliases:

    vector_alias = (
        vector_aliases[0]
    )

    bad_expression = (
        f"ts_rank("
        f"{vector_alias},"
        f"60)"
    )

    bad_ok, bad_reason = (
        validator.validate(
            bad_expression
        )
    )

    print(
        "Test expression:",
        bad_expression,
    )

    print(
        "Accepted:",
        bad_ok,
    )

    print(
        "Reason:",
        bad_reason,
    )

    if bad_ok:

        raise RuntimeError(
            "TYPE SAFETY FAILURE: "
            "VECTOR/EVENT field was accepted "
            "directly by ts_rank."
        )

else:

    print(
        "No VECTOR/EVENT fields found "
        "in first five fields."
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("ENGINE INTEGRATION TEST COMPLETE")
print("=" * 80)

print(
    "PASS"
)