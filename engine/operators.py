# ============================================================
# engine/operators.py
# ============================================================

import re

import pandas as pd


# ============================================================
# LIVE OPERATOR CATALOG
# ============================================================

def build_operator_catalog(
    operators_df: pd.DataFrame,
) -> dict:
    """
    Build a canonical operator catalog from the live BRAIN
    operator dataframe.

    No operator definitions are invented here.
    """

    if not isinstance(
        operators_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "operators_df must be a pandas DataFrame."
        )

    required_columns = {
        "name",
        "definition",
    }

    missing = (
        required_columns
        - set(
            operators_df.columns
        )
    )

    if missing:
        raise KeyError(
            "Operator dataframe is missing "
            f"columns: {sorted(missing)}"
        )

    catalog = {}

    for _, row in operators_df.iterrows():

        name = str(
            row["name"]
        ).strip()

        if not name:
            continue

        catalog[name] = {
            "name": name,
            "definition": str(
                row["definition"]
            ).strip(),
            "category": str(
                row.get(
                    "category",
                    "",
                )
            ).strip(),
            "scope": str(
                row.get(
                    "scope",
                    "",
                )
            ).strip(),
            "description": str(
                row.get(
                    "description",
                    "",
                )
            ).strip(),
            "documentation": str(
                row.get(
                    "documentation",
                    "",
                )
            ).strip(),
            "level": str(
                row.get(
                    "level",
                    "",
                )
            ).strip(),
        }

    return catalog


# ============================================================
# SIGNATURE ARGUMENT SPLITTER
# ============================================================

def split_signature_args(
    text: str,
) -> list[str]:
    """
    Split an operator argument list.

    Example:

        x, lookback=d, k=1

    becomes:

        [
            "x",
            "lookback=d",
            "k=1",
        ]
    """

    arguments = []

    current = []

    depth = 0

    for char in text:

        if char == "(":
            depth += 1

        elif char == ")":
            depth -= 1

        if (
            char == ","
            and depth == 0
        ):

            argument = (
                "".join(
                    current
                ).strip()
            )

            if argument:
                arguments.append(
                    argument
                )

            current = []

        else:

            current.append(
                char
            )

    argument = (
        "".join(
            current
        ).strip()
    )

    if argument:
        arguments.append(
            argument
        )

    return arguments


# ============================================================
# PARSE OPERATOR SIGNATURE
# ============================================================

def parse_operator_signature(
    definition: str,
) -> dict:
    """
    Parse the first function signature in a live BRAIN
    operator definition.

    Example:

        rank(x, rate=2)

    becomes:

        args     = ["x", "rate=2"]
        required = 1
        maximum  = 2
    """

    definition = str(
        definition
    ).strip()

    if not definition:

        return {
            "args": [],
            "required": 0,
            "maximum": 0,
            "definition": definition,
        }

    open_index = (
        definition.find("(")
    )

    if open_index < 0:

        return {
            "args": [],
            "required": 0,
            "maximum": 0,
            "definition": definition,
        }

    depth = 0

    close_index = None

    for index in range(
        open_index,
        len(definition),
    ):

        char = definition[index]

        if char == "(":

            depth += 1

        elif char == ")":

            depth -= 1

            if depth == 0:

                close_index = index

                break

    if close_index is None:

        return {
            "args": [],
            "required": 0,
            "maximum": 0,
            "definition": definition,
        }

    inside = (
        definition[
            open_index + 1:
            close_index
        ].strip()
    )

    args = (
        split_signature_args(
            inside
        )
        if inside
        else []
    )

    required = sum(
        "=" not in argument
        for argument in args
    )

    return {
        "args": args,
        "required": required,
        "maximum": len(args),
        "definition": definition,
    }


# ============================================================
# NORMALIZE ARGUMENT NAME
# ============================================================

def normalize_argument_name(
    argument: str,
) -> str:
    """
    Convert an argument declaration into its bare name.

    Examples:

        rate=2          -> rate
        lookback = d    -> lookback
        x               -> x
    """

    argument = str(
        argument
    ).strip()

    argument = (
        argument
        .split(
            "=",
            1,
        )[0]
        .strip()
        .lower()
    )

    return re.sub(
        r"[^a-z0-9_]",
        "",
        argument,
    )


# ============================================================
# ARGUMENT NAMES
# ============================================================

def operator_argument_names(
    signature: dict,
) -> list[str]:
    """
    Return normalized operator argument names.
    """

    return [
        normalize_argument_name(
            argument
        )
        for argument
        in signature.get(
            "args",
            [],
        )
    ]


# ============================================================
# GET ONE OPERATOR SIGNATURE
# ============================================================

def get_operator_signature(
    operators_df: pd.DataFrame,
    operator_name: str,
) -> dict:
    """
    Retrieve and parse one operator from the live BRAIN
    operator catalog.
    """

    catalog = (
        build_operator_catalog(
            operators_df
        )
    )

    operator_name = str(
        operator_name
    ).strip()

    if (
        operator_name
        not in catalog
    ):

        raise KeyError(
            "Unknown BRAIN operator: "
            f"{operator_name}"
        )

    definition = catalog[
        operator_name
    ]["definition"]

    signature = (
        parse_operator_signature(
            definition
        )
    )

    signature["name"] = (
        operator_name
    )

    signature["category"] = (
        catalog[
            operator_name
        ]["category"]
    )

    signature["scope"] = (
        catalog[
            operator_name
        ]["scope"]
    )

    signature["description"] = (
        catalog[
            operator_name
        ]["description"]
    )

    signature["documentation"] = (
        catalog[
            operator_name
        ]["documentation"]
    )

    signature["level"] = (
        catalog[
            operator_name
        ]["level"]
    )

    return signature


# ============================================================
# BUILD FULL SIGNATURE CATALOG
# ============================================================

def build_signature_catalog(
    operators_df: pd.DataFrame,
) -> dict:
    """
    Parse every operator definition returned by BRAIN.
    """

    catalog = (
        build_operator_catalog(
            operators_df
        )
    )

    signatures = {}

    for name, info in (
        catalog.items()
    ):

        signature = (
            parse_operator_signature(
                info["definition"]
            )
        )

        signature["name"] = name

        signature["category"] = (
            info["category"]
        )

        signature["scope"] = (
            info["scope"]
        )

        signature["description"] = (
            info["description"]
        )

        signature["documentation"] = (
            info["documentation"]
        )

        signature["level"] = (
            info["level"]
        )

        signatures[name] = (
            signature
        )

    return signatures


# ============================================================
# COMPILER OPERATOR REQUIREMENTS
# ============================================================
#
# These rules describe ONLY the subset of each LIVE BRAIN
# operator that our compiler intends to use.
#
# They are not replacements for the live definitions.
# ============================================================

COMPILER_OPERATOR_REQUIREMENTS = {

    "rank": {
        "min_args": 1,
        "max_args": 2,
        "positions": [
            {"x", "value"},
            {"rate", "constant"},
        ],
    },

    "reverse": {
        "min_args": 1,
        "max_args": 1,
        "positions": [
            {"x", "value"},
        ],
    },

    "add": {
        "min_args": 2,
        "max_args": 3,
        "positions": [
            {"x", "a"},
            {"y", "b"},
            {"filter"},
        ],
    },

    "subtract": {
        "min_args": 2,
        "max_args": 3,
        "positions": [
            {"x", "a"},
            {"y", "b"},
            {"filter"},
        ],
    },

    "divide": {
        "min_args": 2,
        "max_args": 2,
        "positions": [
            {"x", "a"},
            {"y", "b"},
        ],
    },

    "ts_backfill": {
        "min_args": 2,
        "max_args": 3,
        "positions": [
            {"x", "a", "value"},
            {"lookback", "d", "window"},
            {"k"},
        ],
    },

    "ts_mean": {
        "min_args": 2,
        "max_args": 2,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
        ],
    },

    "ts_rank": {
        "min_args": 2,
        "max_args": 3,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
            {"constant"},
        ],
    },

    "ts_zscore": {
        "min_args": 2,
        "max_args": 2,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
        ],
    },

    "ts_delta": {
        "min_args": 2,
        "max_args": 2,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
        ],
    },

    "ts_std_dev": {
        "min_args": 2,
        "max_args": 2,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
        ],
    },

    "ts_corr": {
        "min_args": 3,
        "max_args": 3,
        "positions": [
            {"x", "a"},
            {"y", "b"},
            {"d", "lookback", "window"},
        ],
    },

    "ts_decay_linear": {
        "min_args": 2,
        "max_args": 3,
        "positions": [
            {"x", "a", "value"},
            {"d", "lookback", "window"},
            {"dense"},
        ],
    },

    "vec_avg": {
        "min_args": 1,
        "max_args": 1,
        "positions": [
            {"x", "value"},
        ],
    },
}


# ============================================================
# VERIFY ONE OPERATOR
# ============================================================

def verify_operator_for_compiler(
    operators_df: pd.DataFrame,
    operator_name: str,
) -> tuple[
    bool,
    str,
]:
    """
    Check whether the LIVE definition of one operator matches
    the argument structure expected by our compiler.
    """

    if (
        operator_name
        not in COMPILER_OPERATOR_REQUIREMENTS
    ):

        return (
            False,
            "no_compiler_rule",
        )

    try:

        signature = (
            get_operator_signature(
                operators_df,
                operator_name,
            )
        )

    except Exception as exc:

        return (
            False,
            f"operator_lookup_error:{exc}",
        )

    requirement = (
        COMPILER_OPERATOR_REQUIREMENTS[
            operator_name
        ]
    )

    actual_args = (
        operator_argument_names(
            signature
        )
    )

    actual_required = (
        signature["required"]
    )

    actual_maximum = (
        signature["maximum"]
    )

    expected_min = (
        requirement["min_args"]
    )

    expected_max = (
        requirement["max_args"]
    )

    # --------------------------------------------------------
    # The live operator must require no more arguments than
    # the compiler's minimum usage.
    # --------------------------------------------------------

    if (
        actual_required
        > expected_min
    ):

        return (
            False,
            (
                "live_required_args_too_large:"
                f"{actual_required}"
            ),
        )

    # --------------------------------------------------------
    # The live operator must accept at least the minimum number
    # of arguments used by the compiler.
    # --------------------------------------------------------

    if (
        actual_maximum
        < expected_min
    ):

        return (
            False,
            (
                "live_max_args_too_small:"
                f"{actual_maximum}"
            ),
        )

    # --------------------------------------------------------
    # Check every argument position that exists in the live
    # signature.
    # --------------------------------------------------------

    expected_positions = (
        requirement["positions"]
    )

    for index, actual_name in enumerate(
        actual_args
    ):

        if index >= len(
            expected_positions
        ):

            return (
                False,
                (
                    "unexpected_argument_position:"
                    f"{index}"
                ),
            )

        accepted_names = (
            expected_positions[
                index
            ]
        )

        if (
            actual_name
            not in accepted_names
        ):

            return (
                False,
                (
                    f"argument_{index}_name_mismatch:"
                    f"{actual_name}"
                ),
            )

    return (
        True,
        "ok",
    )


# ============================================================
# VERIFY ALL COMPILER OPERATORS
# ============================================================

def verify_compiler_operators(
    operators_df: pd.DataFrame,
) -> tuple[
    dict,
    dict,
]:
    """
    Verify all operators required by the compiler.

    Returns:

        verified
        rejected

    The actual BRAIN definitions are retained in `verified`.
    """

    verified = {}

    rejected = {}

    for operator_name in (
        COMPILER_OPERATOR_REQUIREMENTS
    ):

        ok, reason = (
            verify_operator_for_compiler(
                operators_df,
                operator_name,
            )
        )

        if not ok:

            rejected[
                operator_name
            ] = reason

            continue

        verified[
            operator_name
        ] = get_operator_signature(
            operators_df,
            operator_name,
        )

    return (
        verified,
        rejected,
    )


# ============================================================
# PRINT COMPILER OPERATOR REPORT
# ============================================================

def print_compiler_operator_report(
    operators_df: pd.DataFrame,
) -> None:
    """
    Print a human-readable report of which live BRAIN operators
    are compatible with the compiler.
    """

    verified, rejected = (
        verify_compiler_operators(
            operators_df
        )
    )

    print(
        "=" * 80
    )

    print(
        "BRAIN COMPILER OPERATOR REPORT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "VERIFIED:"
    )

    if not verified:

        print(
            "  None"
        )

    else:

        for name in sorted(
            verified
        ):

            print(
                f"  {name:18s} "
                f"-> "
                f"{verified[name]['definition']}"
            )

    print()

    print(
        "REJECTED:"
    )

    if not rejected:

        print(
            "  None"
        )

    else:

        for name in sorted(
            rejected
        ):

            print(
                f"  {name:18s} "
                f"-> "
                f"{rejected[name]}"
            )

    print()

    print(
        "Verified:",
        len(verified),
    )

    print(
        "Rejected:",
        len(rejected),
    )