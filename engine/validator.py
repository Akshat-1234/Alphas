# ============================================================
# engine/validator.py
# ============================================================

import re


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_RE = re.compile(
    r"\s*("
    r"\d+(?:\.\d+)?"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|[(),]"
    r")"
)

FIELD_ALIAS_RE = re.compile(
    r"^F[1-9][0-9]*$"
)


# ============================================================
# VALIDATOR
# ============================================================

class FastExprValidator:
    """
    Structural validator for FASTEXPR expressions produced by
    the Python-owned compiler.

    This validator is intentionally conservative.

    It checks:

        - allowed operators
        - balanced function syntax
        - field aliases
        - operator arity
        - time-series window positions
        - VECTOR/EVENT safety
        - expression depth
        - expression length

    It does NOT claim to replace BRAIN's own server-side
    validation. BRAIN remains the final authority.
    """

    # --------------------------------------------------------
    # Expected arity for the operators that the compiler uses.
    #
    # Values are:
    #
    #     (minimum_arguments, maximum_arguments)
    #
    # --------------------------------------------------------

    OPERATOR_ARITY = {
        "rank": (1, 2),
        "reverse": (1, 1),

        "add": (2, 2),
        "subtract": (2, 2),
        "divide": (2, 2),

        "ts_backfill": (2, 3),
        "ts_mean": (2, 2),
        "ts_rank": (2, 3),
        "ts_zscore": (2, 2),
        "ts_delta": (2, 2),
        "ts_std_dev": (2, 2),
        "ts_decay_linear": (2, 3),

        "ts_corr": (3, 3),

        "vec_avg": (1, 1),
    }

    # --------------------------------------------------------
    # Position of the time-series window argument.
    #
    # Example:
    #
    #     ts_zscore(x, 60)
    #
    # `60` is argument position 1.
    #
    # For:
    #
    #     ts_corr(x, y, 60)
    #
    # `60` is argument position 2.
    # --------------------------------------------------------

    WINDOW_POSITION = {
        "ts_backfill": 1,
        "ts_mean": 1,
        "ts_rank": 1,
        "ts_zscore": 1,
        "ts_delta": 1,
        "ts_std_dev": 1,
        "ts_decay_linear": 1,
        "ts_corr": 2,
    }

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        field_alias_to_type,
        verified_operators,
        allowed_windows,
        max_length=700,
        max_depth=9,
    ):
        """
        Parameters
        ----------
        field_alias_to_type:
            Example:

                {
                    "F1": "SCALAR",
                    "F2": "VECTOR",
                    "F3": "EVENT",
                }

        verified_operators:
            Operators confirmed against the live BRAIN
            operator catalog.

        allowed_windows:
            Windows accepted by the research pipeline.

        max_length:
            Maximum FASTEXPR string length checked locally.

        max_depth:
            Maximum expression tree depth.
        """

        self.field_alias_to_type = dict(
            field_alias_to_type
        )

        self.verified_operators = set(
            verified_operators
        )

        self.allowed_windows = {
            int(value)
            for value in allowed_windows
        }

        self.max_length = int(
            max_length
        )

        self.max_depth = int(
            max_depth
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def tokenize(
        self,
        expression,
    ):
        """
        Convert an expression string into simple tokens.

        Supported tokens are:

            function/operator names
            field aliases
            numeric constants
            '('
            ')'
            ','
        """

        expression = str(
            expression
        )

        tokens = []

        position = 0

        while position < len(
            expression
        ):

            match = TOKEN_RE.match(
                expression,
                position,
            )

            if not match:

                raise ValueError(
                    "Invalid token near: "
                    f"{expression[position:position + 50]!r}"
                )

            tokens.append(
                match.group(1)
            )

            position = (
                match.end()
            )

        return tokens

    # ========================================================
    # RECURSIVE PARSER
    # ========================================================

    def parse_node(
        self,
        tokens,
        index=0,
    ):
        """
        Parse one expression node.

        Returns:

            (node, next_index)

        A node is one of:

            constant
            field
            call
        """

        if index >= len(
            tokens
        ):

            raise ValueError(
                "Unexpected end of expression."
            )

        token = tokens[
            index
        ]

        # ----------------------------------------------------
        # Numeric constant
        # ----------------------------------------------------

        if re.fullmatch(
            r"\d+(?:\.\d+)?",
            token,
        ):

            return (
                {
                    "kind": "constant",
                    "value": float(
                        token
                    ),
                },
                index + 1,
            )

        # ----------------------------------------------------
        # Field alias
        # ----------------------------------------------------

        if FIELD_ALIAS_RE.fullmatch(
            token
        ):

            if (
                token
                not in self.field_alias_to_type
            ):

                raise ValueError(
                    f"Unknown field alias: {token}"
                )

            return (
                {
                    "kind": "field",
                    "alias": token,
                    "field_type": (
                        self.field_alias_to_type[
                            token
                        ]
                    ),
                },
                index + 1,
            )

        # ----------------------------------------------------
        # Function/operator
        # ----------------------------------------------------

        if (
            token
            not in self.verified_operators
        ):

            raise ValueError(
                f"Unverified operator: {token}"
            )

        # ----------------------------------------------------
        # Function must be followed by '('
        # ----------------------------------------------------

        if (
            index + 1 >= len(
                tokens
            )
            or tokens[
                index + 1
            ] != "("
        ):

            raise ValueError(
                f"Expected '(' after {token}"
            )

        args = []

        index += 2

        # ----------------------------------------------------
        # Parse argument list
        # ----------------------------------------------------

        while True:

            if index >= len(
                tokens
            ):

                raise ValueError(
                    f"Unclosed {token}("
                )

            # Empty argument list.
            if tokens[
                index
            ] == ")":

                return (
                    {
                        "kind": "call",
                        "op": token,
                        "args": args,
                    },
                    index + 1,
                )

            child, index = (
                self.parse_node(
                    tokens,
                    index,
                )
            )

            args.append(
                child
            )

            if index >= len(
                tokens
            ):

                raise ValueError(
                    f"Unclosed {token}("
                )

            # Another argument.
            if tokens[
                index
            ] == ",":

                index += 1
                continue

            # End of argument list.
            if tokens[
                index
            ] == ")":

                return (
                    {
                        "kind": "call",
                        "op": token,
                        "args": args,
                    },
                    index + 1,
                )

            raise ValueError(
                f"Expected ',' or ')' inside "
                f"{token}(...)"
            )

    # ========================================================
    # TREE DEPTH
    # ========================================================

    def tree_depth(
        self,
        node,
    ):
        """
        Return recursive expression-tree depth.
        """

        if node[
            "kind"
        ] != "call":

            return 1

        child_depths = [
            self.tree_depth(
                child
            )
            for child in node[
                "args"
            ]
        ]

        return (
            1
            + max(
                child_depths
                or [0]
            )
        )

    # ========================================================
    # TREE VALIDATION
    # ========================================================

    def validate_tree(
        self,
        node,
    ):
        """
        Validate a parsed expression tree.
        """

        errors = []

        def walk(
            current,
        ):

            # ------------------------------------------------
            # Leaf node
            # ------------------------------------------------

            if (
                current["kind"]
                != "call"
            ):

                return

            operator = current[
                "op"
            ]

            args = current[
                "args"
            ]

            # ------------------------------------------------
            # Operator arity
            # ------------------------------------------------

            if (
                operator
                not in self.OPERATOR_ARITY
            ):

                errors.append(
                    f"{operator}: "
                    "no local arity rule."
                )

            else:

                minimum, maximum = (
                    self.OPERATOR_ARITY[
                        operator
                    ]
                )

                if len(args) < minimum:

                    errors.append(
                        f"{operator}: "
                        f"too few arguments "
                        f"({len(args)} < {minimum})."
                    )

                if len(args) > maximum:

                    errors.append(
                        f"{operator}: "
                        f"too many arguments "
                        f"({len(args)} > {maximum})."
                    )

            # ------------------------------------------------
            # VECTOR / EVENT safety
            # ------------------------------------------------

            for child in args:

                if (
                    child["kind"]
                    == "field"
                    and child[
                        "field_type"
                    ]
                    in {
                        "VECTOR",
                        "EVENT",
                    }
                    and operator
                    != "vec_avg"
                ):

                    errors.append(
                        f"{child['alias']} is "
                        f"{child['field_type']} "
                        "and must first be "
                        "reduced by vec_avg()."
                    )

            # ------------------------------------------------
            # vec_avg safety
            # ------------------------------------------------

            if (
                operator
                == "vec_avg"
                and len(args)
                == 1
            ):

                child = args[
                    0
                ]

                if not (
                    child["kind"]
                    == "field"
                    and child[
                        "field_type"
                    ]
                    in {
                        "VECTOR",
                        "EVENT",
                    }
                ):

                    errors.append(
                        "vec_avg requires "
                        "a VECTOR/EVENT field."
                    )

            # ------------------------------------------------
            # Time-series windows
            # ------------------------------------------------

            if (
                operator
                in self.WINDOW_POSITION
            ):

                window_position = (
                    self.WINDOW_POSITION[
                        operator
                    ]
                )

                if (
                    len(args)
                    <= window_position
                ):

                    errors.append(
                        f"{operator}: "
                        "missing window."
                    )

                else:

                    window = args[
                        window_position
                    ]

                    # Window must be a literal numeric
                    # constant in our compiler subset.
                    if (
                        window["kind"]
                        != "constant"
                    ):

                        errors.append(
                            f"{operator}: "
                            "window must be numeric."
                        )

                    else:

                        value = int(
                            window["value"]
                        )

                        if (
                            value
                            not in self.allowed_windows
                        ):

                            errors.append(
                                f"{operator}: "
                                f"unsupported window "
                                f"{value}."
                            )

            # ------------------------------------------------
            # Recurse
            # ------------------------------------------------

            for child in args:

                walk(
                    child
                )

        walk(
            node
        )

        return errors

    # ========================================================
    # COMPLETE VALIDATION
    # ========================================================

    def validate(
        self,
        expression,
    ):
        """
        Validate a complete FASTEXPR.

        Returns:

            (True, "ok")

        or:

            (False, "reason")
        """

        # ----------------------------------------------------
        # Type check
        # ----------------------------------------------------

        if not isinstance(
            expression,
            str,
        ):

            return (
                False,
                "not_string",
            )

        expression = expression.strip()

        # ----------------------------------------------------
        # Empty expression
        # ----------------------------------------------------

        if not expression:

            return (
                False,
                "empty",
            )

        # ----------------------------------------------------
        # Length
        # ----------------------------------------------------

        if (
            len(expression)
            > self.max_length
        ):

            return (
                False,
                "too_long",
            )

        # ----------------------------------------------------
        # Forbidden formatting
        # ----------------------------------------------------

        if (
            "\n" in expression
            or "\r" in expression
        ):

            return (
                False,
                "newline",
            )

        if (
            "`" in expression
            or "```" in expression
        ):

            return (
                False,
                "markdown",
            )

        # ----------------------------------------------------
        # Tokenize + parse
        # ----------------------------------------------------

        try:

            tokens = self.tokenize(
                expression
            )

            tree, position = (
                self.parse_node(
                    tokens,
                    0,
                )
            )

            if (
                position
                != len(tokens)
            ):

                return (
                    False,
                    "trailing_tokens",
                )

        except Exception as exc:

            return (
                False,
                f"parse_error:{exc}",
            )

        # ----------------------------------------------------
        # Semantic checks
        # ----------------------------------------------------

        errors = (
            self.validate_tree(
                tree
            )
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        if (
            self.tree_depth(tree)
            > self.max_depth
        ):

            errors.append(
                "expression_too_deep"
            )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        if errors:

            return (
                False,
                "; ".join(
                    errors
                ),
            )

        return (
            True,
            "ok",
        )