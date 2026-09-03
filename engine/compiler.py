# ============================================================
# engine/compiler.py
# ============================================================

from dataclasses import dataclass


# ============================================================
# COMPILE RESULT
# ============================================================

@dataclass
class CompileResult:
    """
    Result of compiling a research template into FASTEXPR.
    """

    expression: str
    template: str
    fields: list[str]
    window: int
    backfill_window: int
    direction: str


# ============================================================
# FASTEXPR COMPILER
# ============================================================

class FastExprCompiler:
    """
    Python-owned FASTEXPR compiler.

    The LLM does NOT construct FASTEXPR syntax.

    The caller supplies:

        template
        field aliases
        window
        backfill window
        direction

    Python constructs the actual expression.

    This prevents the LLM from inventing:

        - field names
        - operator names
        - argument order
        - parentheses
        - arbitrary nesting
    """

    # --------------------------------------------------------
    # Supported templates
    # --------------------------------------------------------

    TEMPLATE_FIELD_COUNTS = {
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
    }

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        field_alias_to_id,
        field_alias_to_type,
        verified_operators,
        allowed_windows,
    ):
        """
        Parameters
        ----------
        field_alias_to_id:
            Alias -> actual BRAIN field ID.

        field_alias_to_type:
            Alias -> actual BRAIN field type.

        verified_operators:
            Either:

                set/list of verified operator names

            or:

                dictionary returned by
                verify_compiler_operators()

        allowed_windows:
            Explicit research windows allowed by the pipeline.
        """

        self.field_alias_to_id = {
            str(alias).strip(): str(field_id).strip()
            for alias, field_id
            in dict(field_alias_to_id).items()
        }

        self.field_alias_to_type = {
            str(alias).strip(): (
                str(field_type)
                .strip()
                .upper()
            )
            for alias, field_type
            in dict(field_alias_to_type).items()
        }

        # verify_compiler_operators() returns a dictionary.
        if isinstance(
            verified_operators,
            dict,
        ):

            self.verified_operators = set(
                verified_operators.keys()
            )

        else:

            self.verified_operators = {
                str(operator).strip()
                for operator
                in verified_operators
            }

        self.allowed_windows = {
            int(value)
            for value
            in allowed_windows
        }

        if not self.allowed_windows:

            raise ValueError(
                "allowed_windows cannot be empty."
            )

    # ========================================================
    # OPERATOR CHECK
    # ========================================================

    def _require_operator(
        self,
        operator: str,
    ) -> None:
        """
        Require an operator to have been verified against the
        live BRAIN operator catalog.
        """

        if (
            operator
            not in self.verified_operators
        ):

            raise ValueError(
                f"Operator {operator!r} "
                "has not been verified against "
                "the live BRAIN catalog."
            )

    # ========================================================
    # FIELD CHECK
    # ========================================================

    def _require_field(
        self,
        alias: str,
    ) -> None:
        """
        Verify that the alias exists and has a supported
        BRAIN field type.
        """

        alias = str(
            alias
        ).strip()

        if (
            alias
            not in self.field_alias_to_id
        ):

            raise ValueError(
                f"Unknown field alias: {alias}"
            )

        if (
            alias
            not in self.field_alias_to_type
        ):

            raise ValueError(
                f"No field type recorded for alias: {alias}"
            )

        field_type = (
            self.field_alias_to_type[
                alias
            ]
        )

        if field_type not in {
            "MATRIX",
            "VECTOR",
            "GROUP",
            "UNIVERSE",
            "SYMBOL",
        }:

            raise ValueError(
                f"Unsupported BRAIN field type "
                f"{field_type!r} for {alias}."
            )

    # ========================================================
    # FIELD -> MATRIX-VALUED SERIES
    # ========================================================

    def _field_series(
        self,
        alias: str,
        backfill_window: int,
    ) -> str:
        """
        Convert a field alias into a matrix-valued series.

        MATRIX:

            F1
            ->
            ts_backfill(F1,60)

        VECTOR:

            F1
            ->
            vec_avg(F1)
            ->
            ts_backfill(vec_avg(F1),60)

        GROUP / UNIVERSE / SYMBOL are rejected because these
        templates require ordinary numeric matrix-valued data.
        """

        alias = str(
            alias
        ).strip()

        self._require_field(
            alias
        )

        backfill_window = int(
            backfill_window
        )

        if (
            backfill_window
            not in self.allowed_windows
        ):

            raise ValueError(
                "Invalid backfill window: "
                f"{backfill_window}"
            )

        self._require_operator(
            "ts_backfill"
        )

        field_type = (
            self.field_alias_to_type[
                alias
            ]
        )

        # ----------------------------------------------------
        # MATRIX
        # ----------------------------------------------------

        if field_type == "MATRIX":

            base = alias

        # ----------------------------------------------------
        # VECTOR
        # ----------------------------------------------------

        elif field_type == "VECTOR":

            self._require_operator(
                "vec_avg"
            )

            base = (
                f"vec_avg({alias})"
            )

        # ----------------------------------------------------
        # GROUP / UNIVERSE / SYMBOL
        # ----------------------------------------------------

        else:

            raise ValueError(
                f"Field {alias} has BRAIN type "
                f"{field_type!r}. "
                "It cannot be used as an ordinary "
                "numeric template input."
            )

        return (
            f"ts_backfill("
            f"{base},"
            f"{backfill_window}"
            f")"
        )

    # ========================================================
    # SAFE DIVIDE DENOMINATOR
    # ========================================================

    def _safe_denominator(
        self,
        expression: str,
    ) -> str:
        """
        Protect a denominator from exact zero by adding a fixed
        epsilon.

        The epsilon is generated by Python rather than the LLM,
        so it is not another search parameter.
        """

        self._require_operator(
            "add"
        )

        return (
            f"add("
            f"{expression},"
            f"0.0001"
            f")"
        )

    # ========================================================
    # COMPILE
    # ========================================================

    def compile(
        self,
        template: str,
        fields: list[str],
        window: int = 60,
        backfill_window: int = 60,
        direction: str = "positive",
    ) -> CompileResult:
        """
        Compile one predefined research template into FASTEXPR.
        """

        # ----------------------------------------------------
        # Normalize inputs
        # ----------------------------------------------------

        template = str(
            template
        ).strip().upper()

        fields = [
            str(field).strip()
            for field
            in list(fields)
        ]

        window = int(
            window
        )

        backfill_window = int(
            backfill_window
        )

        direction = str(
            direction
        ).strip().lower()

        # ----------------------------------------------------
        # Template
        # ----------------------------------------------------

        if (
            template
            not in self.TEMPLATE_FIELD_COUNTS
        ):

            raise ValueError(
                f"Unknown template: {template}"
            )

        required_fields = (
            self.TEMPLATE_FIELD_COUNTS[
                template
            ]
        )

        if (
            len(fields)
            != required_fields
        ):

            raise ValueError(
                f"{template} requires "
                f"{required_fields} fields, "
                f"got {len(fields)}."
            )

        # ----------------------------------------------------
        # Windows
        # ----------------------------------------------------

        if (
            window
            not in self.allowed_windows
        ):

            raise ValueError(
                f"Invalid window: {window}"
            )

        if (
            backfill_window
            not in self.allowed_windows
        ):

            raise ValueError(
                "Invalid backfill window: "
                f"{backfill_window}"
            )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if direction not in {
            "positive",
            "negative",
        }:

            raise ValueError(
                "direction must be "
                "'positive' or 'negative'."
            )

        # ----------------------------------------------------
        # First field
        # ----------------------------------------------------

        first = self._field_series(
            fields[0],
            backfill_window,
        )

        # ====================================================
        # ONE-FIELD TEMPLATES
        # ====================================================

        if template == "LEVEL":

            self._require_operator(
                "rank"
            )

            expression = (
                f"rank("
                f"{first}"
                f")"
            )

        # ----------------------------------------------------

        elif template == "HISTORICAL_STATE":

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_zscore"
            )

            expression = (
                f"rank("
                f"ts_zscore("
                f"{first},"
                f"{window}"
                f")"
                f")"
            )

        # ----------------------------------------------------

        elif template == "CHANGE":

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_delta"
            )

            expression = (
                f"rank("
                f"ts_delta("
                f"{first},"
                f"{window}"
                f")"
                f")"
            )

        # ----------------------------------------------------

        elif template == "SMOOTHED":

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_mean"
            )

            expression = (
                f"rank("
                f"ts_mean("
                f"{first},"
                f"{window}"
                f")"
                f")"
            )

        # ----------------------------------------------------

        elif template == "STABILITY":

            self._require_operator(
                "reverse"
            )

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_std_dev"
            )

            expression = (
                f"reverse("
                f"rank("
                f"ts_std_dev("
                f"{first},"
                f"{window}"
                f")"
                f")"
                f")"
            )

        # ----------------------------------------------------

        elif template == "DECAY":

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_decay_linear"
            )

            expression = (
                f"rank("
                f"ts_decay_linear("
                f"{first},"
                f"{window}"
                f")"
                f")"
            )

        # ====================================================
        # TWO-FIELD RATIO TEMPLATES
        # ====================================================

        elif template in {
            "RATIO",
            "RATIO_STATE",
            "RATIO_CHANGE",
        }:

            second = self._field_series(
                fields[1],
                backfill_window,
            )

            self._require_operator(
                "divide"
            )

            # ------------------------------------------------
            # Zero-safe denominator.
            # ------------------------------------------------

            safe_second = (
                self._safe_denominator(
                    second
                )
            )

            ratio = (
                f"divide("
                f"{first},"
                f"{safe_second}"
                f")"
            )

            self._require_operator(
                "rank"
            )

            # ------------------------------------------------
            # Raw ratio
            # ------------------------------------------------

            if template == "RATIO":

                expression = (
                    f"rank("
                    f"{ratio}"
                    f")"
                )

            # ------------------------------------------------
            # Ratio z-score
            # ------------------------------------------------

            elif template == "RATIO_STATE":

                self._require_operator(
                    "ts_zscore"
                )

                expression = (
                    f"rank("
                    f"ts_zscore("
                    f"{ratio},"
                    f"{window}"
                    f")"
                    f")"
                )

            # ------------------------------------------------
            # Ratio change
            # ------------------------------------------------

            else:

                self._require_operator(
                    "ts_delta"
                )

                expression = (
                    f"rank("
                    f"ts_delta("
                    f"{ratio},"
                    f"{window}"
                    f")"
                    f")"
                )

        # ====================================================
        # TWO-FIELD INTERACTION / CONTRAST
        # ====================================================

        elif template in {
            "INTERACTION",
            "CONTRAST",
        }:

            second = self._field_series(
                fields[1],
                backfill_window,
            )

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_zscore"
            )

            first_z = (
                f"ts_zscore("
                f"{first},"
                f"{window}"
                f")"
            )

            second_z = (
                f"ts_zscore("
                f"{second},"
                f"{window}"
                f")"
            )

            # ------------------------------------------------
            # Add normalized signals
            # ------------------------------------------------

            if template == "INTERACTION":

                self._require_operator(
                    "add"
                )

                combined = (
                    f"add("
                    f"{first_z},"
                    f"{second_z}"
                    f")"
                )

            # ------------------------------------------------
            # Difference of normalized signals
            # ------------------------------------------------

            else:

                self._require_operator(
                    "subtract"
                )

                combined = (
                    f"subtract("
                    f"{first_z},"
                    f"{second_z}"
                    f")"
                )

            expression = (
                f"rank("
                f"{combined}"
                f")"
            )

        # ====================================================
        # CORRELATION
        # ====================================================

        elif template == "CORRELATION":

            second = self._field_series(
                fields[1],
                backfill_window,
            )

            self._require_operator(
                "rank"
            )

            self._require_operator(
                "ts_corr"
            )

            expression = (
                f"rank("
                f"ts_corr("
                f"{first},"
                f"{second},"
                f"{window}"
                f")"
                f")"
            )

        # ====================================================
        # SAFETY
        # ====================================================

        else:

            raise RuntimeError(
                "Compiler dispatch failure for "
                f"template: {template}"
            )

        # ====================================================
        # SIGNAL DIRECTION
        # ====================================================

        if direction == "negative":

            self._require_operator(
                "reverse"
            )

            expression = (
                f"reverse("
                f"{expression}"
                f")"
            )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return CompileResult(
            expression=expression,
            template=template,
            fields=fields,
            window=window,
            backfill_window=backfill_window,
            direction=direction,
        )