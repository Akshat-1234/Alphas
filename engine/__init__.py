# ============================================================
# engine/__init__.py
# ============================================================

from .brain import BrainContext

from .fields import (
    build_field_catalog,
    classify_field_type,
)

from .operators import (
    build_operator_catalog,
    get_operator_signature,
    parse_operator_signature,
)

from .compiler import (
    CompileResult,
    FastExprCompiler,
)

from .validator import (
    FastExprValidator,
)

from .simulator import (
    SimulationRunner,
)


__all__ = [
    # BRAIN context
    "BrainContext",

    # Fields
    "build_field_catalog",
    "classify_field_type",

    # Operators
    "build_operator_catalog",
    "get_operator_signature",
    "parse_operator_signature",

    # Compiler
    "CompileResult",
    "FastExprCompiler",

    # Validator
    "FastExprValidator",

    # Simulation
    "SimulationRunner",
]