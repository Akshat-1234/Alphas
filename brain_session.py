# ============================================================
# brain_session.py
# ============================================================

import ace_lib as ace


# ============================================================
# CONFIGURATION
# ============================================================

REGION = "GLB"
UNIVERSE = "TOPDIV3000"
DATASET_ID = "fundamental6"
DELAY = 1


# ============================================================
# SINGLE SESSION
# ============================================================
#
# IMPORTANT:
# This module creates the BRAIN session only once per Python
# process.
#
# Other modules should import `session` from this module rather
# than calling ace.start_session() themselves.
# ============================================================

print("=" * 80)
print("STARTING WORLDQUANT BRAIN SESSION")
print("=" * 80)

session = ace.start_session()

print(
    "BRAIN session created:",
    session is not None,
)


# ============================================================
# SHARED CONTEXT
# ============================================================

from engine.brain import BrainContext


brain = BrainContext(
    session=session,
    region=REGION,
    universe=UNIVERSE,
    dataset_id=DATASET_ID,
    delay=DELAY,
)


# ============================================================
# PUBLIC HELPERS
# ============================================================

def get_session():
    """
    Return the already-authenticated BRAIN session.
    """

    return session


def get_brain_context():
    """
    Return the shared BrainContext.

    This does not create a new session.
    """

    return brain