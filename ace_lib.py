import getpass
import json
import logging
import os
import queue
import random
import threading
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Literal, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
import tqdm
from helpful_functions import (
    expand_dict_columns,
    save_is_tests,
    save_pnl,
    save_simulation_result,
    save_yearly_stats,
)

DEV = False

# Bumped whenever this file is patched, so a quick `ace.ACE_LIB_VERSION` check in a
# notebook can instantly confirm whether a running kernel is on the latest copy of
# this file, rather than re-diagnosing an already-fixed bug against a stale import.
ACE_LIB_VERSION = "2026-07-13-queue-ui"

_UNSET = object()

Category = Optional[
    Literal[
        "PRICE_REVERSION",
        "PRICE_MOMENTUM",
        "VOLUME",
        "FUNDAMENTAL",
        "ANALYST",
        "PRICE_VOLUME",
        "RELATION",
        "SENTIMENT",
    ]
]


class SingleSession(requests.Session):
    _instance = None
    _lock = threading.Lock()
    _relogin_lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not self._initialized:
            super(SingleSession, self).__init__(*args, **kwargs)
            self._initialized = True

    def get_relogin_lock(self):
        return self._relogin_lock


def setup_logger() -> logging.Logger:
    """
    This function sets up a logger that writes log messages to the console and,
    if the global variable DEV is set to True, also to a file named 'ace.log'.

    Returns:
        logger (logging.Logger): The configured logger object.

    The logger's name is set to 'ace.log'. The level of the logger and the console handler
    is set to INFO if DEV is True, and WARNING otherwise. The format for the log messages
    is: 'asctime' - 'name' - 'levelname' - 'message'.
    """
    logger = logging.getLogger("ace")
    level = logging.DEBUG if DEV else logging.INFO

    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("ace.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "save_is_tests_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
brain_url = os.environ.get("BRAIN_URL", "https://platform.worldquantbrain.com")

# Per BRAIN API docs, POST /authentication accepts an optional 'expiry' field
# (seconds, 1..14400 i.e. up to 4 hours). start_session() previously omitted this
# entirely, leaving session length up to an undocumented server-side default that
# turned out to be considerably shorter than 4 hours in practice - causing
# check_session_and_relogin's proactive refresh (triggered with <2000s remaining) to
# fire much sooner than expected, and each such refresh is a genuine new login that
# can re-trigger a persona/biometric step-up challenge. Requesting the maximum
# explicitly reduces how often that refresh - and the interactive challenge it can
# require - happens at all.
SESSION_REQUESTED_EXPIRY_SECONDS = 14400

# How often (seconds) to proactively re-check/refresh the session while polling a
# long-running simulation. Kept comfortably below the 2000s refresh threshold used
# by check_session_and_relogin() so long simulations never drift past their token's
# safe window while sitting in a poll loop.
SESSION_POLL_CHECK_INTERVAL_SECONDS = 600

# --- Simulation status vocabulary (per BRAIN API docs: GET /simulations/<id>) ---
# Terminal statuses that mean the simulation is done and failed.
SIMULATION_FAIL_STATUSES = {"ERROR", "CANCELLED", "TIMEOUT", "FAIL"}
# Terminal statuses that mean the simulation is done and an alpha was produced.
SIMULATION_SUCCESS_STATUSES = {"COMPLETE", "WARNING"}
# Non-terminal statuses (still in progress).
SIMULATION_IN_PROGRESS_STATUSES = {"WAITING", "SIMULATING"}

# Retry/backoff config for CONCURRENT_SIMULATION_LIMIT_EXCEEDED on simulation start.
CONCURRENT_LIMIT_MAX_RETRIES = 5
CONCURRENT_LIMIT_BASE_BACKOFF_SECONDS = 15
CONCURRENT_LIMIT_MAX_BACKOFF_SECONDS = 180


def _get_error_detail(response: requests.Response) -> str:
    """
    Best-effort extraction of the 'detail' field from a failed API response body.

    Args:
        response (requests.Response): The response to inspect.

    Returns:
        str: The value of the 'detail' field, or an empty string if unavailable/unparseable.
    """
    try:
        return response.json().get("detail", "") or ""
    except Exception:
        return ""


def _safe_json(response: requests.Response) -> dict:
    """
    Parse a response body as JSON, tolerating non-JSON bodies (HTML error pages,
    empty bodies, proxy/gateway error pages, etc.) and non-dict JSON (null, a bare
    list, etc.) that can occur alongside HTTP errors. Every call site treats the
    result as a dict (via .get()), so anything else is normalized to {} rather than
    raising later when logging or inspecting the response.

    Args:
        response (requests.Response): The response to parse.

    Returns:
        dict: The parsed JSON body if it's a dict, otherwise an empty dict.
    """
    try:
        body = response.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


# Fixed filename (relative to whatever directory the notebook/script is running in,
# same convention as simulation_results/, alphas_pnl/, etc. in helpful_functions.py)
# that start_session() writes to whenever it needs interactive action (persona
# biometrics, or - after retries exhausted - a fresh credential prompt). The
# companion ace_dashboard.html polls this file so an action-needed banner can show up
# in a browser tab instead of requiring someone to be watching the notebook/terminal.
PENDING_ACTION_FILE = "ace_pending_action.json"
_pending_action_lock = threading.Lock()


def _write_status_json(path: str, status: dict) -> None:
    """
    Best-effort atomic write of a status dict to a JSON file, for a companion UI to
    poll. Writes to a temp file then renames over the target (atomic on POSIX and
    Windows), so a poller never reads a half-written file. Failures here (disk full,
    permissions, etc.) are logged at debug level and never allowed to interrupt
    whatever real work is being monitored - monitoring must not be able to break the
    thing it's monitoring.

    Args:
        path (str): Destination file path.
        status (dict): JSON-serializable status payload.
    """
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(status, f, default=str)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.debug(f"Could not write status file {path}: {e}")


def _set_pending_action(action: Optional[dict]) -> None:
    """
    Write (or clear) the pending-action file that signals a person needs to do
    something interactive (complete biometrics, re-enter credentials, etc.).

    Args:
        action (dict | None): Action payload (e.g. {"type": "persona", "url": ...,
            "message": ...}), or None to clear a previously-set action.
    """
    with _pending_action_lock:
        if action is None:
            _write_status_json(PENDING_ACTION_FILE, {"pending": False})
        else:
            payload = {"pending": True, "since": time.time(), **action}
            _write_status_json(PENDING_ACTION_FILE, payload)


# Retry/backoff config for raw transport-level failures (connection dropped,
# remote end closed connection, read timeout, etc.) - distinct from HTTP-level
# errors, which arrive as a normal response with a non-2xx status code. These
# never reach the response-handling code at all; requests/urllib3 raises before
# a Response object even exists, so every s.get()/s.post() call across this
# module is a potential uncaught-crash site unless wrapped.
TRANSPORT_ERROR_MAX_RETRIES = 4
TRANSPORT_ERROR_BASE_BACKOFF_SECONDS = 5
TRANSPORT_ERROR_MAX_BACKOFF_SECONDS = 60


def _resilient_get(
    s: SingleSession,
    url: str,
    max_retries: int = TRANSPORT_ERROR_MAX_RETRIES,
    base_backoff: float = TRANSPORT_ERROR_BASE_BACKOFF_SECONDS,
) -> requests.Response:
    """
    GET a URL with retry/backoff specifically for transport-level failures
    (requests.exceptions.RequestException and subclasses - ConnectionError,
    RemoteDisconnected, ChunkedEncodingError, Timeout, etc.) that occur before any
    HTTP response is received. These are distinct from - and not caught by - the
    HTTP-status-code retry logic elsewhere in this module, since there's no
    response object to inspect a status code on. A single transient network blip
    here previously crashed the entire calling ThreadPool worker uncaught,
    discarding results for every other alpha already collected in that batch.

    Args:
        s (SingleSession): An authenticated session object.
        url (str): The URL to GET.
        max_retries (int, optional): Max retry attempts. Defaults to
            TRANSPORT_ERROR_MAX_RETRIES.
        base_backoff (float, optional): Base seconds for exponential backoff
            between retries. Defaults to TRANSPORT_ERROR_BASE_BACKOFF_SECONDS.

    Returns:
        requests.Response: The response, once a request actually completes.

    Raises:
        requests.exceptions.RequestException: If every retry attempt still fails
            at the transport level. Callers that can tolerate a missing/partial
            result (e.g. a single alpha's correlation check) should catch this and
            degrade gracefully rather than letting it propagate through a
            ThreadPool and take down an entire batch.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return s.get(url)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt >= max_retries:
                break
            backoff = min(base_backoff * (2**attempt), TRANSPORT_ERROR_MAX_BACKOFF_SECONDS)
            logger.warning(
                f"Network error on GET {url} (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {backoff:.0f}s."
            )
            time.sleep(backoff)
    logger.error(f"GET {url} failed after {max_retries} retries due to network errors: {last_exc}")
    raise last_exc


def _extract_expression(body: dict) -> str:
    """
    Best-effort extraction of the alpha expression from a /simulations/<id> resource
    body, for use in failure logging. The GET /simulations/<id> response echoes back
    the original submitted fields (regular, or selection/combo for SUPER alphas)
    alongside status - unlike the /alphas/<id> resource, where 'regular' is nested
    as {"code": ...}, here it's typically the raw expression string.

    Args:
        body (dict): A parsed /simulations/<id> response body (e.g. from _safe_json).

    Returns:
        str: The expression, or a placeholder if it couldn't be found.
    """
    if not isinstance(body, dict):
        return "<expression unavailable>"

    regular = body.get("regular")
    if regular is not None:
        # Defensive: handle both the raw-string shape (simulations resource) and the
        # nested {"code": ...} shape (alphas resource), in case the API returns either.
        if isinstance(regular, dict):
            return regular.get("code", "<expression unavailable>")
        return str(regular)

    selection = body.get("selection")
    combo = body.get("combo")
    if selection is not None or combo is not None:
        return f"selection={selection}, combo={combo}"

    return "<expression unavailable>"


def _is_concurrent_limit_error(response: requests.Response) -> bool:
    """
    Check whether a failed /simulations POST response was rejected specifically because
    the account's concurrent simulation limit was exceeded (as opposed to a bad request,
    auth failure, or other error).

    Args:
        response (requests.Response): The response from POST /simulations.

    Returns:
        bool: True if the failure is a CONCURRENT_SIMULATION_LIMIT_EXCEEDED rejection.
    """
    if response.status_code // 100 == 2:
        return False
    return "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in _get_error_detail(response)


def get_credentials() -> tuple[str, str]:
    """
    Retrieve or prompt for platform credentials.

    This function attempts to read credentials from a JSON file in the user's home directory.
    If the file doesn't exist or is empty, it prompts the user to enter credentials and saves them.

    Returns:
        tuple: A tuple containing the email and password.

    Raises:
        json.JSONDecodeError: If the credentials file exists but contains invalid JSON.
    """

    credential_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL")
    credential_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD")

    credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
    credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")

    if Path(credentials_file_path).exists() and os.path.getsize(credentials_file_path) > 2:
        with open(credentials_file_path) as file:
            data = json.loads(file.read())
    else:
        os.makedirs(credentials_folder_path, exist_ok=True)
        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            email = input("Email:\n")
            password = getpass.getpass(prompt="Password:")
        data = {"email": email, "password": password}
        with open(credentials_file_path, "w") as file:
            json.dump(data, file)
    return (data["email"], data["password"])


def start_session() -> SingleSession:
    """
    Start a new session with the WorldQuant BRAIN platform.

    This function authenticates the user, handles biometric authentication if required,
    and creates a new session.

    Before assuming a bare 401 means the cached credentials are wrong, this function
    retries a couple of times with the same credentials (a 401 here can also be caused
    by a transient server hiccup or a reCAPTCHA/rate-limit lockout from repeated
    re-authentication calls, per the BRAIN API docs - not necessarily a bad password).
    Only after repeated genuine 401s does it fall back to wiping the cached
    credentials file and forcing an interactive re-login, matching the previous
    behavior as a last resort.

    Returns:
        SingleSession: An authenticated session object.

    Raises:
        requests.exceptions.RequestException: If there's an error during the authentication process.
        RuntimeError: If the platform requires a reCAPTCHA solve, which can't be
            completed non-interactively from this client.
    """

    max_bare_401_retries = 2
    for attempt in range(max_bare_401_retries + 1):
        s = SingleSession()
        s.auth = get_credentials()
        r = s.post(brain_api_url + "/authentication", json={"expiry": SESSION_REQUESTED_EXPIRY_SECONDS})
        try:
            debug_body = r.json()
        except Exception:
            debug_body = "<non-JSON response body>"
        logger.debug(f"New session created (ID: {id(s)}) with authentication response: {r.status_code}, {debug_body}")

        if r.status_code != requests.status_codes.codes.unauthorized:
            _set_pending_action(None)
            return s

        # Use .get() rather than direct indexing: a 401 without this header at all
        # (e.g. a transient/unexpected server response) would otherwise raise a raw
        # KeyError here instead of being handled below.
        www_authenticate = r.headers.get("WWW-Authenticate", "")
        if www_authenticate == "persona":
            persona_url = urljoin(r.url, r.headers["Location"])
            _set_pending_action(
                {
                    "type": "persona",
                    "url": persona_url,
                    "message": "Complete biometrics authentication, then this will continue automatically once done.",
                }
            )
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + persona_url
                + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))

            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input(
                        "Biometrics authentication is not complete. Please try again and press any key when completed \n"
                    )
                else:
                    break
            _set_pending_action(None)
            return s

        try:
            body = r.json()
        except Exception:
            body = {}
        # Per the BRAIN API docs, a 401 that requires reCAPTCHA looks like:
        # {"detail": "...", "recaptcha": ["This field is required."]} - i.e. "recaptcha"
        # is a top-level key. We also defensively check the detail string in case the
        # shape varies, since that check costs nothing and only widens detection.
        detail_text = str(body.get("detail", "")).lower() if isinstance(body, dict) else ""
        recaptcha_flagged = (isinstance(body, dict) and "recaptcha" in body) or "recaptcha" in detail_text
        if recaptcha_flagged:
            _set_pending_action(
                {
                    "type": "recaptcha",
                    "url": brain_url,
                    "message": "Authentication requires a reCAPTCHA solve. Sign in through the browser, then retry.",
                }
            )
            logger.error(
                "\nAuthentication requires a reCAPTCHA solve, which this client can't complete "
                f"automatically. Sign in through the browser at {brain_url}, then retry.\n"
            )
            raise RuntimeError(
                "Authentication requires reCAPTCHA; cannot proceed non-interactively. "
                "Sign in via the browser, then retry."
            )

        if attempt < max_bare_401_retries:
            logger.warning(
                f"Authentication returned 401 (attempt {attempt + 1}/{max_bare_401_retries}) without a persona "
                "challenge or reCAPTCHA requirement - this can be transient (e.g. a brief server "
                "hiccup), so retrying with the same cached credentials before assuming they're invalid."
            )
            time.sleep(5)
            continue

        logger.error("\nIncorrect email or password\n")
        with open(
            os.path.join(os.path.expanduser("~"), "secrets/platform-brain.json"),
            "w",
        ) as file:
            json.dump({}, file)
        # Credentials cache is now cleared; loop back around once more so the next
        # get_credentials() call prompts interactively rather than recursing.
        return start_session()


def check_session_timeout(s: SingleSession) -> Optional[int]:
    """
    Check if the current session has timed out.

    Args:
        s (SingleSession): The current session object.

    Returns:
        Optional[int]: The number of seconds until the session expires, as reported
        by the server. None if the check itself failed (network error, timeout,
        malformed response, etc.) - this is deliberately distinct from a genuine
        low/zero expiry the server actually reported. Conflating the two would mean
        a passing network blip on this one status-check request gets treated
        identically to "session has actually expired", forcing an unnecessary full
        relogin (and, on accounts that require it, a fresh persona/biometric
        challenge) every time this check happens to fail for any reason. Callers
        should treat None as "couldn't verify - don't assume expired", not as 0.
    """

    authentication_url = brain_api_url + "/authentication"
    try:
        result = s.get(authentication_url).json()["token"]["expiry"]
        logger.debug(f"Session (ID: {id(s)}) timeout check result: {result}")
        return result
    except Exception as e:
        logger.debug(f"Session (ID: {id(s)}) timeout check failed (transient?): {e}")
        return None


def generate_alpha(
    regular: Optional[str] = None,
    selection: Optional[str] = None,
    combo: Optional[str] = None,
    alpha_type: Literal["REGULAR", "SUPER"] = "REGULAR",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: Literal[0, 1] = 1,
    decay: int = 0,
    neutralization: str = "INDUSTRY",
    truncation: float = 0.08,
    pasteurization: Literal["ON", "OFF"] = "ON",
    test_period: str = "P0Y0M0D",
    unit_handling: Literal["VERIFY"] = "VERIFY",
    nan_handling: Literal["ON", "OFF"] = "OFF",
    max_trade: Literal["ON", "OFF"] = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 100,
    visualization: bool = False,
) -> dict:
    """
    Generate an alpha dictionary for simulation. If alpha_type='REGULAR',
    function generates alpha dictionary using regular input. If alpha_type='SUPER',
    function generates alpha dictionary using selection and combo inputs.

    Args:
        regular (str, optional): The regular alpha expression.
        selection (str, optional): The selection expression for super alphas.
        combo (str, optional): The combo expression for super alphas.
        alpha_type (str, optional): The type of alpha ("REGULAR" or "SUPER"). Defaults to "REGULAR".
        region (str, optional): The region for the alpha. Defaults to "USA".
        universe (str, optional): The universe for the alpha. Defaults to "TOP3000".
        delay (int, optional): The delay for the alpha. Defaults to 1.
        decay (int, optional): The decay for the alpha. Defaults to 0.
        neutralization (str, optional): The neutralization method. Defaults to "INDUSTRY".
        truncation (float, optional): The truncation value. Defaults to 0.08.
        pasteurization (str, optional): The pasteurization setting. Defaults to "ON".
        test_period (str, optional): The test period. Defaults to "P0Y0M0D".
        unit_handling (str, optional): The unit handling method. Defaults to "VERIFY".
        nan_handling (str, optional): The NaN handling method. Defaults to "OFF".
        max_trade (str, optional): The max trade method. Defaults to "OFF".
        selection_handling (str, optional): The selection handling method for super alphas. Defaults to "POSITIVE".
        selection_limit (int, optional): The selection limit for super alphas. Defaults to 100.
        visualization (bool, optional): Whether to include visualization. Defaults to False.

    Returns:
        dict: A dictionary containing the alpha configuration for simulation.

    Raises:
        ValueError: If an invalid alpha_type is provided.
    """

    settings = {
        "instrumentType": "EQUITY",
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "testPeriod": test_period,
        "unitHandling": unit_handling,
        "nanHandling": nan_handling,
        "maxTrade": max_trade,
        "language": "FASTEXPR",
        "visualization": visualization,
    }
    if alpha_type == "REGULAR":
        simulation_data = {
            "type": alpha_type,
            "settings": settings,
            "regular": regular,
        }
    elif alpha_type == "SUPER":
        simulation_data = {
            "type": alpha_type,
            "settings": {
                **settings,
                "selectionHandling": selection_handling,
                "selectionLimit": selection_limit,
            },
            "combo": combo,
            "selection": selection,
        }
    else:
        logger.error("alpha_type should be REGULAR or SUPER")
        return {}
    return simulation_data


def check_session_and_relogin(s: SingleSession) -> SingleSession:
    """
    Checks for session timeout and if less than 2000 seconds are remaining,
    it attempts to start a new session.

    Parameters:
        s (SingleSession): The current session object.

    Returns:
        s (SingleSession): The original session object if it hasn't timed out,
        otherwise a new session object.

    If the remaining session time is less than 2000 seconds, the function
    attempts to start a new session using the `start_session()` function.
    If `start_session()` fails on the first attempt with a transient/unexpected
    error, it waits for 100 seconds and then tries again. If `start_session()`
    raises `RuntimeError` (i.e. the platform requires a reCAPTCHA solve that can't
    be completed non-interactively), that is re-raised immediately rather than
    retried, since waiting and retrying won't resolve a reCAPTCHA requirement and
    would only bury the message the person needs to see.

    A single failed status check (check_session_timeout returning None - a network
    blip, not a genuine low expiry from the server) does not by itself trigger a
    relogin: the check is retried once after a short pause first, since forcing a
    full relogin (and, on accounts that require it, a fresh persona/biometric
    challenge blocking the whole run) over what might just be a transient hiccup on
    this one status-check request is far more disruptive than a few seconds' delay.
    """
    with s.get_relogin_lock():
        timeout_remaining = check_session_timeout(s)
        if timeout_remaining is None:
            time.sleep(5)
            timeout_remaining = check_session_timeout(s)

        if timeout_remaining is None:
            # Still couldn't verify after a retry - proceed with the existing
            # session rather than treating "unknown" as "expired". If it really has
            # expired, the next actual API call will surface that on its own merits
            # (e.g. a 401), rather than this status check guessing wrong.
            logger.warning(
                f"Could not verify session status after retry (network issue?) for session (ID: {id(s)}) - "
                "continuing with the existing session rather than forcing an unnecessary relogin."
            )
        elif timeout_remaining < 2000:
            logger.debug('Session less than 2000 seconds')
            try:
                s = start_session()
            except RuntimeError:
                raise
            except Exception:
                logger.info('Trying re-login, wait 100 seconds')
                time.sleep(100)
                s = start_session()
        logger.debug(f"Session (ID: {id(s)}) after check and relogin")
    return s


def start_simulation(
    s: SingleSession,
    simulate_data: Union[list[dict], dict],
    max_retries: int = CONCURRENT_LIMIT_MAX_RETRIES,
    base_backoff: float = CONCURRENT_LIMIT_BASE_BACKOFF_SECONDS,
) -> requests.Response:
    """
    Start a simulation with the provided simulation data.

    If the platform rejects the request specifically with
    CONCURRENT_SIMULATION_LIMIT_EXCEEDED (too many simulations already in flight for
    this account), this function waits and retries with exponential backoff instead
    of immediately giving up, since that error is transient by nature (it clears as
    soon as any in-flight simulation finishes). Any other error (bad request, auth
    failure, etc.) is returned immediately without retrying, exactly as before.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A dictionary containing the simulation parameters.
        max_retries (int, optional): Max retry attempts specifically for
            CONCURRENT_SIMULATION_LIMIT_EXCEEDED. Defaults to CONCURRENT_LIMIT_MAX_RETRIES.
        base_backoff (float, optional): Base seconds for exponential backoff between
            retries. Defaults to CONCURRENT_LIMIT_BASE_BACKOFF_SECONDS.

    Returns:
        requests.Response: The response object from the simulation start request.
            Same contract as before: callers (e.g. simulation_progress) that only
            check status_code and headers are unaffected whether or not a retry
            happened internally.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    attempt = 0
    while True:
        simulate_response = s.post(brain_api_url + "/simulations", json=simulate_data)
        if not _is_concurrent_limit_error(simulate_response):
            return simulate_response

        if attempt >= max_retries:
            logger.warning(
                f"CONCURRENT_SIMULATION_LIMIT_EXCEEDED persisted after {attempt} retries, giving up on this alpha."
            )
            return simulate_response

        backoff = min(base_backoff * (2**attempt), CONCURRENT_LIMIT_MAX_BACKOFF_SECONDS)
        logger.info(
            f"CONCURRENT_SIMULATION_LIMIT_EXCEEDED (attempt {attempt + 1}/{max_retries}), "
            f"waiting {backoff:.0f}s for a slot to free up before retrying."
        )
        time.sleep(backoff)
        attempt += 1


def _start_simulation_persistent(
    s: SingleSession,
    simulate_data: Union[list[dict], dict],
    retry_interval: float = 30.0,
    max_wait_seconds: Optional[float] = None,
    on_wait_state_change: Optional[Callable[[bool], None]] = None,
) -> requests.Response:
    """
    Like start_simulation, but retries CONCURRENT_SIMULATION_LIMIT_EXCEEDED at a fixed
    interval (with jitter) for as long as needed (unbounded by default) instead of a
    short bounded exponential backoff. start_simulation's ~7-minute total retry window
    is fine for quick single-alpha submissions, but a full multi-simulation batch
    typically runs 10-40 minutes - a worker waiting for a slot needs to be able to wait
    that long (or longer, if several batches are queued ahead of it), not give up after
    a few minutes. Used by the queue-based simulate_alpha_queue().

    Jitter matters here specifically because simulate_alpha_queue runs many workers
    (e.g. 12) concurrently, all of which typically start waiting at roughly the same
    moment (right after the queue is populated). Without jitter they'd retry in
    lockstep - every worker hitting POST /simulations within the same instant, every
    retry_interval seconds, for however long the wait lasts. Jitter spreads those
    retries out over time instead.

    Any error other than CONCURRENT_SIMULATION_LIMIT_EXCEEDED is returned immediately,
    exactly like start_simulation - this only changes the waiting behavior for that one
    specific, transient, capacity-related rejection.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A dictionary (or list, for multi-simulation) containing
            the simulation parameters.
        retry_interval (float, optional): Base seconds to wait between retries; the
            actual wait is randomized to between 0.7x and 1.3x this value each time, to
            desynchronize multiple concurrently-waiting workers. Defaults to 30.0.
        max_wait_seconds (float, optional): If set, stop retrying and return the last
            rejected response after this many seconds have elapsed in total. Defaults
            to None (retry indefinitely - recommended, since giving up just means the
            batch needs to be resubmitted later anyway).
        on_wait_state_change (callable, optional): Called with True the moment this
            starts waiting for a slot, and False once it stops (accepted, gave up, or
            hit a non-capacity error). Used by simulate_alpha_queue to report live
            "N running / M waiting" worker counts to its status file; purely an
            observability hook, has no effect on retry behavior itself.

    Returns:
        requests.Response: The response object from the simulation start request.
    """
    start_time = time.time()
    last_session_check = time.time()
    is_waiting = False
    while True:
        simulate_response = s.post(brain_api_url + "/simulations", json=simulate_data)
        if not _is_concurrent_limit_error(simulate_response):
            if is_waiting and on_wait_state_change is not None:
                on_wait_state_change(False)
            return simulate_response

        if not is_waiting and on_wait_state_change is not None:
            on_wait_state_change(True)
        is_waiting = True

        elapsed = time.time() - start_time
        if max_wait_seconds is not None and elapsed >= max_wait_seconds:
            logger.warning(
                f"CONCURRENT_SIMULATION_LIMIT_EXCEEDED persisted for {elapsed:.0f}s "
                f"(max_wait_seconds={max_wait_seconds:.0f}), giving up on this batch."
            )
            if on_wait_state_change is not None:
                on_wait_state_change(False)
            return simulate_response

        jittered_wait = retry_interval * random.uniform(0.7, 1.3)
        logger.info(
            f"CONCURRENT_SIMULATION_LIMIT_EXCEEDED - all slots busy, waiting {jittered_wait:.0f}s "
            f"before retrying (waited {elapsed:.0f}s so far)."
        )
        time.sleep(jittered_wait)

        # A worker can sit in this wait loop far longer than the session's requested
        # expiry (SESSION_REQUESTED_EXPIRY_SECONDS = 4h) - that's the normal case for
        # simulate_alpha_queue, whose whole point is waiting out a busy account for as
        # long as it takes. Without this check, the token expires silently while
        # waiting, the next POST above comes back 401 (not a CONCURRENT_SIMULATION_
        # LIMIT_EXCEEDED body), _is_concurrent_limit_error() sees a non-matching
        # rejection and returns it as if it were a final response, and the batch's
        # results are lost - only surfacing later as a forced interactive re-login
        # once the post-batch stats pass calls check_session_and_relogin(). Refreshing
        # here, on the same interval used by the progress-polling loops, keeps the
        # token alive for the whole wait instead of just the initial submission.
        if time.time() - last_session_check > SESSION_POLL_CHECK_INTERVAL_SECONDS:
            s = check_session_and_relogin(s)
            last_session_check = time.time()


def simulation_progress(
    s: SingleSession,
    simulate_response: requests.Response,
    show_progress: bool = False,
) -> dict:
    """
    Monitor the progress of a simulation and return the result when complete.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_response (requests.Response): The response from starting the simulation.
        show_progress (bool, optional): If True, display a live tqdm progress bar driven
            by the 'progress' field (0..1) that BRAIN returns on GET /simulations/<id>
            while status is WAITING/SIMULATING. Defaults to False, since this is
            intended for interactive single-simulation use (e.g. simulate_single_alpha
            called directly) rather than large threaded batches, where many concurrent
            bars would clutter output - simulate_alpha_list_multi does not enable this.

    Returns:
        dict: A dictionary containing the completion status and simulation result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if simulate_response.status_code // 100 != 2:
        detail = _get_error_detail(simulate_response)
        logger.warning(f'Simulation failed. {simulate_response.text}, Status code: {simulate_response.status_code}')
        return {"completed": False, "result": {}, "status": "START_FAILED", "message": detail}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    final_status = None
    final_message = ""
    retry_count = 0
    last_session_check = time.time()
    progress_bar = tqdm.tqdm(total=100, desc="Simulation progress", unit="%", leave=False) if show_progress else None
    try:
        while True:
            # Long-running simulations can outlast the session token even though the
            # token was fine when this call started (check_session_and_relogin only
            # runs once, before submission). Refresh periodically here so an expired
            # token doesn't surface mid-poll as an unexplained HTTP error.
            if time.time() - last_session_check > SESSION_POLL_CHECK_INTERVAL_SECONDS:
                s = check_session_and_relogin(s)
                last_session_check = time.time()
            try:
                simulation_progress_response = _resilient_get(s, simulation_progress_url)
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Simulation {simulation_progress_url} failed after exhausting transport-level retries: {e}"
                )
                error_flag = True
                final_status = "CONNECTION_ERROR"
                final_message = str(e)
                break
            if simulation_progress_response.status_code // 100 != 2:
                logger.error(
                    f'Simulation {simulation_progress_url}, Status code: {simulation_progress_response.status_code}, Retry'
                )
                time.sleep(30)
                retry_count += 1
                if retry_count <= 2:
                    continue
                else:
                    logger.error(
                        f'Simulation {simulation_progress_url} failed, Status code: {simulation_progress_response.status_code}'
                    )
                    error_flag = True
                    final_status = "HTTP_ERROR"
                    break
            if simulation_progress_response.headers.get("Retry-After", 0) == 0:
                body = _safe_json(simulation_progress_response)
                status = body.get("status", "ERROR")
                final_status = status
                final_message = body.get("message", "")
                # Per BRAIN API docs, status can be one of: WAITING, SIMULATING, CANCELLED,
                # COMPLETE, WARNING, ERROR, TIMEOUT, FAIL. Retry-After == 0 means the platform
                # considers this terminal, so any status other than a known success status is
                # treated as a failure - this covers CANCELLED/TIMEOUT/FAIL in addition to the
                # previously-handled ERROR, rather than lumping them all together as "ERROR".
                if status not in SIMULATION_SUCCESS_STATUSES:
                    error_flag = True
                    if status not in SIMULATION_FAIL_STATUSES and status not in SIMULATION_IN_PROGRESS_STATUSES:
                        logger.warning(
                            f"Simulation returned an unrecognized terminal status '{status}' - treating as a "
                            "failure (fail-closed) rather than assuming success. If WQB has added a new status, "
                            "update SIMULATION_SUCCESS_STATUSES/SIMULATION_FAIL_STATUSES accordingly."
                        )
                break

            if progress_bar is not None:
                # Per BRAIN API docs: "progress: <number: 0..1 progress of the
                # asynchronous request if available>" - only present while still running.
                body = _safe_json(simulation_progress_response)
                fraction = body.get("progress")
                if isinstance(fraction, (int, float)):
                    pct = min(max(fraction * 100, 0), 100)
                    progress_bar.n = pct
                    progress_bar.refresh()

            time.sleep(float(simulation_progress_response.headers["Retry-After"]))
    finally:
        if progress_bar is not None:
            progress_bar.n = 100 if not error_flag else progress_bar.n
            progress_bar.refresh()
            progress_bar.close()

    if error_flag:
        logger.error(f"Simulation failed with status={final_status}. {_safe_json(simulation_progress_response)}")
        return {"completed": False, "result": {}, "status": final_status, "message": final_message}

    alpha = _safe_json(simulation_progress_response).get("alpha", 0)
    if alpha == 0:
        logger.warning(
            f'Simulation {_safe_json(simulation_progress_response).get("id")} failed. '
            f'{_safe_json(simulation_progress_response)}'
        )
        return {"completed": False, "result": {}, "status": final_status, "message": final_message}
    simulation_result = get_simulation_result_json(s, alpha)
    if len(simulation_result) == 0:
        return {"completed": False, "result": {}, "status": final_status, "message": final_message}
    return {"completed": True, "result": simulation_result, "status": final_status, "message": final_message}


def get_simulation_result_json(s: SingleSession, alpha_id: str) -> dict:
    """
    Retrieve the full simulation result for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        dict: A dictionary containing the full simulation result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    if alpha_id is None:
        return {}
    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id)
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    try:
        return result.json()
    except Exception:
        logger.error(f"alpha_id {alpha_id}, {result.headers}, {result.text}, {result.status_code}")
        return {}


def multisimulation_progress(
    s: SingleSession,
    simulate_response: requests.Response,
    show_progress: bool = False,
) -> dict:
    """
    Monitor the progress of multiple simulations and return the results when complete.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_response (requests.Response): The response from starting the simulations.
        show_progress (bool, optional): If True, display a live tqdm progress bar driven
            by the parent multi-simulation's 'progress' field (0..1). Defaults to False.
            Not enabled by simulate_alpha_list_multi's batch runs (many concurrent bars
            would clutter output there); intended for direct/interactive use.

    Returns:
        dict: A dictionary containing the completion status and simulation results.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if simulate_response.status_code // 100 != 2:
        detail = _get_error_detail(simulate_response)
        logger.warning(f'Simulation failed. {simulate_response.text}, Status code: {simulate_response.status_code}')
        return {"completed": False, "result": {}, "status": "START_FAILED", "message": detail}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    final_status = None
    retry_count = 0
    last_session_check = time.time()
    progress_bar = (
        tqdm.tqdm(total=100, desc="Multi-simulation progress", unit="%", leave=False) if show_progress else None
    )
    try:
        while True:
            if time.time() - last_session_check > SESSION_POLL_CHECK_INTERVAL_SECONDS:
                s = check_session_and_relogin(s)
                last_session_check = time.time()
            try:
                simulation_progress_response = _resilient_get(s, simulation_progress_url)
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Simulation {simulation_progress_url} failed after exhausting transport-level retries: {e}"
                )
                error_flag = True
                final_status = "CONNECTION_ERROR"
                break
            if simulation_progress_response.status_code // 100 != 2:
                logger.error(
                    f'Simulation {simulation_progress_url}, Status code: {simulation_progress_response.status_code}, Retry'
                )
                time.sleep(30)
                retry_count += 1
                if retry_count <= 2:
                    continue
                else:
                    logger.error(
                        f'Simulation {simulation_progress_url} failed, Status code: {simulation_progress_response.status_code}'
                    )
                    error_flag = True
                    final_status = "HTTP_ERROR"
                    break
            if simulation_progress_response.headers.get("Retry-After", 0) == 0:
                status = _safe_json(simulation_progress_response).get("status", "ERROR")
                final_status = status
                # See simulation_progress() for rationale: treat CANCELLED/TIMEOUT/FAIL as
                # distinct failure statuses rather than lumping everything under "ERROR".
                if status not in SIMULATION_SUCCESS_STATUSES:
                    error_flag = True
                    if status not in SIMULATION_FAIL_STATUSES and status not in SIMULATION_IN_PROGRESS_STATUSES:
                        logger.warning(
                            f"Simulation returned an unrecognized terminal status '{status}' - treating as a "
                            "failure (fail-closed) rather than assuming success."
                        )
                break

            if progress_bar is not None:
                body = _safe_json(simulation_progress_response)
                fraction = body.get("progress")
                if isinstance(fraction, (int, float)):
                    pct = min(max(fraction * 100, 0), 100)
                    progress_bar.n = pct
                    progress_bar.refresh()

            time.sleep(float(simulation_progress_response.headers["Retry-After"]))
    finally:
        if progress_bar is not None:
            progress_bar.n = 100 if not error_flag else progress_bar.n
            progress_bar.refresh()
            progress_bar.close()

    children = _safe_json(simulation_progress_response).get("children", 0)

    if error_flag:
        if children == 0:
            logger.error(f"Simulation failed with status={final_status}. {_safe_json(simulation_progress_response)}")
            return {"completed": False, "result": {}, "status": final_status}
        for child in children:
            child_progress = _resilient_get(s, brain_api_url + "/simulations/" + child)
            child_body = _safe_json(child_progress)
            expression = _extract_expression(child_body)
            logger.error(f"Child Simulation failed (id={child}, expression={expression}): {child_body}")
        return {"completed": False, "result": {}, "status": final_status}

    if len(children) == 0:
        logger.warning(
            f'Multi-Simulation {_safe_json(simulation_progress_response).get("id")} failed. '
            f'{_safe_json(simulation_progress_response)}'
        )
        return {"completed": False, "result": {}, "status": final_status}
    children_list = []
    for child in children:
        child_progress = _resilient_get(s, brain_api_url + "/simulations/" + child)
        child_body = _safe_json(child_progress)
        alpha = child_body.get("alpha", 0)
        if alpha == 0:
            expression = _extract_expression(child_body)
            logger.warning(
                f'Child-Simulation {child_body.get("id")} failed (expression={expression}). {child_body}'
            )
            return {"completed": False, "result": {}, "status": final_status}
        child_result = get_simulation_result_json(s, alpha)
        children_list.append(child_result)
    return {"completed": True, "result": children_list, "status": final_status}


def get_prod_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the production correlation data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the production correlation data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id + "/correlations/prod")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get production correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    prod_corr_df["alpha_max_prod_corr"] = result.json()["max"]
    prod_corr_df["alpha_min_prod_corr"] = result.json()["min"]

    return prod_corr_df


def check_prod_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    Check if the alpha's production correlation passes a specified threshold.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        threshold (float, optional): The correlation threshold. Defaults to 0.7.

    Returns:
        pandas.DataFrame: A DataFrame containing the test result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    prod_corr_df = get_prod_corr(s, alpha_id)
    if prod_corr_df.empty:
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "NONE",
                "limit": threshold,
                "value": None,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "PASS" if value <= threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)


def get_self_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the self-correlation data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the self-correlation data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id + "/correlations/self")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get self correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()

    records_len = len(result.json()["records"])
    if records_len == 0:
        logger.warning(f"No self correlation for alpha_id {alpha_id}")
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    self_corr_df["alpha_max_self_corr"] = result.json()["max"]
    self_corr_df["alpha_min_self_corr"] = result.json()["min"]

    return self_corr_df


def check_self_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    Check if the alpha's self-correlation passes a specified threshold.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        threshold (float, optional): The correlation threshold. Defaults to 0.7.

    Returns:
        pandas.DataFrame: A DataFrame containing the test result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS",
                "limit": threshold,
                "value": 0,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)


def get_check_submission(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the submission check results for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the submission check results.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        logger.warning(f"Cant check submission alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()

    checks_df = pd.DataFrame(result.json()["is"]["checks"]).assign(alpha_id=alpha_id)

    return checks_df


def simulate_multi_alpha(
    s: SingleSession,
    simulate_data_list: list,
) -> list[dict]:
    """
    Simulate a list of alphas using multi-simulation.

    This function checks the session timeout, starts a new session if necessary,
    initiates the simulation, monitors its progress, and sets alpha properties
    upon completion.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A list of dictionaries, each containing the simulation parameters for the alpha.
            These should include all necessary information such as alpha type, settings, and expressions.

    Returns:
        list: A list of dictionaries, each containing:
            - 'alpha_id' (str): The ID of the simulated alpha if successful, None otherwise.
            - 'simulate_data' (dict): The original simulation data provided.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """

    s = check_session_and_relogin(s)
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)

    if not simulation_result["completed"]:
        return [{"alpha_id": None, "simulate_data": x} for x in simulate_data_list]
    result = [
        {
            "alpha_id": x["id"],
            "simulate_data": {
                "type": x["type"],
                "settings": x["settings"],
                "regular": x["regular"]["code"],
            },
        }
        for x in simulation_result["result"]
    ]
    # _ = [set_alpha_properties(s, x["id"]) for x in simulation_result["result"]]
    return result


def _simulate_single_alpha_persistent(
    s: SingleSession,
    simulate_data: dict,
    retry_interval: float = 30.0,
    max_wait_seconds: Optional[float] = None,
    show_progress: bool = False,
    on_wait_state_change: Optional[Callable[[bool], None]] = None,
) -> dict:
    """
    Single-alpha counterpart to simulate_multi_alpha_persistent, used for the edge
    case where a queue batch ends up with exactly one alpha in it (e.g. a list length
    not evenly divisible by limit_of_multi_simulations). Same contract as
    simulate_single_alpha, but submits via _start_simulation_persistent so it waits
    at a fixed interval for a slot rather than giving up after a few minutes.
    """
    s = check_session_and_relogin(s)
    simulate_response = _start_simulation_persistent(
        s, simulate_data, retry_interval, max_wait_seconds, on_wait_state_change=on_wait_state_change
    )
    simulation_result = simulation_progress(s, simulate_response, show_progress=show_progress)
    if not simulation_result["completed"]:
        return {"alpha_id": None, "simulate_data": simulate_data}
    return {
        "alpha_id": simulation_result["result"]["id"],
        "simulate_data": simulate_data,
    }


def simulate_multi_alpha_persistent(
    s: SingleSession,
    simulate_data_list: list,
    retry_interval: float = 30.0,
    max_wait_seconds: Optional[float] = None,
    show_progress: bool = False,
    on_wait_state_change: Optional[Callable[[bool], None]] = None,
) -> list[dict]:
    """
    Same contract as simulate_multi_alpha, but submits via _start_simulation_persistent
    so a CONCURRENT_SIMULATION_LIMIT_EXCEEDED rejection waits at a fixed interval for a
    slot to free up - by default for as long as it takes - instead of giving up after a
    handful of bounded exponential-backoff attempts. Used by simulate_alpha_queue's
    worker threads, where waiting out a slot on a long-running batch is the whole point.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data_list (list): A list of dictionaries, each containing the
            simulation parameters for one alpha in the batch.
        retry_interval (float, optional): Fixed seconds between retries while waiting
            for a slot. Defaults to 30.0.
        max_wait_seconds (float, optional): If set, give up waiting for a slot after
            this many seconds. Defaults to None (wait indefinitely).
        show_progress (bool, optional): Show a live progress bar for this batch's
            simulation once it starts running. Defaults to False.
        on_wait_state_change (callable, optional): Called with True/False as this
            batch starts/stops waiting for a submission slot. See
            _start_simulation_persistent for details.

    Returns:
        list: A list of dictionaries, each containing 'alpha_id' and 'simulate_data',
            same shape as simulate_multi_alpha's return value.
    """
    s = check_session_and_relogin(s)
    if len(simulate_data_list) == 1:
        return [
            _simulate_single_alpha_persistent(
                s,
                simulate_data_list[0],
                retry_interval,
                max_wait_seconds,
                show_progress=show_progress,
                on_wait_state_change=on_wait_state_change,
            )
        ]
    simulate_response = _start_simulation_persistent(
        s, simulate_data_list, retry_interval, max_wait_seconds, on_wait_state_change=on_wait_state_change
    )
    simulation_result = multisimulation_progress(s, simulate_response, show_progress=show_progress)

    if not simulation_result["completed"]:
        return [{"alpha_id": None, "simulate_data": x} for x in simulate_data_list]
    result = [
        {
            "alpha_id": x["id"],
            "simulate_data": {
                "type": x["type"],
                "settings": x["settings"],
                "regular": x["regular"]["code"],
            },
        }
        for x in simulation_result["result"]
    ]
    return result


def _infer_max_concurrent_multisims(alpha_list: list) -> int:
    """
    Best-effort inference of the account's real concurrent multi-simulation cap, based
    on region.

    Per WQB support (GLB concurrency update announcement), the GLB region has
    different slot accounting than other regions: each GLB alpha simulation consumes
    2 of the account's 8 total concurrency slots, effectively capping GLB at 4
    concurrent multi-simulations rather than the standard 8. This is a heuristic, not
    a platform guarantee - WQB could revise slot costs for other regions too. A list
    mixing GLB with other regions is treated conservatively (as if entirely GLB),
    since under-driving concurrency just costs a bit of parallelism, while
    over-driving it recreates the exact retry-storm this inference exists to avoid.

    Args:
        alpha_list (list): The alpha configurations about to be queued.

    Returns:
        int: The inferred safe concurrent multi-simulation cap (4 if any alpha in the
            list targets GLB, otherwise 8).
    """
    regions = {a.get("settings", {}).get("region") for a in alpha_list if isinstance(a, dict)}
    if "GLB" in regions:
        return 4
    return 8


def simulate_alpha_queue(
    s: SingleSession,
    alpha_list: list,
    limit_of_multi_simulations: int = 10,
    num_workers: Optional[int] = None,
    retry_interval_seconds: float = 30.0,
    max_wait_seconds: Optional[float] = None,
    simulation_config: Optional[dict] = None,
    show_progress: bool = True,
    show_batch_progress: bool = False,
    status_file: Optional[str] = "ace_queue_status.json",
) -> list:
    """
    Queue-based alternative to simulate_alpha_list_multi that removes the need to tune
    a concurrency limit up front.

    Instead of a fixed-size ThreadPool that submits N batches at once and gives up
    quickly on CONCURRENT_SIMULATION_LIMIT_EXCEEDED, this puts every batch into a
    shared queue and spins up a pool of worker threads that continuously pull from it.
    Whenever a worker gets rejected for exceeding the account's concurrent
    multi-simulation limit, it simply waits retry_interval_seconds (with jitter) and
    tries again - indefinitely by default - rather than failing that batch. This suits
    multi-simulations well, since a full batch typically takes 10-40 minutes to run - a
    bounded exponential backoff (a few minutes total, as used elsewhere in this module)
    isn't long enough to wait out a slot on batches that size, but a fixed-interval
    indefinite retry is.

    Note: multi-simulation is REGULAR-only (same restriction as
    simulate_alpha_list_multi's underlying platform behavior). Unlike
    simulate_alpha_list_multi, this raises ValueError on any SUPER alpha in the list
    rather than silently falling back to single-alpha simulation, since silently
    switching architectures partway through a queue run would be confusing.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_list (list): A list of alpha configurations to simulate (REGULAR only).
        limit_of_multi_simulations (int, optional): Alphas per multi-simulation batch,
            1..10 per the BRAIN API docs. Defaults to 10.
        num_workers (int, optional): Number of worker threads pulling from the queue.
            If not specified, this is inferred from alpha_list's region: 4 if any
            alpha targets GLB (per WQB's revised GLB concurrency slot cost - each GLB
            simulation uses 2 of the account's 8 slots), otherwise 8. Setting this
            higher than the real cap doesn't increase throughput - extra workers just
            wait their turn - but it does mean more wasted rejected-submission retries
            and log noise while they wait, so the inferred default deliberately matches
            the real cap rather than padding it. Pass this explicitly to override the
            inference (e.g. if you know your actual cap differs).
        retry_interval_seconds (float, optional): Base seconds between retries when a
            worker is rejected for exceeding the concurrency limit (actual wait is
            jittered to 0.7x-1.3x this value per attempt, to desynchronize workers).
            Defaults to 30.0.
        max_wait_seconds (float, optional): If set, a worker gives up on its current
            batch after waiting this long total for a slot (that batch's alphas are
            then marked failed, same as an ordinary submission failure, and can be
            re-queued by resubmitting the corresponding slice of alpha_list). Defaults
            to None (wait indefinitely - recommended, since giving up just means the
            batch needs to be resubmitted later anyway).
        simulation_config (dict, optional): Passed through to get_specified_alpha_stats
            for each batch's stats-fetching pass, which now runs immediately after that
            batch finishes (live), not after every batch in the whole run completes.
            Same semantics as simulate_alpha_list_multi otherwise. Defaults to None,
            which resolves to whatever ace.DEFAULT_CONFIG currently is at call time
            (so reassigning ace.DEFAULT_CONFIG before calling this is honored, unlike
            passing it as a literal function-default value would be).
        show_progress (bool, optional): Show an overall progress bar tracking how many
            batches have completed. Defaults to True.
        show_batch_progress (bool, optional): Also show each running batch's own
            live percentage (from BRAIN's per-simulation 'progress' field), on top of
            the overall batches-completed bar. Off by default: with several workers
            running concurrently, each rendering its own progress updates, output in
            a plain-text/log-capturing notebook can interleave into noisy repeated
            lines rather than a single clean in-place bar (unlike a real terminal,
            where tqdm overwrites in place). Turn on if you want that finer-grained
            visibility and don't mind the extra output.
        status_file (str, optional): Path to a JSON file this function keeps updated
            with live queue status (batches total/completed, workers busy/waiting,
            elapsed/estimated remaining time, recent batch-completion events), for the
            companion ace_dashboard.html to poll and render in a browser tab instead
            of reading raw log output. Defaults to "ace_queue_status.json" in the
            current working directory. Set to None to disable.

    Returns:
        list: A list of dictionaries containing simulation results for each alpha,
            same shape as simulate_alpha_list_multi's return value.

    Raises:
        ValueError: If any alpha in alpha_list is type SUPER.
    """
    if not alpha_list:
        return []

    if simulation_config is None:
        # Looked up here (call-time), not as the parameter's default value
        # (def-time), so that reassigning ace.DEFAULT_CONFIG to a new dict later
        # (e.g. ace.DEFAULT_CONFIG = {**ace.DEFAULT_CONFIG, "check_self_corr": False})
        # is actually honored by calls that don't pass simulation_config explicitly.
        # A def-time default would keep silently pointing at the original dict object
        # from import time no matter what ace.DEFAULT_CONFIG gets reassigned to.
        simulation_config = DEFAULT_CONFIG

    if any(d["type"] == "SUPER" for d in alpha_list):
        raise ValueError(
            "simulate_alpha_queue does not support SUPER alphas (multi-simulation is REGULAR-only). "
            "Use simulate_alpha_list for SUPER alphas instead."
        )

    if num_workers is None:
        num_workers = _infer_max_concurrent_multisims(alpha_list)
        logger.info(
            f"num_workers not specified - inferred a concurrent multi-simulation cap of {num_workers} "
            "based on region (GLB effectively allows 4 due to WQB's revised GLB concurrency slot cost; "
            "other regions allow the standard 8). Pass num_workers explicitly to override."
        )

    if (limit_of_multi_simulations < 1) or (limit_of_multi_simulations > 10):
        logger.warning("Limit of multi-simulation should be 1..10, will be set to 10")
        limit_of_multi_simulations = 10

    tasks = [
        alpha_list[i : i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)
    ]

    num_workers = max(1, min(num_workers, len(tasks)))

    task_queue: "queue.Queue[list]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    results_lock = threading.Lock()
    stats_list_result: list = []
    progress_lock = threading.Lock()
    progress_bar = tqdm.tqdm(total=len(tasks), desc="Batches completed") if show_progress else None

    def _fetch_and_save_stats(x):
        try:
            return get_specified_alpha_stats(s, x["alpha_id"], x["simulate_data"], **simulation_config)
        except Exception as e:
            # Same rationale as the try/except inside get_specified_alpha_stats: this
            # runs inside stats_pool.map(), which aborts and discards every other
            # alpha's already-fetched stats in that call the instant any single call
            # raises. Anything unexpected slipping past get_specified_alpha_stats's own
            # handling (a non-RequestException from a helper, a malformed result body
            # it didn't anticipate, etc.) must not be allowed to take the rest of this
            # batch's stats down with it.
            logger.error(f"Unexpected error fetching stats for alpha_id {x['alpha_id']}: {e}")
            return {
                "alpha_id": x["alpha_id"],
                "simulate_data": x["simulate_data"],
                "is_stats": None,
                "pnl": None,
                "stats": None,
                "is_tests": None,
                "train": None,
                "test": None,
            }

    # Shared, bounded (3 concurrent) pool for fetching+saving each alpha's stats.
    # Every worker submits its own batch to this same pool as soon as that batch's
    # simulations finish - not after every other batch in the whole queue has also
    # finished - so results land on disk (via get_specified_alpha_stats's save_*_file
    # flags) live, batch by batch, throughout the run instead of all at once at the
    # very end. Kept at 3 concurrent regardless of num_workers so a run with many
    # workers doesn't fan this out into dozens of simultaneous stats-fetch calls.
    stats_pool = ThreadPool(3)

    # --- Live status tracking for the companion ace_dashboard.html ---
    run_started_at = time.time()
    status_lock = threading.RLock()
    worker_states = {i: "idle" for i in range(num_workers)}
    succeeded_batches = 0
    failed_batches = 0
    recent_events: list = []
    max_recent_events = 25
    run_finished = False

    def _record_event(message: str) -> None:
        with status_lock:
            recent_events.append({"time": time.time(), "message": message})
            if len(recent_events) > max_recent_events:
                del recent_events[: len(recent_events) - max_recent_events]

    def _write_queue_status() -> None:
        if not status_file:
            return
        with status_lock:
            completed = succeeded_batches + failed_batches
            elapsed = time.time() - run_started_at
            avg_per_batch = elapsed / completed if completed > 0 else None
            remaining = len(tasks) - completed
            # Rough steady-state estimate: average wall-clock time per completed batch
            # so far, projected across the remaining batches and divided by however many
            # workers are actually running right now (not waiting). A basic estimate,
            # not a guarantee - actual time depends on how many slots free up and when.
            running_now = sum(1 for v in worker_states.values() if v == "running") or 1
            eta_seconds = (avg_per_batch * remaining / running_now) if avg_per_batch else None
            payload = {
                "total_batches": len(tasks),
                "completed_batches": completed,
                "succeeded_batches": succeeded_batches,
                "failed_batches": failed_batches,
                "num_workers": num_workers,
                "workers_running": sum(1 for v in worker_states.values() if v == "running"),
                "workers_waiting": sum(1 for v in worker_states.values() if v == "waiting"),
                "workers_idle": sum(1 for v in worker_states.values() if v == "idle"),
                "worker_states": [worker_states[i] for i in range(num_workers)],
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
                "recent_events": list(reversed(recent_events)),
                "running": not run_finished,
                "updated_at": time.time(),
            }
        _write_status_json(status_file, payload)

    def _worker(worker_id: int) -> None:
        def _on_wait_state_change(waiting: bool) -> None:
            with status_lock:
                worker_states[worker_id] = "waiting" if waiting else "running"
            _write_queue_status()

        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                with status_lock:
                    worker_states[worker_id] = "idle"
                _write_queue_status()
                return
            with status_lock:
                worker_states[worker_id] = "running"
            try:
                batch_result = simulate_multi_alpha_persistent(
                    s,
                    task,
                    retry_interval=retry_interval_seconds,
                    max_wait_seconds=max_wait_seconds,
                    show_progress=show_batch_progress,
                    on_wait_state_change=_on_wait_state_change,
                )
            except Exception as e:
                logger.error(f"Worker failed on a batch of {len(task)} alphas: {e}")
                batch_result = [{"alpha_id": None, "simulate_data": x} for x in task]

            batch_succeeded = sum(1 for r in batch_result if r["alpha_id"] is not None)
            batch_failed = len(batch_result) - batch_succeeded

            # Fetch + save stats for this batch right now, while other workers keep
            # pulling the next task off the queue - this is what makes saving "live"
            # rather than deferred until every batch in the whole run has finished.
            batch_stats = stats_pool.map(_fetch_and_save_stats, batch_result)
            with results_lock:
                stats_list_result.extend(batch_stats)

            nonlocal succeeded_batches, failed_batches
            with status_lock:
                if batch_failed == 0:
                    succeeded_batches += 1
                    _record_event(f"Batch completed: {batch_succeeded}/{len(batch_result)} alphas succeeded")
                else:
                    failed_batches += 1
                    _record_event(
                        f"Batch finished with issues: {batch_succeeded}/{len(batch_result)} alphas succeeded"
                    )
            if progress_bar is not None:
                with progress_lock:
                    progress_bar.update(1)
            _write_queue_status()
            task_queue.task_done()

    _write_queue_status()

    def _heartbeat() -> None:
        # Keeps the status file's elapsed/ETA fresh even during long stretches with no
        # batch completions (a single multi-simulation can run 10-40 minutes), so the
        # dashboard doesn't look frozen/stale in between events.
        while not run_finished:
            _write_queue_status()
            time.sleep(3)

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    if status_file:
        heartbeat_thread.start()

    workers = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(num_workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    run_finished = True
    _write_queue_status()

    if progress_bar is not None:
        progress_bar.close()

    stats_pool.close()
    stats_pool.join()

    return _delete_duplicates_from_result(stats_list_result)


def get_specified_alpha_stats(
    s: SingleSession,
    alpha_id: Union[str, None],
    simulate_data: dict,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    save_is_tests_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
) -> dict:
    """
    Retrieve and process specified statistics for a given alpha.

    Args:
        s (SingleSession): The authenticated session object.
        alpha_id (str): The ID of the alpha to retrieve statistics for.
        simulate_data (dict): The original simulation data for the alpha.
        get_pnl (bool, optional): Whether to retrieve PnL data. Defaults to False.
        get_stats (bool, optional): Whether to retrieve yearly stats. Defaults to False.
        save_pnl_file (bool, optional): Whether to save PnL data to a file. Defaults to False.
        save_stats_file (bool, optional): Whether to save yearly stats to a file. Defaults to False.
        save_result_file (bool, optional): Whether to save the simulation result to a file. Defaults to False.
        save_is_tests_file (bool, optional): Whether to save the check results (submission
            checks and/or self/prod-correlation checks) to a file via save_is_tests.
            Defaults to False.
        check_submission (bool, optional): Whether to fetch the platform's submission
            checks (via get_check_submission) and use them as the base is_tests.
            Defaults to False.
        check_self_corr (bool, optional): Whether to additionally run a client-side
            self-correlation check (check_self_corr_test) and merge it into is_tests,
            overwriting any existing SELF_CORRELATION row from check_submission's
            result. Runs independently of check_submission. Defaults to False.
        check_prod_corr (bool, optional): Whether to additionally run a client-side
            production-correlation check (check_prod_corr_test) and merge it into
            is_tests, overwriting any existing PROD_CORRELATION row from
            check_submission's result. Runs independently of check_submission.
            Defaults to False.

    Returns:
        dict: A dictionary containing various statistics and information about the alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error retrieving data from the API.
    """
    pnl = None
    stats = None
    s = check_session_and_relogin(s)
    logger.debug(f"Session (ID: {id(s)}) used in get_specified_alpha_stats for alpha_id: {alpha_id}")
    if alpha_id is None:
        return {
            "alpha_id": None,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": pnl,
            "stats": stats,
            "is_tests": None,
            "train": None,
            "test": None,
        }

    try:
        result = get_simulation_result_json(s, alpha_id)
    except requests.exceptions.RequestException as e:
        # The core result fetch exhausted its transport-level retries. Rather than
        # letting this propagate and crash the whole ThreadPool batch (losing every
        # other alpha's already-fetched stats), degrade gracefully: keep the
        # alpha_id so it can be re-fetched later (e.g. via get_specified_alpha_stats
        # directly, without re-simulating), and leave the stats fields empty/None so
        # prettify_result's existing "is_stats is not None" filter just skips it.
        logger.error(
            f"Could not retrieve simulation result for alpha_id {alpha_id} after transport-level retries "
            f"were exhausted: {e}. This alpha was simulated successfully but its stats could not be fetched "
            "right now - alpha_id is preserved so it can be re-fetched later."
        )
        return {
            "alpha_id": alpha_id,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": pnl,
            "stats": stats,
            "is_tests": None,
            "train": None,
            "test": None,
        }
    try:
        region = result["settings"]["region"]
        train = result["train"]
        test = result["test"]
        is_stats = pd.DataFrame([{key: value for key, value in result["is"].items() if key != "checks"}]).assign(
            alpha_id=alpha_id
        )
        is_tests = pd.DataFrame(result["is"]["checks"]).assign(alpha_id=alpha_id)
    except Exception as e:
        # A malformed/partial result body here (e.g. from a response caught mid-relogin,
        # or an alpha that errored after all) must not raise: this function runs inside
        # pool.map() in simulate_alpha_queue, and pool.map() aborts and discards every
        # other already-computed alpha's stats in the same call the moment any single
        # call raises. Degrading this one alpha to an empty result (same shape as the
        # network-error case above) keeps the rest of the batch's results intact.
        logger.error(f"Failed to parse simulation result for alpha_id {alpha_id}: {result}, {e}")
        return {
            "alpha_id": alpha_id,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": pnl,
            "stats": stats,
            "is_tests": None,
            "train": None,
            "test": None,
        }

    if get_pnl:
        try:
            pnl = get_alpha_pnl(s, alpha_id)
            if save_pnl_file:
                save_pnl(pnl, alpha_id, region)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch PnL for alpha_id {alpha_id} (network error, skipping): {e}")

    if get_stats:
        try:
            stats = get_alpha_yearly_stats(s, alpha_id)
            if save_stats_file:
                save_yearly_stats(stats, alpha_id, region)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch yearly stats for alpha_id {alpha_id} (network error, skipping): {e}")

    if save_result_file:
        save_simulation_result(result)

    if check_submission:
        try:
            is_tests = get_check_submission(s, alpha_id)
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Could not fetch submission check for alpha_id {alpha_id} (network error) - falling back to "
                f"in-sample checks from the simulation result instead: {e}"
            )

    if check_self_corr:
        try:
            self_corr_test = check_self_corr_test(s, alpha_id)
            # keep="last": a self-correlation row already present (e.g. from
            # get_check_submission's own SELF_CORRELATION check, at whatever
            # threshold WQB uses) is overwritten by this client-side check, run at
            # the fixed threshold in check_self_corr_test.
            is_tests = (
                pd.concat([is_tests, self_corr_test], ignore_index=True)
                .drop_duplicates(subset=["test"], keep="last")
                .reset_index(drop=True)
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not check self-correlation for alpha_id {alpha_id} (network error, skipping): {e}")
    if check_prod_corr:
        try:
            prod_corr_test = check_prod_corr_test(s, alpha_id)
            # Same keep="last" overwrite behavior as check_self_corr above, for
            # PROD_CORRELATION.
            is_tests = (
                pd.concat([is_tests, prod_corr_test], ignore_index=True)
                .drop_duplicates(subset=["test"], keep="last")
                .reset_index(drop=True)
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not check prod correlation for alpha_id {alpha_id} (network error, skipping): {e}")

    if save_is_tests_file:
        save_is_tests(is_tests, alpha_id, region)

    return {
        "alpha_id": alpha_id,
        "simulate_data": simulate_data,
        "is_stats": is_stats,
        "pnl": pnl,
        "stats": stats,
        "is_tests": is_tests,
        "train": train,
        "test": test,
    }


def simulate_single_alpha(
    s: SingleSession,
    simulate_data: dict,
    show_progress: bool = False,
) -> dict:
    """
    Simulate a single alpha using the provided session and simulation data.

    This function checks the session timeout, starts a new session if necessary,
    initiates the simulation, monitors its progress, and sets alpha properties
    upon completion.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A dictionary containing the simulation parameters for the alpha.
            This should include all necessary information such as alpha type, settings, and expressions.
        show_progress (bool, optional): If True, display a live progress bar while the
            simulation runs (see simulation_progress). Intended for direct/interactive
            calls; simulate_alpha_list/simulate_alpha_list_multi do not enable this for
            their threaded batch runs. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'alpha_id' (str): The ID of the simulated alpha if successful, None otherwise.
            - 'simulate_data' (dict): The original simulation data provided.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """

    s = check_session_and_relogin(s)
    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response, show_progress=show_progress)

    if not simulation_result["completed"]:
        return {"alpha_id": None, "simulate_data": simulate_data}
    return {
        "alpha_id": simulation_result["result"]["id"],
        "simulate_data": simulate_data,
    }


def simulate_alpha_list(
    s: SingleSession,
    alpha_list: list,
    limit_of_concurrent_simulations: int = 3,
    simulation_config: dict = DEFAULT_CONFIG,
) -> list:
    """
    Simulate a list of alphas concurrently.

    Args:
        s (SingleSession): The authenticated session object.
        alpha_list (list): A list of alpha configurations to simulate.
        limit_of_concurrent_simulations (int, optional): The maximum number of concurrent simulations. Defaults to 3.
        simulation_config (dict, optional): Configuration for the simulation. Defaults to DEFAULT_CONFIG.

    Returns:
        list: A list of dictionaries containing simulation results for each alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error during the simulation process.
    """
    if (limit_of_concurrent_simulations < 1) or (limit_of_concurrent_simulations > 8):
        logger.warning("Limit of concurrent simulation should be 1..8, will be set to 3")
        limit_of_concurrent_simulations = 3

    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            for result in pool.imap_unordered(partial(simulate_single_alpha, s), alpha_list):
                result_list.append(result)
                pbar.update()

    stats_list_result = []

    def func(x):
        return get_specified_alpha_stats(s, x["alpha_id"], x["simulate_data"], **simulation_config)

    with ThreadPool(3) as pool:
        for result in pool.map(func, result_list):
            stats_list_result.append(result)

    return _delete_duplicates_from_result(stats_list_result)


def simulate_alpha_list_multi(
    s: SingleSession,
    alpha_list: list,
    limit_of_concurrent_simulations: int = 3,
    limit_of_multi_simulations: int = 10,
    simulation_config: dict = DEFAULT_CONFIG,
) -> list:
    """
    Simulate a list of alphas using multi-simulation when possible.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_list (list): A list of alpha configurations to simulate.
        limit_of_concurrent_simulations (int, optional): The maximum number of concurrent simulation batches. Defaults to 3.
        limit_of_multi_simulations (int, optional): The maximum number of alphas in a multi-simulation. Defaults to 3.
        simulation_config (dict, optional): Configuration for the simulation. Defaults to DEFAULT_CONFIG.

    Returns:
        list: A list of dictionaries containing simulation results for each alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if (limit_of_multi_simulations < 2) or (limit_of_multi_simulations > 10):
        logger.warning("Limit of multi-simulation should be 2..10, will be set to 10")
        limit_of_multi_simulations = 10
    if (limit_of_concurrent_simulations < 1) or (limit_of_concurrent_simulations > 8):
        logger.warning("Limit of concurrent simulation should be 1..8, will be set to 3")
        limit_of_concurrent_simulations = 3
    if any(d["type"] == "SUPER" for d in alpha_list):
        logger.warning("Multi-Simulation is not supported for SuperAlphas, single concurrent simulations will be used")
        return simulate_alpha_list(
            s,
            alpha_list,
            limit_of_concurrent_simulations=3,
            simulation_config=simulation_config,
        )

    tasks = [
        alpha_list[i : i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)
    ]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        with tqdm.tqdm(total=len(tasks)) as pbar:
            for result in pool.imap_unordered(partial(simulate_multi_alpha, s), tasks):
                result_list.append(result)
                pbar.update()
    result_list_flat = [item for sublist in result_list for item in sublist]

    stats_list_result = []

    def func(x):
        return get_specified_alpha_stats(s, x["alpha_id"], x["simulate_data"], **simulation_config)

    with ThreadPool(3) as pool:
        for result in pool.map(func, result_list_flat):
            stats_list_result.append(result)

    return _delete_duplicates_from_result(stats_list_result)


def _delete_duplicates_from_result(result: list) -> list:
    """
    Remove duplicate alpha results from the simulation output.

    Args:
        result (list): A list of dictionaries containing simulation results.

    Returns:
        list: A deduplicated list of simulation results.
    """
    alpha_id_lst = []
    result_new = []
    for x in result:
        if x["alpha_id"] is not None:
            if x["alpha_id"] not in alpha_id_lst:
                result_new.append(x)
                alpha_id_lst.append(x["alpha_id"])
        else:
            result_new.append(x)
    return result_new


def set_alpha_properties(
    s: SingleSession,
    alpha_id: str,
    name: Union[str, object] = _UNSET,
    color: Union[str, object] = _UNSET,
    category: Union[Category, object] = _UNSET,
    regular_desc: Union[str, object] = _UNSET,
    selection_desc: Union[str, object] = _UNSET,
    combo_desc: Union[str, object] = _UNSET,
    osmosis_points: Union[int, object] = _UNSET,
    tags: Union[list[str], object] = _UNSET,
) -> requests.Response:
    """
    Update the properties of an alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha to update.
        name (str, optional): The new name for the alpha. If not ptovided - is not changed.
            If set to None - description is removed.
        color (str, optional): The new color for the alpha. If not ptovided - is not changed.
            If set to None - color is removed.
        category (str, optional): Alpha category. If not ptovided - is not changed.
            If set to None - category is removed.
        regular_desc (str, optional): Description for regular alpha. If not ptovided - is not changed.
            If set to None - description is removed.
        selection_desc (str, optional): Description for the selection part of a super alpha. If not ptovided - is not changed.
        combo_desc (str, optional): Description for the combo part of a super alpha. If not ptovided - is not changed.
        osmosis_points (int, optional): Osmosis points, int from 1 to 100_000. If not ptovided - is not changed.
            If set to None - points are removed.
        tags (list, optional): List of tags to apply to the alpha. If not ptovided - is not changed.
            If set to empty list - [] tags are removed.

    Returns:
        requests.Response: The response object from the API call.
    """

    if osmosis_points is not _UNSET and osmosis_points is not None:
        if not isinstance(osmosis_points, int):
            raise TypeError(f"osmosis_points must be int or None, got {type(osmosis_points)!r}")
        if not (1 <= osmosis_points <= 100_000):
            raise ValueError(f"osmosis_points must be between 1 and 100000, got {osmosis_points}")
    option_map = {
        "name": name,
        "color": color,
        "category": category,
        "tags": tags,
        "osmosisPoints": osmosis_points,
        "regular": {"description": regular_desc} if regular_desc is not _UNSET else _UNSET,
        "selection": {"description": selection_desc} if selection_desc is not _UNSET else _UNSET,
        "combo": {"description": combo_desc} if combo_desc is not _UNSET else _UNSET,
    }
    params = {k: v for k, v in option_map.items() if v is not _UNSET}
    response = s.patch(brain_api_url + "/alphas/" + alpha_id, json=params)

    return response


def _get_alpha_pnl(
    s: SingleSession,
    alpha_id: str,
    pnl_type: str = "pnl",
) -> pd.DataFrame:
    """
    Retrieve the PnL data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        pnl_type (str): 'pnl' to get cumulative pnl, 'daily-pnl' to get daily pnl.

    Returns:
        pandas.DataFrame: A DataFrame containing the PnL data for the alpha.
    """

    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id + f"/recordsets/{pnl_type}")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json()
    if pnl.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in pnl["schema"]["properties"]]
    pnl_df = (
        pd.DataFrame(pnl["records"], columns=columns)
        .assign(alpha_id=alpha_id, date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"))
        .set_index("date")
    )
    return pnl_df


def get_alpha_pnl(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the cumulative PnL data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the PnL data for the alpha.
    """

    return _get_alpha_pnl(s, alpha_id, "pnl")


def get_alpha_yearly_stats(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the yearly statistics for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the yearly statistics for the alpha.
    """

    while True:
        result = _resilient_get(s, brain_api_url + "/alphas/" + alpha_id + "/recordsets/yearly-stats")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    stats = result.json()

    if stats.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in stats["schema"]["properties"]]
    yearly_stats_df = pd.DataFrame(stats["records"], columns=columns).assign(alpha_id=alpha_id)
    return yearly_stats_df


def _check_rate_limit(response: requests.Response) -> None:
    """Sleep based on rate-limit headers."""

    header_keys = {
        "limit_minute": "x-ratelimit-limit-minute",
        "remaining_minute": "x-ratelimit-remaining-minute",
        "limit_second": "x-ratelimit-limit-second",
        "remaining_second": "x-ratelimit-remaining-second",
    }
    parsed = {}
    for key, header_name in header_keys.items():
        val = response.headers.get(header_name)
        if val is None:
            logger.warning(f"Failed to parse rate-limit values: missing header {header_name}")
            return
        try:
            parsed[key] = int(val)
        except (ValueError, TypeError):
            parsed[key] = 30
            logger.warning(f"Failed to parse rate-limit values: cannot convert {header_name}={val} to int")
            return
    logger.debug(
        f"""
        Rate limit:
        remaining_minute={parsed["remaining_minute"]},
        limit_minute={parsed["limit_minute"]};
        remaining_second={parsed["remaining_second"]},
        limit_second={parsed["limit_second"]}
        """
    )
    if parsed["remaining_second"] < 1:
        logger.debug(f"Status code: {response.status_code}, sleep 1 sec")
        time.sleep(1)
    if parsed["remaining_minute"] <= 1:
        logger.info(f"Rate limit {parsed['limit_minute']} reached (per minute). Sleeping for a minute...")
        time.sleep(60)


def get_datasets(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Retrieve available datasets based on specified parameters.

    Args:
        s (SingleSession): An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str, optional): The region. Defaults to "USA".
        delay (int, optional): The delay. Defaults to 1.
        universe (str, optional): The universe. Defaults to "TOP3000".
        theme (bool | None, optional):
            - True  -> return only datasets that are in a theme
            - False -> return only datasets that are not in a theme
            - None  -> ignore theme filter (all datasets)
          Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing information about available datasets.
    """
    url = (
        brain_api_url
        + "/data-sets?"
        + f"instrumentType={instrument_type}&region={region}&delay={delay}&universe={universe}"
    )
    if theme is not None:
        theme_str = "true" if theme else "false"
        url += f"&theme={theme_str}"
    result = _resilient_get(s, url)
    _check_rate_limit(result)
    datasets_df = pd.DataFrame(result.json()["results"])
    datasets_df = expand_dict_columns(datasets_df)
    return datasets_df


def get_datafields(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    search: str = "",
) -> pd.DataFrame:
    """
    Retrieve available datafields based on specified parameters.

    Args:
        s (SingleSession): An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str): The region. Defaults to "USA".
        delay (int): The delay. Defaults to 1.
        universe (str): The universe. Defaults to "TOP3000".
        search (str, optional): A search string to filter datafields. Defaults to "".

    Returns:
        pandas.DataFrame: A DataFrame containing information about available datafields.
    """

    base = (
        brain_api_url
        + "/data-fields?"
        + f"&instrumentType={instrument_type}"
        + f"&region={region}"
        + f"&delay={delay}"
        + f"&universe={universe}"
    )

    if len(search) == 0:
        logger.info(f"Getting fields for: region={region}, delay={delay}, universe={universe}")
        result = _resilient_get(s, base)
        logger.debug(f"Get datafields, status_code:{result.status_code}")
        _check_rate_limit(result)
        datafields = result.json()["results"]

    else:
        logger.info(
            f"Getting fields for: region={region}, delay={delay}, universe={universe}, search key word: {search}"
        )
        url_template = base + "&limit=50" + f"&search={search}" + "&offset=0"
        result = _resilient_get(s, url_template)
        _check_rate_limit(result)
        datafields = result.json()["results"]

    datafields_df = pd.DataFrame(datafields)
    datafields_df = expand_dict_columns(datafields_df)
    return datafields_df


def get_operators(s: SingleSession) -> pd.DataFrame:
    """
    Fetches and processes the list of operators from the WorldQuant Brain API.

    This function retrieves the operators from the provided session `s`,
    explodes the 'scope' column (which contains lists) into separate rows,
    and returns the resulting DataFrame.

    Args:
    s (SingleSession): An authenticated session object.

    Returns:
    pd.DataFrame: A DataFrame containing the operators with each scope entry
    as a separate row.
    """
    df = pd.DataFrame(_resilient_get(s, brain_api_url + "/operators").json())
    return df.explode('scope').reset_index(drop=True)


def get_instrument_type_region_delay(s: SingleSession) -> pd.DataFrame:
    """
    Retrieves and organizes instrument type, region, and delay data into a DataFrame.

    Parameters:
        s (SingleSession): The session object used for making the API call.

    Returns:
        df (pd.DataFrame): A DataFrame containing the instrument type, region, delay, universe, and neutralization data.

    The function fetches the settings options from the simulations endpoint and extracts the 'Instrument type',
    'Region', 'Universe', 'Delay', and 'Neutralization' data. It then organizes this data into a list of dictionaries,
    each containing the instrument type, region, delay, universe, and neutralization for a particular combination
    of instrument type, region, and delay. This list is then converted into a DataFrame and returned.
    """

    settings_options = s.options(brain_api_url + '/simulations').json()['actions']['POST']['settings']['children']
    data = [
        {settings_options[key]['label']: settings_options[key]['choices']}
        for key in settings_options.keys()
        if settings_options[key]['type'] == 'choice'
    ]

    instrument_type_data = {}
    region_data = {}
    universe_data = {}
    delay_data = {}
    neutralization_data = {}

    for item in data:
        if 'Instrument type' in item:
            instrument_type_data = item['Instrument type']
        elif 'Region' in item:
            region_data = item['Region']['instrumentType']
        elif 'Universe' in item:
            universe_data = item['Universe']['instrumentType']
        elif 'Delay' in item:
            delay_data = item['Delay']['instrumentType']
        elif 'Neutralization' in item:
            neutralization_data = item['Neutralization']['instrumentType']

    data_list = []

    for instrument_type in instrument_type_data:
        for region in region_data[instrument_type['value']]:
            for delay in delay_data[instrument_type['value']]['region'][region['value']]:
                row = {'InstrumentType': instrument_type['value'], 'Region': region['value'], 'Delay': delay['value']}
                row['Universe'] = [
                    item['value'] for item in universe_data[instrument_type['value']]['region'][region['value']]
                ]
                row['Neutralization'] = [
                    item['value'] for item in neutralization_data[instrument_type['value']]['region'][region['value']]
                ]
                data_list.append(row)

    df = (
        pd.DataFrame(data_list)
        .sort_values(
            by=['InstrumentType', 'Region', 'Delay'],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return df


def performance_comparison(
    s: SingleSession, alpha_id: str, team_id: Optional[str] = None, competition: Optional[str] = None
) -> dict:
    """
    Retrieve performance comparison data for merged performance check.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        team_id (str, optional): The ID of the team for comparison. Defaults to None.
        competition (str, optional): The ID of the competition for comparison. Defaults to None.

    Returns:
        dict: A dictionary containing the performance comparison data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    if competition is not None:
        part_url = f"competitions/{competition}"
    elif team_id is not None:
        part_url = f"teams/{team_id}"
    else:
        part_url = "users/self"
    while True:
        result = _resilient_get(s, brain_api_url + f"/{part_url}/alphas/" + alpha_id + "/before-and-after-performance")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("stats", 0) == 0:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}
    if result.status_code != 200:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}

    return result.json()