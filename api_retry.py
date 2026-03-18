"""Shared retry logic for external API calls.

Provides disconnect-aware retrying: fast retries for server drops,
slower backoff for rate limits and overload.  Used by tts.py and
image_gen.py to avoid duplicating retry logic.
"""

import logging
import time

from pipeline_config import (
    MAX_RETRIES,
    RETRY_DELAYS_DISCONNECT,
    RETRY_DELAYS_OTHER,
)

log = logging.getLogger(__name__)

_DISCONNECT_KEYWORDS = ["disconnect", "reset", "eof", "broken pipe"]
_TRANSIENT_KEYWORDS = ["timeout", "connection", "unavailable", "502", "503", "504"]


def is_transient_error(error: Exception) -> tuple[bool, bool]:
    """Classify an exception as transient and/or a disconnect.

    Returns:
        (is_transient, is_disconnect) — both booleans.
    """
    err = str(error).lower()
    is_disconnect = any(k in err for k in _DISCONNECT_KEYWORDS)
    is_other = any(k in err for k in _TRANSIENT_KEYWORDS)
    return (is_disconnect or is_other, is_disconnect)


def retry_api_call(
    fn,
    *,
    description: str = "API call",
    max_retries: int = MAX_RETRIES,
    on_retry=None,
    recreate_client=None,
):
    """Execute ``fn()`` with retries on transient errors.

    Args:
        fn:               Callable that makes the API request and returns
                          the result.  Any exception inside fn() is caught
                          and evaluated for retryability.
        description:      Label for log messages (e.g. "TTS generation").
        max_retries:      Maximum number of retry attempts.
        on_retry:         Optional callback(attempt, wait_seconds, error_msg)
                          for UI progress updates.
        recreate_client:  Optional callable invoked on disconnect errors
                          to reset the HTTP client / connection pool.

    Returns:
        The return value of ``fn()``.

    Raises:
        The original exception if all retries are exhausted or the error
        is not transient.
    """
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            return fn()
        except Exception as e:
            elapsed = time.time() - t0
            transient, disconnect = is_transient_error(e)

            if not transient or attempt >= max_retries:
                log.error(
                    "%s failed after %.1fs (attempt %d/%d, giving up): %s: %s",
                    description, elapsed, attempt + 1, max_retries + 1,
                    type(e).__name__, e,
                )
                raise

            delays = RETRY_DELAYS_DISCONNECT if disconnect else RETRY_DELAYS_OTHER
            wait = delays[min(attempt, len(delays) - 1)]

            log.warning(
                "%s attempt %d/%d failed after %.1fs (%s): %s. Retrying in %ds…",
                description, attempt + 1, max_retries + 1, elapsed,
                "disconnect" if disconnect else "transient", e, wait,
            )

            if on_retry:
                on_retry(attempt, wait, str(e))

            time.sleep(wait)

            if disconnect and recreate_client:
                log.info("%s: recreating client after disconnect", description)
                recreate_client()
