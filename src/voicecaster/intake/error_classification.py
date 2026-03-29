# src/voicecaster/intake/error_classification.py
from __future__ import annotations

from typing import Any

import requests


NETWORK_ERROR_TYPES = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.TooManyRedirects,
)


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, NETWORK_ERROR_TYPES):
        return "network"
    if isinstance(exc, requests.exceptions.RequestException):
        return "network"
    return "unknown"


def classify_http_status(status_code: int) -> str:
    if 500 <= status_code <= 599:
        return "network"
    return "content"


def build_error_payload(
    *,
    message: str,
    error_type: str,
    exception: BaseException | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": message,
        "error_type": error_type,
    }
    if exception is not None:
        payload["exception_class"] = exception.__class__.__name__
        payload["exception_message"] = str(exception)
    if extra:
        payload["extra"] = extra
    return payload
