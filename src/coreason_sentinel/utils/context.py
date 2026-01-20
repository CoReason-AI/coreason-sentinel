# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from contextvars import ContextVar
from typing import Optional

# Context variable to store the Request ID
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Retrieves the current request ID from context."""
    return request_id_ctx.get()


def set_request_id(request_id: str) -> None:
    """Sets the request ID in the current context."""
    request_id_ctx.set(request_id)
