# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _logger

if TYPE_CHECKING:
    from loguru import Logger

logger: "Logger" = _logger

# Remove default handler
logger.remove()

# Sink 1: Stdout (Human-readable)
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# Ensure logs directory exists
log_path = Path("logs")
if not log_path.exists():  # pragma: no cover
    log_path.mkdir(parents=True, exist_ok=True)

# Sink 2: File (JSON, Rotation, Retention)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    serialize=True,
    enqueue=True,
    level="INFO",
)

__all__ = ["logger"]
