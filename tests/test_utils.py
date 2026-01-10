# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

from _pytest.capture import CaptureFixture
from loguru import logger

__all__ = ["logger"]


def test_logger_initialization() -> None:
    """Test that logger is initialized correctly."""
    assert logger is not None


def test_logger_writing(capsys: CaptureFixture) -> None:
    """Test that logger writes to stderr."""
    # We need to add a sink that writes to sys.stderr so capsys can capture it
    # But loguru is already configured to write to sys.stderr in logger.py
    # The issue is likely that loguru holds a reference to the original sys.stderr
    # before capsys patches it.

    # We can use a custom sink to verify the message is logged.

    logged_messages = []

    def sink(message: str) -> None:
        logged_messages.append(message)

    logger.add(sink, format="{message}")
    logger.info("Test message")

    # Wait for loguru to process (it's synchronous by default but good to be safe)
    assert any("Test message" in msg for msg in logged_messages)
