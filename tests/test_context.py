# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_sentinel

import unittest

from coreason_sentinel.utils import context


class TestContext(unittest.TestCase):
    def test_request_id_lifecycle(self) -> None:
        """Test setting and retrieving request ID."""
        req_id = "req-123"
        context.set_request_id(req_id)
        self.assertEqual(context.get_request_id(), req_id)

        # Verify isolation (basic check, though full async isolation needs async test)
        # Resetting context var in same context
        context.set_request_id("req-456")
        self.assertEqual(context.get_request_id(), "req-456")
