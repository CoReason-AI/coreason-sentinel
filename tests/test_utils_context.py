import unittest

from coreason_sentinel.utils.context import get_request_id, request_id_ctx, set_request_id


class TestUtilsContext(unittest.TestCase):
    def test_context_vars(self) -> None:
        # Default is None
        self.assertIsNone(get_request_id())

        # Set ID
        set_request_id("req-123")
        self.assertEqual(get_request_id(), "req-123")

        # Verify underlying contextvar
        self.assertEqual(request_id_ctx.get(), "req-123")
