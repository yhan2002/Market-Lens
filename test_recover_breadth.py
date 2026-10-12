import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from recover_breadth import recover_prices


class RecoveryTest(unittest.TestCase):
    def test_failed_batch_retries_members_and_preserves_other_successes(self):
        with patch("recover_breadth.download_prices", side_effect=[RuntimeError(), (1, 1, ""), RuntimeError()]) as download:
            failed = recover_prices(Path("unused.db"), ("A", "B"), date(2026, 9, 9))
        self.assertEqual(failed, ["B"])
        self.assertEqual(download.call_args_list[1].kwargs["tickers"], ("A",))
        self.assertEqual(download.call_args_list[2].kwargs["require_date"], date(2026, 9, 9))
