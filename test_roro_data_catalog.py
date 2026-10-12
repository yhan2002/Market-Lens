import csv
import tempfile
import unittest
from pathlib import Path

from roro_data_catalog import COLUMNS, SIGNALS, validate_catalog


def _rows() -> list[dict[str, str]]:
    rows = []
    for block, signal in SIGNALS:
        pit = "NOT_STRICT_PIT"
        status = "ACTIVE_SHADOW"
        if (block, signal) in set(SIGNALS[:4]) | set(SIGNALS[18:21]):
            pit = "CONDITIONAL_PRICE_NO_VINTAGE"
        if (block, signal) == SIGNALS[4]:
            pit = "PIT_CAUSAL"
        if (block, signal) in set(SIGNALS[8:18]):
            pit = "CONDITIONAL_EVENT_PIT"
        if (block, signal) in {SIGNALS[5], SIGNALS[6], SIGNALS[7]}:
            pit = status = "DATA_BLOCKED"
        row = {column: "fixture" for column in COLUMNS}
        row.update(block=block, signal=signal, rows="0", pit_status=pit, status=status)
        rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, str]], columns=COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class RoroDataCatalogTest(unittest.TestCase):
    def test_valid_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.csv"
            _write(path, _rows())
            self.assertEqual(len(validate_catalog(path)), 22)

    def test_header_order_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.csv"
            _write(path, _rows(), (*COLUMNS[1:], COLUMNS[0]))
            with self.assertRaisesRegex(ValueError, "columns or column order"):
                validate_catalog(path)

    def test_duplicate_or_missing_signal_is_rejected(self) -> None:
        rows = _rows()
        rows[-1] = rows[0].copy()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.csv"
            _write(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_catalog(path)

    def test_protected_pit_and_blocked_status_cannot_be_weakened(self) -> None:
        for index, field, value in ((5, "status", "ACTIVE"), (8, "pit_status", "PIT_CAUSAL")):
            rows = _rows()
            rows[index][field] = value
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "catalog.csv"
                _write(path, rows)
                with self.assertRaises(ValueError):
                    validate_catalog(path)

    def test_rows_must_be_non_negative_integer_or_blank(self) -> None:
        for value in ("-1", "2.5"):
            rows = _rows()
            rows[0]["rows"] = value
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "catalog.csv"
                _write(path, rows)
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    validate_catalog(path)


if __name__ == "__main__":
    unittest.main()
