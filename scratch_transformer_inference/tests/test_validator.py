import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "validate_predictions.py"
SPEC = importlib.util.spec_from_file_location("validate_predictions", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ValidatorTests(unittest.TestCase):
    def test_valid_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {"id": "a", "split": "test", "source": "A", "reference": "R"},
                    {"id": "b", "split": "test", "source": "B", "reference": "R"},
                ],
            )
            write_jsonl(
                predictions,
                [
                    {
                        "id": sample_id,
                        "system": "scratch_transformer",
                        "config_id": "beam4_lp1.1_nr3",
                        "prediction": f"summary {sample_id}",
                        "status": "ok",
                        "error": None,
                    }
                    for sample_id in ("a", "b")
                ],
            )
            summary, issues = validator.validate(
                manifest, predictions, "beam4_lp1.1_nr3"
            )
            self.assertEqual(issues, [])
            self.assertTrue(summary["valid_for_handoff"])
            self.assertEqual(summary["success"], 2)

    def test_duplicate_missing_and_failed_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {"id": "a", "source": "A"},
                    {"id": "b", "source": "B"},
                ],
            )
            error_record = {
                "id": "a",
                "system": "scratch_transformer",
                "config_id": "cfg",
                "prediction": "",
                "status": "error",
                "error": "mock failure",
            }
            write_jsonl(predictions, [error_record, error_record])
            summary, issues = validator.validate(manifest, predictions)
            self.assertTrue(issues)
            self.assertFalse(summary["valid_for_handoff"])
            self.assertEqual(summary["duplicate"], 1)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["failed"], 2)


if __name__ == "__main__":
    unittest.main()
