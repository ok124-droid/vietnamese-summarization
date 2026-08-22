import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
MODULE_PATH = PACKAGE_ROOT / "src" / "infer.py"
SPEC = importlib.util.spec_from_file_location("scratch_infer", MODULE_PATH)
assert SPEC and SPEC.loader
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


class MockModel:
    def encode(self, source, src_pad_mask=None):
        return torch.zeros((*source.shape, 4), dtype=torch.float32)

    def decode(self, tgt_tokens, enc_out, src_pad_mask=None):
        batch, length = tgt_tokens.shape
        logits = torch.full((batch, length, 8), -20.0)
        next_token = 4 if length == 1 else 3
        logits[:, -1, next_token] = 20.0
        return logits


class InferenceUtilityTests(unittest.TestCase):
    def test_parse_cli_and_alias(self) -> None:
        args = infer.parse_args(
            [
                "--input", "in.jsonl",
                "--output", "out.jsonl",
                "--checkpoint", "model.pt",
                "--tokenizer", "spm.model",
                "--num-beams", "4",
                "--batch-size", "3",
            ]
        )
        self.assertEqual(args.beam_size, 4)
        self.assertEqual(args.batch_size, 3)

    def test_batched_greedy_generation(self) -> None:
        source = torch.tensor([[5, 3, 0], [6, 7, 3]])
        mask = source.eq(0)
        outputs = infer.generate_batch_tokens(
            MockModel(),
            source,
            mask,
            bos_id=2,
            eos_id=3,
            pad_id=0,
            decoding={
                "strategy": "greedy",
                "beam_size": 1,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "max_new_tokens": 4,
                "min_new_tokens": 0,
            },
        )
        self.assertEqual(outputs, [[2, 4, 3], [2, 4, 3]])

    def test_resume_keeps_success_and_retries_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            records = [
                {
                    "id": "a",
                    "system": "scratch_transformer",
                    "config_id": "cfg",
                    "prediction": "done",
                    "status": "ok",
                    "error": None,
                },
                {
                    "id": "b",
                    "system": "scratch_transformer",
                    "config_id": "cfg",
                    "prediction": "",
                    "status": "error",
                    "error": "mock",
                },
            ]
            with output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            manifest = [{"id": "a"}, {"id": "b"}]
            successful, prior_errors = infer.normalize_resume_files(
                manifest, output, "scratch_transformer", "cfg", 0, False
            )
            self.assertEqual(set(successful), {"a"})
            self.assertEqual(prior_errors, 1)
            remaining = infer.load_jsonl(output)
            self.assertEqual([item["id"] for item in remaining], ["a"])


if __name__ == "__main__":
    unittest.main()
