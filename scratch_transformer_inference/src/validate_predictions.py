#!/usr/bin/env python3
"""Kiểm tra prediction JSONL của Scratch Transformer trước khi bàn giao."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REQUIRED_FIELDS = (
    "id",
    "system",
    "config_id",
    "prediction",
    "status",
    "error",
)
EXPECTED_SYSTEM = "scratch_transformer"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manifest và prediction Scratch Transformer."
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSONL chuẩn.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL.")
    parser.add_argument(
        "--expected-config-id",
        default=None,
        help="Nếu truyền, mọi record phải dùng đúng config_id này.",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path, label: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    issues: List[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], [f"Không mở được {label} {path}: {exc}"]

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                issues.append(f"{label}:{line_number}: dòng rỗng.")
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{label}:{line_number}: JSON không hợp lệ: {exc}")
                continue
            if not isinstance(item, dict):
                issues.append(f"{label}:{line_number}: mỗi dòng phải là JSON object.")
                continue
            item["_validator_line"] = line_number
            records.append(item)
    return records, issues


def validate(
    manifest_path: Path,
    prediction_path: Path,
    expected_config_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    manifest, manifest_issues = read_jsonl(manifest_path, "manifest")
    predictions, prediction_issues = read_jsonl(prediction_path, "prediction")
    issues = manifest_issues + prediction_issues

    manifest_ids: List[str] = []
    for item in manifest:
        line = item["_validator_line"]
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(f"manifest:{line}: id phải là chuỗi không rỗng.")
            continue
        manifest_ids.append(sample_id)

    manifest_counts = Counter(manifest_ids)
    manifest_duplicate_count = sum(count - 1 for count in manifest_counts.values())
    if manifest_duplicate_count:
        issues.append(f"Manifest có {manifest_duplicate_count} ID trùng.")

    prediction_ids: List[str] = []
    config_ids: set[str] = set()
    success_count = 0
    failed_count = 0
    required_set = set(REQUIRED_FIELDS)

    for item in predictions:
        line = item["_validator_line"]
        public_fields = set(item) - {"_validator_line"}
        missing_fields = sorted(required_set - public_fields)
        extra_fields = sorted(public_fields - required_set)
        if missing_fields:
            issues.append(
                f"prediction:{line}: thiếu trường {', '.join(missing_fields)}."
            )
        if extra_fields:
            issues.append(
                f"prediction:{line}: có trường ngoài schema {', '.join(extra_fields)}."
            )

        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(f"prediction:{line}: id phải là chuỗi không rỗng.")
        else:
            prediction_ids.append(sample_id)

        if item.get("system") != EXPECTED_SYSTEM:
            issues.append(
                f"prediction:{line}: system phải bằng '{EXPECTED_SYSTEM}'."
            )

        config_id = item.get("config_id")
        if not isinstance(config_id, str) or not config_id:
            issues.append(f"prediction:{line}: config_id phải là chuỗi không rỗng.")
        else:
            config_ids.add(config_id)
            if expected_config_id is not None and config_id != expected_config_id:
                issues.append(
                    f"prediction:{line}: config_id={config_id!r}, "
                    f"cần {expected_config_id!r}."
                )

        prediction = item.get("prediction")
        if not isinstance(prediction, str):
            issues.append(f"prediction:{line}: prediction phải là chuỗi.")

        status = item.get("status")
        if status == "ok":
            success_count += 1
            if not isinstance(prediction, str) or not prediction.strip():
                issues.append(
                    f"prediction:{line}: status='ok' nhưng prediction rỗng."
                )
            if item.get("error") is not None:
                issues.append(f"prediction:{line}: record ok phải có error=null.")
        elif status == "error":
            failed_count += 1
            error = item.get("error")
            if not isinstance(error, str) or not error.strip():
                issues.append(
                    f"prediction:{line}: record error phải có thông báo lỗi."
                )
        else:
            issues.append(
                f"prediction:{line}: status chỉ được là 'ok' hoặc 'error'."
            )

    prediction_counts = Counter(prediction_ids)
    duplicate_count = sum(count - 1 for count in prediction_counts.values())
    if duplicate_count:
        issues.append(f"Prediction có {duplicate_count} ID trùng.")
    if len(config_ids) > 1:
        issues.append(f"config_id không nhất quán: {sorted(config_ids)}")

    manifest_set = set(manifest_ids)
    prediction_set = set(prediction_ids)
    missing_ids = sorted(manifest_set - prediction_set)
    extra_ids = sorted(prediction_set - manifest_set)
    if missing_ids:
        issues.append(
            f"Thiếu {len(missing_ids)} ID, ví dụ: {missing_ids[:5]}"
        )
    if extra_ids:
        issues.append(f"Thừa {len(extra_ids)} ID, ví dụ: {extra_ids[:5]}")
    if failed_count:
        issues.append(
            f"Có {failed_count} record status='error'; chưa đủ điều kiện bàn giao."
        )

    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "predictions": str(prediction_path),
        "input": len(manifest),
        "output": len(predictions),
        "success": success_count,
        "failed": failed_count,
        "duplicate": duplicate_count,
        "manifest_duplicate": manifest_duplicate_count,
        "missing": len(missing_ids),
        "extra": len(extra_ids),
        "config_ids": sorted(config_ids),
        "valid_for_handoff": not issues,
        "issue_count": len(issues),
        "issues": issues[:50],
    }
    return summary, issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    summary, issues = validate(
        Path(args.manifest).expanduser().resolve(),
        Path(args.predictions).expanduser().resolve(),
        args.expected_config_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
