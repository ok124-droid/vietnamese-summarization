from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "validation_select": {
        "path": ROOT / "data" / "validation_select.jsonl",
        "expected_count": 2000,  # Đã chốt 2000 mẫu
        "expected_split": "validation",
        "id_pattern": r"validation_\d{6}",
    },
    "test_smoke_10": {
        "path": ROOT / "data" / "test_smoke_10.jsonl",
        "expected_count": 10,
        "expected_split": "test",
        "id_pattern": r"test_\d{6}",
    },
    "test_core_2000": {
        "path": ROOT / "data" / "test_core_2000.jsonl",
        "expected_count": 2000,
        "expected_split": "test",
        "id_pattern": r"test_\d{6}",
    },
}

REQUIRED_FIELDS = {"id", "split", "source", "reference"}
CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def add_issue(bucket, code, detail, count=1):
    item = bucket.setdefault(
        code,
        {"count": 0, "examples": []},
    )
    item["count"] += count

    if len(item["examples"]) < 5:
        item["examples"].append(detail)


def count_issues(bucket):
    return sum(item["count"] for item in bucket.values())


def sha256_file(path):
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def text_statistics(values):
    if not values:
        return {}

    sorted_values = sorted(values)

    def percentile(p):
        index = round((len(sorted_values) - 1) * p)
        return sorted_values[index]

    return {
        "min": sorted_values[0],
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "max": sorted_values[-1],
    }


def validate_file(name, spec):
    path = spec["path"]

    report = {
        "path": str(path),
        "sha256": None,
        "physical_lines": 0,
        "valid_rows": 0,
        "errors": {},
        "warnings": {},
        "statistics": {},
    }

    rows = []

    if not path.exists():
        add_issue(
            report["errors"],
            "FILE_NOT_FOUND",
            str(path),
        )
        return report, rows

    report["sha256"] = sha256_file(path)

    with path.open("rb") as file:
        if file.read(3) == b"\xef\xbb\xbf":
            add_issue(
                report["warnings"],
                "UTF8_BOM",
                "File có UTF-8 BOM",
            )

    try:
        lines = path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
    except UnicodeDecodeError as exc:
        add_issue(
            report["errors"],
            "INVALID_UTF8",
            str(exc),
        )
        return report, rows

    report["physical_lines"] = len(lines)
    seen_ids = {}

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            add_issue(
                report["errors"],
                "BLANK_LINE",
                {"line": line_number},
            )
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            add_issue(
                report["errors"],
                "INVALID_JSON",
                {
                    "line": line_number,
                    "message": str(exc),
                },
            )
            continue

        if not isinstance(row, dict):
            add_issue(
                report["errors"],
                "NOT_JSON_OBJECT",
                {"line": line_number},
            )
            continue

        fields = set(row)
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - REQUIRED_FIELDS)

        if missing:
            add_issue(
                report["errors"],
                "MISSING_FIELDS",
                {"line": line_number, "fields": missing},
            )

        if extra:
            add_issue(
                report["errors"],
                "EXTRA_FIELDS",
                {"line": line_number, "fields": extra},
            )

        if missing:
            continue

        invalid_types = [
            field
            for field in REQUIRED_FIELDS
            if not isinstance(row[field], str)
        ]

        if invalid_types:
            add_issue(
                report["errors"],
                "INVALID_FIELD_TYPE",
                {
                    "line": line_number,
                    "fields": invalid_types,
                },
            )
            continue

        sample_id = row["id"]

        if sample_id in seen_ids:
            add_issue(
                report["errors"],
                "DUPLICATE_ID",
                {
                    "id": sample_id,
                    "first_line": seen_ids[sample_id],
                    "duplicate_line": line_number,
                },
            )
        else:
            seen_ids[sample_id] = line_number

        if re.fullmatch(spec["id_pattern"], sample_id) is None:
            add_issue(
                report["errors"],
                "INVALID_ID_FORMAT",
                {"line": line_number, "id": sample_id},
            )

        if row["split"] != spec["expected_split"]:
            add_issue(
                report["errors"],
                "INVALID_SPLIT",
                {
                    "line": line_number,
                    "id": sample_id,
                    "actual": row["split"],
                    "expected": spec["expected_split"],
                },
            )

        for field in ("source", "reference"):
            value = row[field]

            if not value.strip():
                add_issue(
                    report["errors"],
                    "EMPTY_TEXT",
                    {
                        "line": line_number,
                        "id": sample_id,
                        "field": field,
                    },
                )

            if value != value.strip():
                add_issue(
                    report["warnings"],
                    "OUTER_WHITESPACE",
                    {"id": sample_id, "field": field},
                )

            if unicodedata.normalize("NFC", value) != value:
                add_issue(
                    report["warnings"],
                    "NON_NFC_UNICODE",
                    {"id": sample_id, "field": field},
                )

            if "\ufffd" in value:
                add_issue(
                    report["warnings"],
                    "REPLACEMENT_CHARACTER",
                    {"id": sample_id, "field": field},
                )

            if CONTROL_CHAR_PATTERN.search(value):
                add_issue(
                    report["warnings"],
                    "CONTROL_CHARACTER",
                    {"id": sample_id, "field": field},
                )

        source_words = len(row["source"].split())
        reference_words = len(row["reference"].split())

        if 0 < source_words < 20:
            add_issue(
                report["warnings"],
                "VERY_SHORT_SOURCE",
                {"id": sample_id, "words": source_words},
            )

        if 0 < reference_words < 3:
            add_issue(
                report["warnings"],
                "VERY_SHORT_REFERENCE",
                {"id": sample_id, "words": reference_words},
            )

        if reference_words > source_words > 0:
            add_issue(
                report["warnings"],
                "REFERENCE_LONGER_THAN_SOURCE",
                {
                    "id": sample_id,
                    "source_words": source_words,
                    "reference_words": reference_words,
                },
            )

        if (
            row["source"].strip()
            and row["source"].strip() == row["reference"].strip()
        ):
            add_issue(
                report["warnings"],
                "SOURCE_EQUALS_REFERENCE",
                {"id": sample_id},
            )

        rows.append(row)

    report["valid_rows"] = len(rows)

    expected_count = spec["expected_count"]

    if expected_count is None:
        add_issue(
            report["warnings"],
            "EXPECTED_COUNT_NOT_SET",
            "Cần điền số mẫu đã chốt cho file này",
        )
    elif len(rows) != expected_count:
        add_issue(
            report["errors"],
            "WRONG_ROW_COUNT",
            {
                "actual": len(rows),
                "expected": expected_count,
            },
        )

    source_to_ids = defaultdict(list)

    for row in rows:
        if row["source"].strip():
            source_to_ids[row["source"]].append(row["id"])

    for ids in source_to_ids.values():
        if len(ids) > 1:
            add_issue(
                report["warnings"],
                "DUPLICATE_SOURCE",
                {"ids": ids[:10]},
            )

    source_lengths = [
        len(row["source"].split())
        for row in rows
        if row["source"].strip()
    ]
    reference_lengths = [
        len(row["reference"].split())
        for row in rows
        if row["reference"].strip()
    ]

    report["statistics"] = {
        "source_words": text_statistics(source_lengths),
        "reference_words": text_statistics(reference_lengths),
    }

    return report, rows


def main():
    file_reports = {}
    all_rows = {}

    for name, spec in SPECS.items():
        report, rows = validate_file(name, spec)
        file_reports[name] = report
        all_rows[name] = rows

    cross_report = {
        "errors": {},
        "warnings": {},
    }

    row_maps = {
        name: {row["id"]: row for row in rows}
        for name, rows in all_rows.items()
    }

    validation_ids = set(row_maps["validation_select"])
    smoke_ids = set(row_maps["test_smoke_10"])
    core_ids = set(row_maps["test_core_2000"])

    validation_test_overlap = validation_ids & core_ids

    if validation_test_overlap:
        add_issue(
            cross_report["errors"],
            "VALIDATION_TEST_ID_OVERLAP",
            {"ids": sorted(validation_test_overlap)[:10]},
            count=len(validation_test_overlap),
        )

    missing_from_core = smoke_ids - core_ids

    if missing_from_core:
        add_issue(
            cross_report["errors"],
            "SMOKE_NOT_SUBSET_OF_CORE",
            {"ids": sorted(missing_from_core)[:10]},
            count=len(missing_from_core),
        )

    for sample_id in sorted(smoke_ids & core_ids):
        smoke_row = row_maps["test_smoke_10"][sample_id]
        core_row = row_maps["test_core_2000"][sample_id]

        for field in REQUIRED_FIELDS:
            if smoke_row[field] != core_row[field]:
                add_issue(
                    cross_report["errors"],
                    "SMOKE_CORE_CONTENT_MISMATCH",
                    {"id": sample_id, "field": field},
                )

    validation_sources = defaultdict(list)
    core_sources = defaultdict(list)

    for row in all_rows["validation_select"]:
        validation_sources[row["source"]].append(row["id"])

    for row in all_rows["test_core_2000"]:
        core_sources[row["source"]].append(row["id"])

    shared_sources = (
        set(validation_sources)
        & set(core_sources)
        - {""}
    )

    for source in shared_sources:
        add_issue(
            cross_report["warnings"],
            "EXACT_SOURCE_LEAKAGE",
            {
                "validation_ids": validation_sources[source][:5],
                "test_ids": core_sources[source][:5],
            },
        )

    final_report = {
        "files": file_reports,
        "cross_file_checks": cross_report,
    }

    output_path = (
        ROOT
        / "outputs"
        / "data_checks"
        / "manifest_validation_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_errors = count_issues(cross_report["errors"])

    for name, report in file_reports.items():
        errors = count_issues(report["errors"])
        warnings = count_issues(report["warnings"])
        total_errors += errors

        print(
            f"{name}: "
            f"rows={report['valid_rows']}, "
            f"errors={errors}, "
            f"warnings={warnings}"
        )

    print(
        "cross-file: "
        f"errors={count_issues(cross_report['errors'])}, "
        f"warnings={count_issues(cross_report['warnings'])}"
    )
    print(f"Report: {output_path}")

    if total_errors:
        raise SystemExit(
            f"FAILED: phát hiện {total_errors} lỗi nghiêm trọng"
        )

    print("PASSED: không có lỗi nghiêm trọng")


if __name__ == "__main__":
    main()