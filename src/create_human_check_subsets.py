from pathlib import Path
import json
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs" / "data_checks"
RANDOM_SEED = 42
TARGET_TOTAL = 10


def get_row_warnings(row):
    warns = []
    source = row.get("source", "")
    ref = row.get("reference", "")

    if source != source.strip() or ref != ref.strip():
        warns.append("OUTER_WHITESPACE")

    if (
        unicodedata.normalize("NFC", source) != source
        or unicodedata.normalize("NFC", ref) != ref
    ):
        warns.append("NON_NFC_UNICODE")

    if "\ufffd" in source or "\ufffd" in ref:
        warns.append("REPLACEMENT_CHARACTER")

    if re.search(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", source
    ) or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", ref):
        warns.append("CONTROL_CHARACTER")

    sw = len(source.split())
    rw = len(ref.split())

    if 0 < sw < 20:
        warns.append("VERY_SHORT_SOURCE")

    if 0 < rw < 3:
        warns.append("VERY_SHORT_REFERENCE")

    if rw > sw > 0:
        warns.append("REFERENCE_LONGER_THAN_SOURCE")

    if source.strip() and source.strip() == ref.strip():
        warns.append("SOURCE_EQUALS_REFERENCE")

    return warns


def extract_subset(filename, target_total=TARGET_TOTAL, seed=RANDOM_SEED):
    file_path = DATA_DIR / filename
    lines = file_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    selected_dict = {}

    # 1. Warning samples
    for r in rows:
        w = get_row_warnings(r)
        if w:
            selected_dict[r["id"]] = (r, f"warning ({', '.join(w)})")

    # 2. Min source_words
    sorted_rows = sorted(
        rows, key=lambda r: len(r["source"].split())
    )
    min_row = sorted_rows[0]
    if min_row["id"] not in selected_dict:
        selected_dict[min_row["id"]] = (
            min_row,
            f"min_source_words ({len(min_row['source'].split())} words)",
        )

    # 3. Max source_words
    max_row = sorted_rows[-1]
    if max_row["id"] not in selected_dict:
        selected_dict[max_row["id"]] = (
            max_row,
            f"max_source_words ({len(max_row['source'].split())} words)",
        )

    # 4. Random samples
    needed_random = target_total - len(selected_dict)
    remaining_rows = [
        r for r in rows if r["id"] not in selected_dict
    ]

    random.seed(seed)
    random_samples = random.sample(remaining_rows, needed_random)

    for r in random_samples:
        selected_dict[r["id"]] = (r, "random")

    final_rows = [item[0] for item in selected_dict.values()]
    details = {r_id: item[1] for r_id, item in selected_dict.items()}

    return final_rows, details


def main():
    specs = [
        (
            "validation_select.jsonl",
            "validation_select_human_check.jsonl",
        ),
        (
            "test_core_2000.jsonl",
            "test_core_2000_human_check.jsonl",
        ),
    ]

    manifest = {
        "random_seed": RANDOM_SEED,
        "target_total_per_file": TARGET_TOTAL,
        "sampling_methodology": [
            "1. Lấy tất cả mẫu bị cảnh báo (warning) ở cấp dòng dữ liệu (CONTROL_CHARACTER, OUTER_WHITESPACE, v.v.).",
            "2. Lấy mẫu có độ dài source_words nhỏ nhất (min_source_words).",
            "3. Lấy mẫu có độ dài source_words lớn nhất (max_source_words).",
            "4. Lấy mẫu ngẫu nhiên không hoàn lại từ tập còn lại với random.seed=42 cho đến khi đạt đủ tổng số mẫu mục tiêu (10 mẫu/file).",
        ],
        "subsets": {},
    }

    for src_name, dst_name in specs:
        rows, details = extract_subset(
            src_name, target_total=TARGET_TOTAL, seed=RANDOM_SEED
        )
        dst_path = DATA_DIR / dst_name

        with dst_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        subset_info = []
        for r in rows:
            subset_info.append(
                {
                    "id": r["id"],
                    "reason": details[r["id"]],
                    "source_words": len(r["source"].split()),
                    "reference_words": len(r["reference"].split()),
                }
            )

        manifest["subsets"][dst_name] = {
            "source_file": src_name,
            "sample_count": len(rows),
            "samples": subset_info,
        }

        print(f"=== {dst_name} ({len(rows)} samples) ===")
        for r_id, reason in details.items():
            print(f"  - {r_id}: {reason}")
        print()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS_DIR / "human_check_sampling_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
