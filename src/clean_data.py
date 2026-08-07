from pathlib import Path
import json
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FILES = [
    "validation_select.jsonl",
    "test_smoke_10.jsonl",
    "test_core_2000.jsonl",
]


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\u0008", "")
    text = unicodedata.normalize("NFC", text)
    return text


def clean_file(filename):
    file_path = DATA_DIR / filename
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    cleaned_rows = []

    cleaned_count = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        modified = False
        for key, val in row.items():
            if isinstance(val, str):
                new_val = clean_text(val)
                if new_val != val:
                    modified = True
                row[key] = new_val
        if modified:
            cleaned_count += 1
        cleaned_rows.append(row)

    with file_path.open("w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Cleaned {filename}: {cleaned_count} rows modified out of {len(cleaned_rows)} total rows.")


def main():
    for fname in FILES:
        clean_file(fname)


if __name__ == "__main__":
    main()
