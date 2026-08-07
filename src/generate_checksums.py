from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs" / "data_checks"

FILES = [
    "validation_select.jsonl",
    "test_smoke_10.jsonl",
    "test_core_2000.jsonl",
]


def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    checksums = {}
    print("=== SHA-256 Checksums ===")
    for filename in FILES:
        path = DATA_DIR / filename
        if path.exists():
            checksum = sha256_file(path)
            checksums[filename] = checksum
            print(f"{filename}:\n  {checksum}")
        else:
            print(f"{filename}: FILE NOT FOUND")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / "sha256_checksums.json"
    out_file.write_text(
        json.dumps(checksums, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved checksums to: {out_file}")


if __name__ == "__main__":
    main()
