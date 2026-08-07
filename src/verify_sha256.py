from pathlib import Path
import hashlib


DATA_DIR = Path("data")
CHECKSUM_FILE = DATA_DIR / "MANIFEST_SHA256.txt"


def calculate_sha256(path):
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


for line in CHECKSUM_FILE.read_text(
    encoding="utf-8"
).splitlines():

    if not line.strip():
        continue

    expected_hash, filename = line.split(maxsplit=1)
    filename = filename.strip()

    actual_hash = calculate_sha256(DATA_DIR / filename)

    if actual_hash == expected_hash:
        print(f"OK: {filename}")
    else:
        print(f"SAI CHECKSUM: {filename}")
        print(f"  Mong đợi: {expected_hash}")
        print(f"  Thực tế:  {actual_hash}")
