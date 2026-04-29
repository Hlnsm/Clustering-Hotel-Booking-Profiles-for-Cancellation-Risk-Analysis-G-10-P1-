import hashlib
import os

def load_expected_hashes(sha_file_path):

    expected_hashes = {}

    with open(sha_file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                continue  

            hash_value, filename = parts

            expected_hashes[filename] = hash_value

    return expected_hashes



EXPECTED_HASHES=load_expected_hashes("data/SHA256SUMS.txt")



def compute_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_dataset_integrity(data_dir):
   
    results = {}
    all_ok = True

    for filename, expected_hash in EXPECTED_HASHES.items():
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            results[filename] = {
                "status": "missing",
                "expected": expected_hash,
                "actual": None
            }
            all_ok = False
            continue

        actual_hash = compute_sha256(file_path)

        if actual_hash == expected_hash:
            results[filename] = {
                "status": "ok",
                "expected": expected_hash,
                "actual": actual_hash
            }
        else:
            results[filename] = {
                "status": "mismatch",
                "expected": expected_hash,
                "actual": actual_hash
            }
            all_ok = False

    return all_ok, results


