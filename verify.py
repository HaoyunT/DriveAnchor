"""Verify the packaged file hashes and parse Python sources without running them."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
errors = []
for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        errors.append(entry["path"])
    elif path.suffix == ".py":
        ast.parse(path.read_text(), filename=entry["path"])
if errors:
    raise SystemExit("Hash verification failed: " + ", ".join(errors))
print(f"Verified {len(manifest['files'])} packaged files. No model evaluation performed.")
