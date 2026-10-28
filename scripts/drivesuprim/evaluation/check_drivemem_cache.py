"""Read-only cache coverage/integrity check; does not import NAVSIM or torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect(root):
    records = json.loads((root / "splits.json").read_text(encoding="utf-8"))["records"]
    tokens = [r["token"] for r in records]
    path = root / "cache_index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    expected = set(tokens)
    report = {"status": "incomplete", "root": str(root), "selected_scenes": len(tokens),
              "indexed_scenes": len(index), "valid_files": 0, "missing_tokens": sorted(expected - set(index)),
              "extra_tokens": sorted(set(index) - expected), "invalid_files": []}
    for token in sorted(expected & set(index)):
        entry = index[token]
        file = Path(entry["path"])
        try:
            file.resolve().relative_to((root / "metric_cache").resolve())
            if not file.is_file():
                reason = "missing file"
            elif sha(file) != entry["sha256"]:
                reason = "SHA256 mismatch"
            else:
                report["valid_files"] += 1
                continue
        except ValueError:
            reason = "path outside run metric_cache"
        except OSError as exc:
            reason = str(exc)
        report["invalid_files"].append({"token": token, "path": str(file), "reason": reason})
    if (tokens and len(tokens) == len(expected) and report["valid_files"] == len(tokens)
            and not report["extra_tokens"]):
        report["status"] = "complete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?",
                        default=Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory/pilot_v1")
    args = parser.parse_args()
    report = inspect(args.root)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
