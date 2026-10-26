"""Read back selected optimization evidence and save a reproducible file index.

This checks serialization and numeric integrity. It is not an independent
electromagnetic solver or a certification of lasing performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pcselsim.run_output import prepare_output, write_provenance


def json_numbers(value):
    if isinstance(value, dict):
        return sum(json_numbers(v) for v in value.values())
    if isinstance(value, list):
        return sum(json_numbers(v) for v in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise ValueError("JSON contains a non-finite number")
        return 1
    return 0


def check_file(path: Path):
    row = {"path": path.relative_to(ROOT).as_posix(),
           "bytes": path.stat().st_size,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        row["finite_numeric_values"] = json_numbers(payload)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            numeric = 0
            for name in arrays.files:
                arr = arrays[name]
                if arr.dtype.kind in "iufc":
                    if not np.isfinite(arr).all():
                        raise ValueError(f"Non-finite array: {path.name}:{name}")
                    numeric += 1
            row.update(arrays=len(arrays.files), finite_numeric_arrays=numeric)
    elif path.suffix.lower() == ".png":
        # Matplotlib is already a platform dependency; Pillow is not required.
        import matplotlib.image as mpimg
        pixels = mpimg.imread(path)
        if not np.isfinite(pixels).all():
            raise ValueError(f"Non-finite image: {path}")
        row["image_shape"] = list(pixels.shape)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({(p if p.is_absolute() else ROOT / p).resolve() for p in args.files})
    for path in paths:
        if ROOT not in path.parents or not path.is_file():
            raise ValueError(f"Evidence must be an existing simulation file: {path}")
    target = args.output if args.output.is_absolute() else ROOT / args.output
    output = prepare_output(target, ROOT)
    write_provenance(output, ROOT)
    status = output / "run_status.json"
    status.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    try:
        rows = [check_file(path) for path in paths]
        report = {"scope": "Selected evidence readback and SHA256 index; not physical certification",
                  "files": rows, "file_count": len(rows),
                  "npz_numeric_arrays": sum(r.get("finite_numeric_arrays", 0) for r in rows),
                  "json_numeric_values": sum(r.get("finite_numeric_values", 0) for r in rows),
                  "png_count": sum("image_shape" in r for r in rows)}
        (output / "evidence_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        status.write_text(json.dumps({"status": "completed", "completed_at": datetime.now().isoformat(),
                                     "file_count": len(rows)}, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "files"}))
    except BaseException as exc:
        status.write_text(json.dumps({"status": "failed", "error": repr(exc)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
