"""Read-only Zotero PDF catalogue; does not edit the library or its database.

python scripts/index_literature_library.py
Optional --metadata requires pypdf and extracts first-two-page DOI candidates.
中文：目录/hash/重复附件核对，DOI候选仍需与出版商原始页面核对。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()
DEFAULT_LIBRARY = Path("C:/Users/19254/Zotero/storage")
OUTPUT_FILE = PROJECT_ROOT / "results" / "audit_20261008" / "literature_library_index.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    if not args.library.is_dir():
        raise FileNotFoundError(f"Zotero storage not found: {args.library}; set --library on this computer")
    reader_class = None
    if args.metadata:
        from pypdf import PdfReader
        reader_class = PdfReader
    records, hashes = [], {}
    for path in sorted(args.library.rglob("*.pdf")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        record = {"path": str(path.resolve()), "attachment_id": path.parent.name,
                  "sha256": digest, "bytes": len(raw),
                  "relevance_from_filename": bool(re.search(r"photonic|PCSEL|coupled.wave|YAG|晶体.*激光|光子晶体|topolog|nanolithography", path.name, re.I))}
        hashes.setdefault(digest, []).append(str(path.resolve()))
        if reader_class:
            try:
                reader = reader_class(path)
                first = "\n".join(page.extract_text() or "" for page in reader.pages[:2])
                record.update({"pages": len(reader.pages), "pdf_title": str((reader.metadata or {}).get("/Title", "")),
                               "doi_candidates": sorted(set(re.findall(r"10\.\d{4,9}/[^\s<>]+", first)))[:12]})
            except Exception as error:
                record["metadata_error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    result = {"date": "2026-10-08", "library": str(args.library.resolve()),
              "pdf_count": len(records), "unique_pdf_count": len(hashes), "records": records,
              "byte_identical_duplicates": [paths for paths in hashes.values() if len(paths) > 1],
              "scope": "file/metadata catalogue; not a claim that every PDF was substantively reviewed"}
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PDFs={len(records)}, unique={len(hashes)}, byte-identical duplicate groups={len(result['byte_identical_duplicates'])}")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
