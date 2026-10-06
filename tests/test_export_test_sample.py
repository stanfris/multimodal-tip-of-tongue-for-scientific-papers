from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/packaging/export_test_sample.py"
DATASETS = ("ACL", "Biology", "Engineering", "Medicine", "Physics")


def test_exports_only_query_linked_test_pdfs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    split = {"test": [], "train": []}
    for dataset in DATASETS:
        query_dir = data / "query_collections" / dataset / "query_generation_test" / "visual_and_text"
        query_dir.mkdir(parents=True)
        rows = []
        for number in range(3):
            relative = f"{dataset}/paper-{number}.pdf"
            split["test"].append(relative)
            pdf = data / "pdf_datasets" / relative
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(b"%PDF-1.4\n" + relative.encode())
            if number < 2:
                rows.append({
                    "query_id": f"q-{number}",
                    "query": f"Find {dataset} paper {number}",
                    "relevant_ids": [f"{dataset}_paper-{number}"],
                })
        (query_dir / "queries.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    split_path = data / "splits/pdf_dataset_split.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "sample.zip"

    subprocess.run([
        sys.executable, str(SCRIPT), "--data-root", str(data), "--output", str(output),
        "--count", "2", "--offline",
    ], check=True)

    with zipfile.ZipFile(output) as archive:
        pdfs = sorted(name for name in archive.namelist() if name.endswith(".pdf"))
        queries = json.loads(archive.read("queries.json"))
        assert len(pdfs) == 10
        assert len(queries) == 10
        for row in queries:
            assert row["pdf"] in pdfs
            assert row["document_id"] == row["dataset"] + "_" + Path(row["pdf"]).stem
            assert row["query"].startswith("Find ")
            assert archive.read(row["pdf"]).startswith(b"%PDF-1.4")
