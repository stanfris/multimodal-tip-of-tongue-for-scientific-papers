from __future__ import annotations

import json
import urllib.error

import pytest

from dataset_generation.document_downloads.arxiv_open_reuse import HTTPFetchError
from dataset_generation.document_downloads.arxiv_open_reuse import build_arxiv_open_reuse_corpus
from dataset_generation.document_downloads.arxiv_open_reuse import download_eligible_pdfs
from dataset_generation.document_downloads.arxiv_open_reuse import filter_candidate
from dataset_generation.document_downloads.arxiv_open_reuse import fetch_bytes
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_license
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_license_url
from dataset_generation.document_downloads.arxiv_open_reuse import pdf_filename_for_arxiv_id


def kaggle_record(
    arxiv_id: str,
    *,
    categories: str = "physics.ins-det",
    license_url: str | None = "http://creativecommons.org/licenses/by/4.0/",
    created: str = "Mon, 1 Jan 2024 00:00:00 GMT",
) -> dict:
    return {
        "id": arxiv_id,
        "submitter": "Marie Curie",
        "authors": "Marie Curie",
        "authors_parsed": [["Curie", "Marie", ""]],
        "title": "A robust instrument",
        "comments": "12 pages",
        "journal-ref": None,
        "doi": "10.1234/example",
        "report-no": None,
        "categories": categories,
        "license": license_url,
        "abstract": "Measurement details.",
        "versions": [{"version": "v1", "created": created}],
        "update_date": "2024-01-02",
    }


def write_snapshot(path, rows: list[dict | str]) -> None:  # type: ignore[no-untyped-def]
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_license_normalization_maps_urls_to_allowed_labels() -> None:
    assert normalize_license_url("http://creativecommons.org/licenses/by/4.0/") == (
        "https://creativecommons.org/licenses/by/4.0"
    )
    assert normalize_license("https://creativecommons.org/licenses/by-sa/4.0/").label == "CC BY-SA 4.0"
    assert normalize_license("https://creativecommons.org/publicdomain/zero/1.0/").label == "CC0 1.0"
    assert normalize_license("http://arxiv.org/licenses/nonexclusive-distrib/1.0/").family == "arxiv-default"


def test_filter_candidate_uses_snapshot_license_before_pdf_download() -> None:
    candidate, rejection = filter_candidate(
        kaggle_record("2401.01234v2", categories="physics.ins-det eess.SP"),
        category_prefixes=("physics.", "eess."),
        start_year=2020,
        end_year=2026,
        allowed_licenses={"CC BY 4.0"},
    )

    assert rejection is None
    assert candidate is not None
    assert candidate["arxiv_id"] == "2401.01234v2"
    assert candidate["license"] == "CC BY 4.0"
    assert candidate["categories"] == ["physics.ins-det", "eess.SP"]
    assert candidate["primary_category"] == "physics.ins-det"
    assert candidate["latest_version_date"] == "Mon, 1 Jan 2024 00:00:00 GMT"
    assert candidate["pdf_url"] == "https://arxiv.org/pdf/2401.01234v2"


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (kaggle_record("2401.1", categories="cs.CL"), "wrong_category"),
        (kaggle_record("2401.2", license_url=None), "missing_license"),
        (
            kaggle_record("2401.3", license_url="http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
            "disallowed_license",
        ),
        (kaggle_record("2401.4", created="Mon, 1 Jan 2010 00:00:00 GMT"), "outside_date_range"),
    ],
)
def test_filter_candidate_rejects_unsuitable_records(record, reason) -> None:  # type: ignore[no-untyped-def]
    candidate, rejection = filter_candidate(
        record,
        category_prefixes=("physics.", "eess."),
        start_year=2020,
        end_year=2026,
        allowed_licenses={"CC BY 4.0"},
    )

    assert candidate is None
    assert rejection == reason


def test_build_corpus_streams_snapshot_and_writes_selected_manifest(tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(
        metadata,
        [
            kaggle_record("2401.00001v1", categories="cs.CL"),
            kaggle_record("2401.00002v1", license_url="http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
            "{bad json",
            kaggle_record("2401.00003v1", categories="physics.ins-det"),
            kaggle_record("2401.00004v1", categories="eess.SP", license_url="https://creativecommons.org/licenses/by-sa/4.0/"),
        ],
    )

    result = build_arxiv_open_reuse_corpus(
        metadata_path=metadata,
        output_dir=tmp_path / "out",
        metadata_only=True,
        target_count=2,
        category_prefixes=("physics.", "eess."),
        allowed_licenses={"CC BY 4.0", "CC BY-SA 4.0"},
        progress_interval=0,
    )

    selected_path = tmp_path / "out" / "selected_arxiv_documents.jsonl"
    selected = [json.loads(line) for line in selected_path.read_text(encoding="utf-8").splitlines()]
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))

    assert result.metadata_records == 5
    assert result.eligible_records == 2
    assert [record["arxiv_id"] for record in selected] == ["2401.00003v1", "2401.00004v1"]
    assert report["rejection_counts"]["wrong_category"] == 1
    assert report["rejection_counts"]["disallowed_license"] == 1
    assert report["rejection_counts"]["malformed_metadata"] == 1
    assert report["selected_license_counts"] == {"CC BY 4.0": 1, "CC BY-SA 4.0": 1}


def test_pdf_filename_handles_old_style_arxiv_ids() -> None:
    assert pdf_filename_for_arxiv_id("hep-th/9901001v1") == "hep-th_9901001v1.pdf"


def test_download_eligible_pdfs_resumes_existing_valid_pdf(monkeypatch, tmp_path) -> None:
    existing = tmp_path / "2401.00001v1.pdf"
    existing.write_bytes(b"%PDF\n")

    def fail_download_pdf(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("valid existing PDFs should be skipped")

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_pdf", fail_download_pdf)

    stats = download_eligible_pdfs(
        [
            {
                "arxiv_id": "2401.00001v1",
                "pdf_url": "https://arxiv.org/pdf/2401.00001v1",
                "pdf_filename": "2401.00001v1.pdf",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "normalized_license_url": "https://creativecommons.org/licenses/by/4.0",
                "license": "CC BY 4.0",
                "license_family": "cc-by",
            }
        ],
        output_dir=tmp_path,
        overwrite=False,
        request_delay_seconds=0,
        timeout_seconds=10,
        max_retries=1,
        max_workers=1,
    )

    assert stats["downloaded"] == 0
    assert stats["skipped"] == 1
    assert stats["failed"] == 0


def test_fetch_bytes_retries_transient_http_errors(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"%PDF\n"

    def fake_urlopen(request, *, timeout):  # type: ignore[no-untyped-def]
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = fetch_bytes(
        "https://arxiv.org/pdf/2401.00001v1",
        request_delay_seconds=0,
        timeout_seconds=10,
        max_retries=1,
    )

    assert payload == b"%PDF\n"
    assert len(calls) == 2


def test_fetch_bytes_stops_on_permanent_http_errors(monkeypatch) -> None:
    def fake_urlopen(request, *, timeout):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HTTPFetchError):
        fetch_bytes(
            "https://arxiv.org/pdf/missing",
            request_delay_seconds=0,
            timeout_seconds=10,
            max_retries=5,
        )
