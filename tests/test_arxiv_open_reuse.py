from __future__ import annotations

import json
import threading
import time
import urllib.error
import xml.etree.ElementTree as ET
from datetime import date

import pytest

from dataset_generation.document_downloads.arxiv_open_reuse import HTTPFetchError
from dataset_generation.document_downloads.arxiv_open_reuse import build_arxiv_open_reuse_corpus
from dataset_generation.document_downloads.arxiv_open_reuse import build_parser
from dataset_generation.document_downloads.arxiv_open_reuse import build_target_counts
from dataset_generation.document_downloads.arxiv_open_reuse import download_eligible_pdfs
from dataset_generation.document_downloads.arxiv_open_reuse import filter_candidate
from dataset_generation.document_downloads.arxiv_open_reuse import fetch_bytes
from dataset_generation.document_downloads.arxiv_open_reuse import ensure_kaggle_auth
from dataset_generation.document_downloads.arxiv_open_reuse import iter_oai_records
from dataset_generation.document_downloads.arxiv_open_reuse import kaggle_pdf_object_name
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_license
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_license_url
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_oai_base_url
from dataset_generation.document_downloads.arxiv_open_reuse import parse_oai_arxiv_record
from dataset_generation.document_downloads.arxiv_open_reuse import pdf_filename_for_arxiv_id
from dataset_generation.document_downloads.arxiv_open_reuse import resolve_kaggle_snapshot
from dataset_generation.document_downloads.arxiv_open_reuse import versioned_arxiv_id


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
    assert normalize_license_url("https://creativecommons.org/licenses/by/4.0") == (
        "https://creativecommons.org/licenses/by/4.0"
    )
    assert normalize_license("https://creativecommons.org/licenses/by-sa/4.0/").label == "CC BY-SA 4.0"
    assert normalize_license("https://creativecommons.org/publicdomain/zero/1.0/").label == "CC0 1.0"
    assert normalize_license("http://arxiv.org/licenses/nonexclusive-distrib/1.0/").family == "arxiv-default"


def test_obsolete_oai_endpoint_is_rewritten_to_current_endpoint() -> None:
    assert normalize_oai_base_url("https://export.arxiv.org/oai2") == "https://oaipmh.arxiv.org/oai"
    assert normalize_oai_base_url("http://export.arxiv.org/oai2/") == "https://oaipmh.arxiv.org/oai"


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
    assert candidate["pdf_arxiv_id"] == "2401.01234v2"
    assert candidate["kaggle_gcs_object"] == "arxiv/arxiv/pdf/2401/2401.01234v2.pdf"
    assert candidate["pdf_url"].startswith("https://storage.googleapis.com/download/storage/v1/b/arxiv-dataset/o/")


def test_parser_defaults_to_snapshot_and_strict_cc_by_4() -> None:
    args = build_parser().parse_args([])

    assert args.metadata_source == "snapshot"
    assert args.download_snapshot is True
    assert args.allowed_licenses is None


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (kaggle_record("2401.1", categories="cs.CL"), "wrong_category"),
        (kaggle_record("2401.2", license_url=None), "missing_license"),
        (
            kaggle_record("2401.3", license_url="http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
            "disallowed_license",
        ),
        (
            kaggle_record("2401.5", license_url="https://creativecommons.org/licenses/by-sa/4.0/"),
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


def test_category_matching_accepts_existing_physics_and_engineering_prefixes() -> None:
    physics, physics_rejection = filter_candidate(
        kaggle_record("2401.01000v1", categories="physics.optics"),
        category_prefixes=("physics.", "eess."),
        start_year=None,
        end_year=None,
        allowed_licenses={"CC BY 4.0"},
    )
    engineering, engineering_rejection = filter_candidate(
        kaggle_record("2401.01001v1", categories="eess.SY"),
        category_prefixes=("physics.", "eess."),
        start_year=None,
        end_year=None,
        allowed_licenses={"CC BY 4.0"},
    )

    assert physics_rejection is None
    assert engineering_rejection is None
    assert physics is not None
    assert engineering is not None


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


def test_target_per_domain_balances_physics_and_engineering(tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(
        metadata,
        [
            kaggle_record("2401.00001v1", categories="physics.ins-det"),
            kaggle_record("2401.00002v1", categories="physics.optics"),
            kaggle_record("2401.00003v1", categories="physics.acc-ph"),
            kaggle_record("2401.00004v1", categories="eess.SP"),
            kaggle_record("2401.00005v1", categories="eess.SY"),
            kaggle_record("2401.00006v1", categories="eess.AS"),
        ],
    )

    result = build_arxiv_open_reuse_corpus(
        snapshot_path=metadata,
        output_dir=tmp_path / "out",
        metadata_only=True,
        target_count=4,
        target_per_domain=2,
        category_prefixes=("physics.", "eess."),
        progress_interval=0,
    )
    selected = [
        json.loads(line)
        for line in (tmp_path / "out" / "selected_arxiv_documents.jsonl").read_text().splitlines()
    ]
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))

    assert result.eligible_records == 4
    assert [record["arxiv_id"] for record in selected] == [
        "2401.00001v1",
        "2401.00002v1",
        "2401.00004v1",
        "2401.00005v1",
    ]
    assert report["target_per_domain"] == 2
    assert report["selected_domain_counts"] == {"eess.": 2, "physics.": 2}


def test_target_for_prefix_allows_higher_engineering_target(tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(
        metadata,
        [
            kaggle_record("2401.00001v1", categories="physics.ins-det"),
            kaggle_record("2401.00002v1", categories="physics.optics"),
            kaggle_record("2401.00003v1", categories="eess.SP"),
            kaggle_record("2401.00004v1", categories="eess.SY"),
            kaggle_record("2401.00005v1", categories="eess.AS"),
        ],
    )

    result = build_arxiv_open_reuse_corpus(
        snapshot_path=metadata,
        output_dir=tmp_path / "out",
        metadata_only=True,
        target_count=4,
        target_counts={"physics.": 1, "eess.": 3},
        category_prefixes=("physics.", "eess."),
        progress_interval=0,
    )
    selected = [
        json.loads(line)
        for line in (tmp_path / "out" / "selected_arxiv_documents.jsonl").read_text().splitlines()
    ]
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))

    assert result.eligible_records == 4
    assert [record["arxiv_id"] for record in selected] == [
        "2401.00001v1",
        "2401.00003v1",
        "2401.00004v1",
        "2401.00005v1",
    ]
    assert report["target_domain_counts"] == {"eess.": 3, "physics.": 1}
    assert report["selected_domain_counts"] == {"eess.": 3, "physics.": 1}


def test_build_target_counts_accepts_prefix_overrides() -> None:
    counts = build_target_counts(
        category_prefixes=("physics.", "eess."),
        target_count=20_000,
        target_per_domain=None,
        target_for_prefix=("physics.=30000", "eess.=60000"),
    )

    assert counts == {"physics.": 30_000, "eess.": 60_000}


def test_build_corpus_default_license_allowlist_rejects_non_cc_by_4(tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(
        metadata,
        [
            kaggle_record("2401.00003v1", categories="physics.ins-det"),
            kaggle_record("2401.00004v1", categories="eess.SP", license_url="https://creativecommons.org/licenses/by-sa/4.0/"),
        ],
    )

    result = build_arxiv_open_reuse_corpus(
        snapshot_path=metadata,
        output_dir=tmp_path / "out",
        metadata_only=True,
        target_count=20_000,
        category_prefixes=("physics.", "eess."),
        progress_interval=0,
    )
    selected = [json.loads(line) for line in (tmp_path / "out" / "selected_arxiv_documents.jsonl").read_text().splitlines()]

    assert result.eligible_records == 1
    assert [record["arxiv_id"] for record in selected] == ["2401.00003v1"]


def test_selection_is_deterministic_snapshot_order(tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(
        metadata,
        [
            kaggle_record("2401.00001v1"),
            kaggle_record("2401.00002v1"),
            kaggle_record("2401.00003v1"),
        ],
    )

    ids_by_run = []
    for name in ("out-a", "out-b"):
        build_arxiv_open_reuse_corpus(
            snapshot_path=metadata,
            output_dir=tmp_path / name,
            metadata_only=True,
            target_count=2,
            category_prefixes=("physics.", "eess."),
            progress_interval=0,
            selection_seed=123,
        )
        rows = [json.loads(line) for line in (tmp_path / name / "selected_arxiv_documents.jsonl").read_text().splitlines()]
        ids_by_run.append([row["arxiv_id"] for row in rows])

    assert ids_by_run == [["2401.00001v1", "2401.00002v1"], ["2401.00001v1", "2401.00002v1"]]


def test_reuses_existing_snapshot_without_kaggle_download(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "arxiv-metadata-oai-snapshot.json"
    snapshot.write_text("", encoding="utf-8")

    def fail_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("existing snapshot should be reused")

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_kaggle_snapshot", fail_download)

    assert resolve_kaggle_snapshot(
        snapshot_path=snapshot,
        output_dir=tmp_path,
        download_snapshot=True,
        kaggle_cli="kaggle",
    ) == snapshot


def test_missing_snapshot_without_download_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Kaggle snapshot not found"):
        resolve_kaggle_snapshot(
            snapshot_path=tmp_path / "arxiv-metadata-oai-snapshot.json",
            output_dir=tmp_path,
            download_snapshot=False,
            kaggle_cli="kaggle",
        )


def test_missing_kaggle_credentials_explain_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/kaggle")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(RuntimeError, match="Kaggle credentials are missing"):
        ensure_kaggle_auth("kaggle")


def test_resume_skips_existing_selected_metadata_and_pdf(monkeypatch, tmp_path) -> None:
    metadata = tmp_path / "arxiv-metadata-oai-snapshot.json"
    write_snapshot(metadata, [kaggle_record("2401.00001v1")])
    output = tmp_path / "out"
    pdf_dir = output / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "2401.00001v1.pdf").write_bytes(b"%PDF\n")
    selected = filter_candidate(
        kaggle_record("2401.00001v1"),
        category_prefixes=("physics.", "eess."),
        start_year=None,
        end_year=None,
        allowed_licenses={"CC BY 4.0"},
    )[0]
    assert selected is not None
    write_snapshot(output / "selected_arxiv_documents.jsonl", [selected])

    def fail_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("valid existing PDF should be skipped on resume")

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_url_to_file", fail_download)

    result = build_arxiv_open_reuse_corpus(
        snapshot_path=metadata,
        output_dir=output,
        metadata_only=False,
        target_count=1,
        category_prefixes=("physics.", "eess."),
        progress_interval=0,
    )

    assert result.downloaded == 1
    report = json.loads((output / "report.json").read_text())
    assert report["already_present_pdfs"] == 1


def test_pdf_filename_handles_old_style_arxiv_ids() -> None:
    assert pdf_filename_for_arxiv_id("hep-th/9901001v1") == "hep-th_9901001v1.pdf"


def test_kaggle_pdf_object_paths_cover_new_and_old_arxiv_ids() -> None:
    assert versioned_arxiv_id("2401.00003", [{"version": "v3"}]) == "2401.00003v3"
    assert kaggle_pdf_object_name("2401.00003v3") == "arxiv/arxiv/pdf/2401/2401.00003v3.pdf"
    assert kaggle_pdf_object_name("hep-th/9901001v1") == "arxiv/hep-th/pdf/9901/9901001v1.pdf"


def test_download_eligible_pdfs_resumes_existing_valid_pdf(monkeypatch, tmp_path) -> None:
    existing = tmp_path / "2401.00001v1.pdf"
    existing.write_bytes(b"%PDF\n")

    def fail_download(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("valid existing PDFs should be skipped before Kaggle access")

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_url_to_file", fail_download)

    stats = download_eligible_pdfs(
        [
            {
                "arxiv_id": "2401.00001v1",
                "pdf_arxiv_id": "2401.00001v1",
                "pdf_url": "https://arxiv.org/pdf/2401.00001v1",
                "kaggle_gcs_object": "arxiv/arxiv/pdf/2401/2401.00001v1.pdf",
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


def test_download_eligible_pdfs_downloads_exact_kaggle_object(monkeypatch, tmp_path) -> None:
    urls = []

    def fake_download(url, destination, **kwargs):  # type: ignore[no-untyped-def]
        urls.append(url)
        destination.write_bytes(b"%PDF\nbody")

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_url_to_file", fake_download)

    stats = download_eligible_pdfs(
        [
            {
                "arxiv_id": "2401.00001",
                "pdf_arxiv_id": "2401.00001v1",
                "pdf_url": "https://storage.googleapis.com/download/storage/v1/b/arxiv-dataset/o/arxiv%2Farxiv%2Fpdf%2F2401%2F2401.00001v1.pdf?alt=media",
                "kaggle_gcs_object": "arxiv/arxiv/pdf/2401/2401.00001v1.pdf",
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

    assert stats["downloaded"] == 1
    assert stats["unavailable"] == 0
    assert urls == [
        "https://storage.googleapis.com/download/storage/v1/b/arxiv-dataset/o/arxiv%2Farxiv%2Fpdf%2F2401%2F2401.00001v1.pdf?alt=media"
    ]
    assert (tmp_path / "2401.00001v1.pdf").read_bytes().startswith(b"%PDF")


def test_download_eligible_pdfs_uses_multiple_workers(monkeypatch, tmp_path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_download(url, destination, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        destination.write_bytes(b"%PDF\nbody")
        with lock:
            active -= 1

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_url_to_file", fake_download)

    records = [
        {
            "arxiv_id": f"2401.0000{index}",
            "pdf_arxiv_id": f"2401.0000{index}v1",
            "pdf_url": f"https://example.test/{index}.pdf",
            "kaggle_gcs_object": f"arxiv/arxiv/pdf/2401/2401.0000{index}v1.pdf",
            "pdf_filename": f"2401.0000{index}v1.pdf",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "normalized_license_url": "https://creativecommons.org/licenses/by/4.0",
            "license": "CC BY 4.0",
            "license_family": "cc-by",
        }
        for index in range(4)
    ]

    stats = download_eligible_pdfs(
        records,
        output_dir=tmp_path,
        overwrite=False,
        request_delay_seconds=0,
        timeout_seconds=10,
        max_retries=1,
        max_workers=4,
    )

    assert stats["downloaded"] == 4
    assert max_active > 1


def test_download_eligible_pdfs_reports_kaggle_unavailable_ids(monkeypatch, tmp_path) -> None:
    def fake_download(url, destination, **kwargs):  # type: ignore[no-untyped-def]
        raise HTTPFetchError(url, urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None), status_code=404)

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.download_url_to_file", fake_download)

    stats = download_eligible_pdfs(
        [
            {
                "arxiv_id": "2401.99999",
                "pdf_arxiv_id": "2401.99999v1",
                "pdf_url": "https://storage.googleapis.com/download/storage/v1/b/arxiv-dataset/o/missing?alt=media",
                "kaggle_gcs_object": "arxiv/arxiv/pdf/2401/2401.99999v1.pdf",
                "pdf_filename": "2401.99999v1.pdf",
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

    unavailable = [json.loads(line) for line in (tmp_path / "kaggle_unavailable_ids.jsonl").read_text().splitlines()]

    assert stats["downloaded"] == 0
    assert stats["unavailable"] == 1
    assert unavailable[0]["arxiv_id"] == "2401.99999"
    assert unavailable[0]["kaggle_gcs_object"] == "arxiv/arxiv/pdf/2401/2401.99999v1.pdf"


def test_oai_record_parses_to_existing_metadata_shape() -> None:
    xml = """
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>2401.00001</id>
      <created>2024-01-01</created>
      <updated>2024-01-02</updated>
      <authors>
        <author><keyname>Curie</keyname><forenames>Marie</forenames></author>
      </authors>
      <title>A robust instrument</title>
      <categories>physics.ins-det eess.SP</categories>
      <license>http://creativecommons.org/licenses/by/4.0/</license>
      <abstract>Measurement details.</abstract>
      <doi>10.1234/example</doi>
    </arXiv>
    """

    record = parse_oai_arxiv_record(ET.fromstring(xml))

    assert record["id"] == "2401.00001"
    assert record["authors"] == "Marie Curie"
    assert record["categories"] == "physics.ins-det eess.SP"
    assert record["license"] == "http://creativecommons.org/licenses/by/4.0/"


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


def test_oai_harvest_skips_http_406_windows(monkeypatch) -> None:
    def fake_fetch_bytes(url, **kwargs):  # type: ignore[no-untyped-def]
        raise HTTPFetchError(url, urllib.error.HTTPError(url, 406, "Not Acceptable", hdrs=None, fp=None), status_code=406)

    monkeypatch.setattr("dataset_generation.document_downloads.arxiv_open_reuse.fetch_bytes", fake_fetch_bytes)

    items = list(
        iter_oai_records(
            base_url="https://oaipmh.arxiv.org/oai",
            set_spec=None,
            start_resumption_token=None,
            start_until_date=date(2026, 9, 6),
            earliest_date=date(2026, 9, 6),
            window_days=1,
            request_delay_seconds=0,
            timeout_seconds=10,
            max_retries=1,
        )
    )

    assert len(items) == 1
    assert items[0].record is None
    assert items[0].status == "skipped_http_406"
