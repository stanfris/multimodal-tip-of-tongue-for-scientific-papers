from __future__ import annotations

from pathlib import Path

import httpx

from dataset_generation.document_downloads.pmc_oa_subset import (
    DEFAULT_AWS_METADATA_WORKERS,
    DEFAULT_END_YEAR,
    DEFAULT_PUBMED_BATCH_SIZE,
    DEFAULT_START_YEAR,
    DEFAULT_TARGET_PER_DOMAIN,
    build_parser,
    build_report,
    classify_broad_domain,
    fetch_pubmed_metadata,
    flag_enabled,
    is_eligible_article_metadata,
    is_eligible_license,
    normalize_pmcid_number,
    normalize_pmc_object_url,
    parse_esearch_result,
    read_aws_metadata_cache,
    read_pubmed_metadata_cache,
    should_exclude_article_type,
    verify_pdf,
)


def test_parser_defaults_to_recent_pmc_year_window() -> None:
    args = build_parser().parse_args([])

    assert args.start_year == DEFAULT_START_YEAR == 2026
    assert args.end_year == DEFAULT_END_YEAR == 2026
    assert args.target_per_domain == DEFAULT_TARGET_PER_DOMAIN == 30_000
    assert args.aws_metadata_workers == DEFAULT_AWS_METADATA_WORKERS == 64
    assert args.pubmed_batch_size == DEFAULT_PUBMED_BATCH_SIZE == 200


def test_license_policy_only_accepts_cc_by_and_cc0() -> None:
    assert is_eligible_license("cc by")
    assert is_eligible_license("CC0")
    assert not is_eligible_license("cc by-nc")
    assert not is_eligible_license("cc by-nd")
    assert not is_eligible_license("cc by-nc-nd")
    assert not is_eligible_license("cc by-sa")
    assert not is_eligible_license("publisher-specific")
    assert not is_eligible_license(None)


def test_parse_esearch_result_uses_xml_not_fragile_json() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
      <Count>3</Count>
      <RetMax>2</RetMax>
      <RetStart>0</RetStart>
      <IdList>
        <Id>13570669</Id>
        <Id>13568481</Id>
      </IdList>
      <QueryTranslation>("cc by license"[Filter])</QueryTranslation>
    </eSearchResult>
    """

    count, ids = parse_esearch_result(xml)

    assert count == 3
    assert ids == ["13570669", "13568481"]


def test_aws_metadata_eligibility_accepts_boolean_flags() -> None:
    assert is_eligible_article_metadata(
        {
            "is_retracted": False,
            "is_pmc_openaccess": True,
            "license_code": "CC BY",
            "pdf_url": "s3://pmc-oa-opendata/PMC1.1/PMC1.1.pdf?md5=abc",
        }
    )
    assert flag_enabled("yes")
    assert flag_enabled(True)
    assert not flag_enabled(False)


def test_normalize_pmc_object_url_converts_s3_bucket_urls_to_https() -> None:
    assert normalize_pmc_object_url("s3://pmc-oa-opendata/PMC1.1/PMC1.1.pdf?md5=abc") == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/PMC1.1.pdf?md5=abc"
    )


def test_resume_cache_readers_skip_retryable_failures(tmp_path: Path) -> None:
    aws_path = tmp_path / "aws.jsonl"
    aws_path.write_text(
        '{"pmcid": "PMC1", "status": "eligible", "metadata": {"pmcid": "PMC1"}}\n'
        '{"pmcid": "PMC2", "status": "failed", "error": "timeout"}\n',
        encoding="utf-8",
    )
    pubmed_path = tmp_path / "pubmed.jsonl"
    pubmed_path.write_text('{"pmid": "10", "title": "A"}\n{"pmid": "10", "title": "B"}\n', encoding="utf-8")

    assert set(read_aws_metadata_cache(aws_path)) == {"1"}
    assert read_pubmed_metadata_cache(pubmed_path)["10"]["title"] == "B"
    assert normalize_pmcid_number("PMC123") == "123"


def test_resume_cache_readers_skip_truncated_rows(tmp_path: Path) -> None:
    aws_path = tmp_path / "aws.jsonl"
    aws_path.write_text(
        '{"pmcid": "PMC1", "status": "eligible", "metadata": {"pmcid": "PMC1"}}\n'
        '{"pmcid": "PMC2", "status": "eligible", "metadata": {"pmcid": "PMC2"}\n',
        encoding="utf-8",
    )
    pubmed_path = tmp_path / "pubmed.jsonl"
    pubmed_path.write_text('{"pmid": "10", "title": "A"}\n{"pmid": "11", "title": ', encoding="utf-8")

    assert set(read_aws_metadata_cache(aws_path)) == {"1"}
    assert set(read_pubmed_metadata_cache(pubmed_path)) == {"10"}


def test_fetch_pubmed_metadata_retries_truncated_xml(tmp_path: Path) -> None:
    responses = iter(
        [
            httpx.Response(200, text="<PubmedArticleSet><PubmedArticle>"),
            httpx.Response(
                200,
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>10</PMID><Article><ArticleTitle>Recovered</ArticleTitle>"
                    "<Language>eng</Language></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                ),
            ),
        ]
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    cache_path = tmp_path / "pubmed.jsonl"

    records = fetch_pubmed_metadata(
        client,
        ["10"],
        email=None,
        api_key=None,
        tool="test",
        request_sleep_seconds=0,
        retries=1,
        cache_path=cache_path,
    )

    assert records["10"]["title"] == "Recovered"
    assert read_pubmed_metadata_cache(cache_path)["10"]["title"] == "Recovered"


def test_fetch_pubmed_metadata_skips_persistently_truncated_xml() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<PubmedArticleSet><PubmedArticle>"))
    )

    records = fetch_pubmed_metadata(
        client,
        ["10"],
        email=None,
        api_key=None,
        tool="test",
        request_sleep_seconds=0,
        retries=1,
    )

    assert records == {}


def test_article_type_filter_excludes_non_research_material() -> None:
    assert should_exclude_article_type(["Editorial"])
    assert should_exclude_article_type(["Letter", "Journal Article"])
    assert not should_exclude_article_type(["Journal Article"])
    assert not should_exclude_article_type(["Review"])


def test_domain_classifier_uses_subject_metadata() -> None:
    biology = classify_broad_domain(
        {"mesh_terms": ["Genomics", "Cell Line"], "keywords": ["protein expression"]},
        {"title": "Regulation of cellular signaling"},
        {},
    )
    medical = classify_broad_domain(
        {"mesh_terms": ["Humans", "Neoplasms", "Drug Therapy"], "keywords": ["clinical trial"]},
        {"title": "Oncology therapy outcomes"},
        {},
    )

    assert biology == "Biology"
    assert medical == "Medical/Clinical Research"


def test_build_report_states_whether_targets_are_met() -> None:
    records = [
        {"exact_license": "cc by", "broad_domain": "Biology", "publication_year": 2024, "detailed_subject_metadata": {"mesh_terms": ["Genomics"]}},
        {"exact_license": "cc0", "broad_domain": "Medical/Clinical Research", "publication_year": 2024, "detailed_subject_metadata": {"mesh_terms": ["Humans"]}},
    ]

    report = build_report(records, total_discovered=3, target_per_domain=1)

    assert report["cc_by_count"] == 1
    assert report["cc0_count"] == 1
    assert report["biology_count"] == 1
    assert report["medical_clinical_count"] == 1
    assert report["strict_filter_has_enough_material"]["both_targets_met"]


def test_verify_pdf_checks_magic_bytes_and_optional_md5(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbody")

    assert verify_pdf(pdf_path)
    assert verify_pdf(pdf_path, "f2858a027dffe3ecddd99a0bf58b7d06")
    assert not verify_pdf(pdf_path, "0" * 32)
