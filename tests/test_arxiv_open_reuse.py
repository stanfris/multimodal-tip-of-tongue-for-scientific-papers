from __future__ import annotations

import urllib.error
from xml.etree import ElementTree

from dataset_generation.document_downloads.arxiv_open_reuse import classify_broad_domains
from dataset_generation.document_downloads.arxiv_open_reuse import fetch_bytes
from dataset_generation.document_downloads.arxiv_open_reuse import classify_license
from dataset_generation.document_downloads.arxiv_open_reuse import normalize_license_url
from dataset_generation.document_downloads.arxiv_open_reuse import parse_oai_record
from dataset_generation.document_downloads.arxiv_open_reuse import select_records_for_domain_targets
from dataset_generation.document_downloads.arxiv_open_reuse import url_with_params


def test_license_normalization_accepts_only_cc_by_and_cc0() -> None:
    assert classify_license(normalize_license_url("http://creativecommons.org/licenses/by/4.0/")) == "cc-by"
    assert classify_license(normalize_license_url("https://creativecommons.org/publicdomain/zero/1.0/")) == "cc0"
    assert classify_license(normalize_license_url("http://creativecommons.org/licenses/publicdomain/")) == "cc0"

    assert (
        classify_license(normalize_license_url("http://arxiv.org/licenses/nonexclusive-distrib/1.0/"))
        == "arxiv-default"
    )
    assert (
        classify_license(normalize_license_url("https://creativecommons.org/licenses/by-nc/4.0/"))
        == "other-creative-commons"
    )
    assert classify_license(normalize_license_url("https://creativecommons.org/licenses/by-sa/4.0/")) == (
        "other-creative-commons"
    )


def test_category_classification_keeps_cross_domain_records() -> None:
    assert classify_broad_domains(["hep-th", "quant-ph"]) == ["physics"]
    assert classify_broad_domains(["eess.SP"]) == ["engineering"]
    assert classify_broad_domains(["astro-ph.IM", "cs.RO"]) == ["physics", "engineering"]
    assert classify_broad_domains(["cs.CL"]) == []


def test_parse_oai_record_extracts_required_metadata_and_pdf_location() -> None:
    xml = """
    <record xmlns="http://www.openarchives.org/OAI/2.0/">
      <header>
        <identifier>oai:arXiv.org:2401.01234</identifier>
        <datestamp>2024-01-05</datestamp>
        <setSpec>physics</setSpec>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2401.01234</id>
          <created>2024-01-02</created>
          <title> A robust instrument </title>
          <authors>
            <author>
              <keyname>Curie</keyname>
              <forenames>Marie</forenames>
            </author>
          </authors>
          <categories>physics.ins-det eess.SP</categories>
          <license>http://creativecommons.org/licenses/by/4.0/</license>
          <abstract> Measurement details. </abstract>
        </arXiv>
      </metadata>
    </record>
    """

    record = parse_oai_record(ElementTree.fromstring(xml), source_set="physics")

    assert record is not None
    assert record["arxiv_id"] == "2401.01234"
    assert record["title"] == "A robust instrument"
    assert record["authors"] == [{"forenames": "Marie", "keyname": "Curie", "name": "Marie Curie", "suffix": None}]
    assert record["subject_categories"] == ["physics.ins-det", "eess.SP"]
    assert record["primary_category"] == "physics.ins-det"
    assert record["broad_domains"] == ["physics", "engineering"]
    assert record["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert record["normalized_license_url"] == "https://creativecommons.org/licenses/by/4.0"
    assert record["license_family"] == "cc-by"
    assert record["pdf_url"] == "https://arxiv.org/pdf/2401.01234"
    assert record["pdf_filename"] == "2401.01234.pdf"


def test_select_records_for_domain_targets_deduplicates_and_caps_per_domain() -> None:
    records = [
        {"arxiv_id": "3", "created": "2020-01-03", "broad_domains": ["physics"]},
        {"arxiv_id": "1", "created": "2020-01-01", "broad_domains": ["physics", "engineering"]},
        {"arxiv_id": "2", "created": "2020-01-02", "broad_domains": ["engineering"]},
        {"arxiv_id": "4", "created": "2020-01-04", "broad_domains": ["engineering"]},
    ]

    selected = select_records_for_domain_targets(records, target_per_domain=2)

    assert [record["arxiv_id"] for record in selected] == ["1", "2", "3"]


def test_url_with_params_does_not_double_encode_resumption_tokens() -> None:
    url = url_with_params(
        "https://oaipmh.arxiv.org/oai",
        {
            "verb": "ListRecords",
            "resumptionToken": "verb%3DListRecords%26metadataPrefix%3DarXiv%26set%3Dphysics%26skip%3D1300",
        },
    )

    assert "resumptionToken=verb%3DListRecords%26metadataPrefix%3DarXiv%26set%3Dphysics%26skip%3D1300" in url
    assert "%253D" not in url


def test_fetch_bytes_retries_406_with_fallback_accept_header(monkeypatch) -> None:
    requests = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"<OAI-PMH />"

    def fake_urlopen(request, *, timeout):  # type: ignore[no-untyped-def]
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 406, "Not Acceptable", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = fetch_bytes(
        "https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv&set=physics",
        request_delay_seconds=0,
        timeout_seconds=10,
        max_retries=1,
    )

    assert payload == b"<OAI-PMH />"
    assert requests[0].headers["Accept"] == "application/xml,text/xml,*/*"
    assert requests[1].headers["Accept"] == "*/*"
