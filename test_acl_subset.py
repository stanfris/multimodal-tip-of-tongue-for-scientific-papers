from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_generation import acl_subset


def write_collection(path: Path, collection_id: str, volume_id: str, venue: str = "acl") -> None:
    path.write_text(
        f"""<?xml version='1.0' encoding='UTF-8'?>
<collection id="{collection_id}">
  <volume id="{volume_id}" type="proceedings">
    <meta>
      <month>July</month>
      <year>2026</year>
      <venue>{venue}</venue>
    </meta>
    <frontmatter>
      <url>{collection_id}-{volume_id}.0</url>
    </frontmatter>
    <paper id="1">
      <title><fixed-case>C</fixed-case>ase Study</title>
      <author><first>Ada</first><last>Lovelace</last></author>
      <pages>1-12</pages>
      <abstract> A compact   abstract. </abstract>
      <url>{collection_id}-{volume_id}.1</url>
      <doi>10.18653/v1/{collection_id}-{volume_id}.1</doi>
    </paper>
  </volume>
</collection>
""",
        encoding="utf-8",
    )


def test_iter_target_papers_uses_exact_configured_volumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl_subset, "TARGET_VOLUMES", {"acl_2026": "2026.acl-long"})
    write_collection(tmp_path / "2026.acl.xml", "2026.acl", "long")

    papers = list(acl_subset.iter_target_papers(tmp_path))

    assert len(papers) == 1
    assert papers[0]["anthology_id"] == "2026.acl-long.1"
    assert papers[0]["title"] == "Case Study"
    assert papers[0]["authors"] == [{"first": "Ada", "last": "Lovelace"}]
    assert papers[0]["year"] == 2026
    assert papers[0]["month"] == "July"
    assert papers[0]["venue"] == "ACL"
    assert papers[0]["volume_id"] == "2026.acl-long"
    assert papers[0]["url"] == "https://aclanthology.org/2026.acl-long.1/"
    assert papers[0]["pdf_url"] == "https://aclanthology.org/2026.acl-long.1.pdf"


def test_build_acl_subset_writes_jsonl_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl_subset, "TARGET_VOLUMES", {"emnlp_2024": "2024.emnlp-main"})
    xml_cache = tmp_path / "xml"
    xml_cache.mkdir()
    write_collection(xml_cache / "2024.emnlp.xml", "2024.emnlp", "main", venue="emnlp")

    result = acl_subset.build_acl_subset(tmp_path / "out", xml_cache_dir=xml_cache)

    rows = [json.loads(line) for line in result.papers_path.read_text(encoding="utf-8").splitlines()]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.paper_count == 1
    assert rows[0]["anthology_id"] == "2024.emnlp-main.1"
    assert rows[0]["venue"] == "EMNLP"
    assert metadata["counts_by_volume"] == {"2024.emnlp-main": 1}


def test_missing_configured_volume_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl_subset, "TARGET_VOLUMES", {"acl_2026": "2026.acl-long"})
    write_collection(tmp_path / "2026.acl.xml", "2026.acl", "short")

    with pytest.raises(ValueError, match="Volume 2026.acl-long not found"):
        list(acl_subset.iter_target_papers(tmp_path))
