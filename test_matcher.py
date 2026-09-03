import unittest

from dataset_generation.matching import (
    extract_paper_id_from_filename,
    match_records_to_papers,
    normalize_id,
)

class TestMatcher(unittest.TestCase):
    def test_normalize_id(self):
        self.assertEqual(normalize_id("2007.sigdial-1.12.pdf"), "2007.sigdial-1.12")
        self.assertEqual(normalize_id("  P18-1001  "), "p18-1001")
        self.assertEqual(normalize_id("https://aclanthology.org/2020.acl-main.128.pdf"), "2020.acl-main.128")
        self.assertEqual(normalize_id("https://aclanthology.org/P18-1001/"), "p18-1001")
        self.assertEqual(normalize_id("2020.acl-main.16.dataset"), "2020.acl-main.16")
        self.assertEqual(normalize_id("2020.acl-main.129v2"), "2020.acl-main.129")
        self.assertEqual(normalize_id(""), "")
        self.assertEqual(normalize_id(None), "")
        self.assertEqual(normalize_id("HTTP://example.com/P18-1001.pdf"), "http://example.com/p18-1001")

    def test_extract_paper_id_from_filename(self):
        self.assertEqual(
            extract_paper_id_from_filename("/path/to/2007.sigdial-1.12.pdf-Figure4.png"),
            "2007.sigdial-1.12"
        )
        self.assertEqual(
            extract_paper_id_from_filename("2020.acl-main.128.pdf-Figure2.png"),
            "2020.acl-main.128"
        )
        self.assertEqual(
            extract_paper_id_from_filename("P18-1001.png"),
            "P18-1001"
        )

    def test_match_records_to_papers(self):
        records = [
            {
                "filename": "/path/to/2007.sigdial-1.12.pdf-Figure4.png",
                "extracted_paper_id": "2007.sigdial-1.12",
                "normalized_paper_id": "2007.sigdial-1.12",
                "label": "Line graph_chart"
            },
            {
                "filename": "/path/to/w14-33.pdf-Figure1.png",
                "extracted_paper_id": "w14-33",
                "normalized_paper_id": "w14-33",
                "label": "pie chart"
            },
            {
                "filename": "/path/to/2020.nuse-1.pdf-Figure1.png",
                "extracted_paper_id": "2020.nuse-1",
                "normalized_paper_id": "2020.nuse-1",
                "label": "bar charts"
            },
            {
                "filename": "/path/to/unmatched-1.1.pdf-Figure1.png",
                "extracted_paper_id": "unmatched-1.1",
                "normalized_paper_id": "unmatched-1.1",
                "label": "trees"
            }
        ]

        papers = [
            {
                "anthology_id": "2007.sigdial-1.12",
                "markdown": "# Sigdial Paper\nThis is paper content."
            },
            {
                "anthology_id": "w14-3300",
                "markdown": "# Volume proceedings."
            },
            {
                "anthology_id": "2020.nuse-1.0",
                "markdown": "# NUSE Paper."
            }
        ]

        enriched, stats = match_records_to_papers(records, papers)

        self.assertEqual(stats["total_records"], 4)
        self.assertEqual(stats["matched_records"], 3)
        self.assertEqual(stats["unique_matched_ids"], 3)
        self.assertEqual(stats["unique_unmatched_ids"], 1)

        # Check direct matched record values
        r0 = [r for r in enriched if r["filename"].endswith("Figure4.png")][0]
        self.assertEqual(r0["match_status"], "matched")
        self.assertEqual(r0["markdown"], "# Sigdial Paper\nThis is paper content.")
        self.assertEqual(r0["resolved_paper_id"], "2007.sigdial-1.12")

        # Check volume fallback matching
        r1 = [r for r in enriched if r["filename"].endswith("w14-33.pdf-Figure1.png")][0]
        self.assertEqual(r1["match_status"], "matched")
        self.assertEqual(r1["markdown"], "# Volume proceedings.")
        self.assertEqual(r1["resolved_paper_id"], "w14-3300")

        r2 = [r for r in enriched if r["filename"].endswith("2020.nuse-1.pdf-Figure1.png")][0]
        self.assertEqual(r2["match_status"], "matched")
        self.assertEqual(r2["markdown"], "# NUSE Paper.")
        self.assertEqual(r2["resolved_paper_id"], "2020.nuse-1.0")

        # Check unmatched record
        r3 = [r for r in enriched if r["normalized_paper_id"] == "unmatched-1.1"][0]
        self.assertEqual(r3["match_status"], "unmatched")
        self.assertEqual(r3["markdown"], "")

if __name__ == "__main__":
    unittest.main()
