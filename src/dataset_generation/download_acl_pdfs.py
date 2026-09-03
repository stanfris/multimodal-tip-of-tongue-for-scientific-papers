import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def acl_pdf_url(paper_id: str) -> str:
    return f"https://aclanthology.org/{paper_id}.pdf"

def download_pdf(
    paper_id: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Optional[Path]:
    output_path = output_dir / f"{paper_id}.pdf"
    
    if output_path.exists() and not overwrite:
        # Check if the existing file is valid (non-empty and has PDF header)
        if output_path.stat().st_size > 0:
            try:
                with open(output_path, "rb") as f:
                    header = f.read(4)
                if header == b"%PDF":
                    logger.info(f"[skip] {paper_id}")
                    return output_path
            except Exception:
                pass
    
    urls_to_try = [acl_pdf_url(paper_id)]
    if paper_id.upper() != paper_id:
        urls_to_try.append(acl_pdf_url(paper_id.upper()))

    tmp_path = output_path.with_suffix(".pdf.part")
    
    # Retry transient errors with backoff
    max_retries = 3
    success = False
    
    for url_idx, url in enumerate(urls_to_try):
        if success:
            break
            
        for attempt in range(max_retries + 1):
            try:
                with requests.get(url, stream=True, timeout=60) as response:
                    if response.status_code == 404:
                        if url_idx < len(urls_to_try) - 1:
                            # Try the uppercase fallback
                            break
                        else:
                            # Permanent error, do not retry
                            logger.error(f"[failed] {paper_id}: HTTP 404")
                            return None
                    
                    response.raise_for_status()
                    
                    with tmp_path.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                success = True
                break # Success, break out of retry loop
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    # Basic backoff
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    if url_idx < len(urls_to_try) - 1:
                        # Try the uppercase fallback
                        break
                    else:
                        logger.error(f"[failed] {paper_id}: {e}")
                        if tmp_path.exists():
                            tmp_path.unlink()
                        return None
    
    # Lightweight validation
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        logger.error(f"[failed] {paper_id}: Downloaded file is empty")
        if tmp_path.exists():
            tmp_path.unlink()
        return None
        
    try:
        with open(tmp_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            logger.error(f"[failed] {paper_id}: Not a valid PDF (missing %PDF header)")
            tmp_path.unlink()
            return None
    except Exception as e:
        logger.error(f"[failed] {paper_id}: Error validating PDF: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return None
        
    tmp_path.rename(output_path)
    logger.info(f"[download] {paper_id}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download ACL Anthology PDFs.")
    parser.add_argument("--input", type=Path, help="Path to input JSON/JSONL or text file containing paper IDs.")
    parser.add_argument("--paper-id", type=str, help="A single paper ID to download.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the PDFs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    if not args.input and not args.paper_id:
        parser.error("Must provide either --input or --paper-id")
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    paper_ids = []
    
    if args.paper_id:
        paper_ids.append(args.paper_id)
        
    if args.input:
        if args.input.suffix == ".jsonl":
            with open(args.input, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if "paper_id" in data:
                        paper_ids.append(data["paper_id"])
        elif args.input.suffix == ".json":
            with open(args.input, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "paper_id" in item:
                            paper_ids.append(item["paper_id"])
        else:
            # Assume text file with one ID per line
            with open(args.input, "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        paper_ids.append(pid)
                        
    # Ensure uniqueness
    paper_ids = list(dict.fromkeys(paper_ids))
    
    total = len(paper_ids)
    downloaded = 0
    skipped = 0
    failed = 0
    failures = []
    
    for paper_id in paper_ids:
        # Check if it should be skipped
        output_path = args.output_dir / f"{paper_id}.pdf"
        was_skipped = False
        if output_path.exists() and not args.overwrite:
            if output_path.stat().st_size > 0:
                try:
                    with open(output_path, "rb") as f:
                        header = f.read(4)
                    if header == b"%PDF":
                        was_skipped = True
                except Exception:
                    pass
        
        result = download_pdf(paper_id, args.output_dir, args.overwrite)
        
        if result:
            if was_skipped:
                skipped += 1
            else:
                downloaded += 1
        else:
            failed += 1
            failures.append({
                "paper_id": paper_id,
                "url": acl_pdf_url(paper_id),
                "error": "Failed to download or validate PDF"
            })
            
    print("\nPDF download complete\n")
    print(f"Total:      {total}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:       {failed}")
    
    if failures:
        failures_path = args.output_dir / "download_failures.json"
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
            
if __name__ == "__main__":
    main()
