#!/usr/bin/env python3
"""
Wild Basin GPU PaddleOCR runner.

Designed for Google Colab T4 GPU:
- downloads trail-camera images from Box in parallel
- crops the bottom strip of each image before OCR
- runs PaddleOCR on GPU when available
- appends results to JSONL so the job can resume

Expected input JSON format: list of records containing at least file_id.
Optional fields preserved in output: file_name, file_url/web_url, path.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth
from dotenv import find_dotenv, load_dotenv, set_key
from paddleocr import PaddleOCR

try:
    import paddle
except Exception:  # pragma: no cover
    paddle = None


from box_downloads import (
    download_images_parallel,
    store_tokens,
    build_client,
    load_json_list,
    load_processed_file_ids,
    append_result,
    cleanup_temp_files,
    log
)


# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------


def paddle_gpu_available() -> bool:
    if paddle is None:
        return False
    try:
        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False


def build_ocr(device: str, lang: str = "en") -> PaddleOCR:
    """
    Create PaddleOCR instance.

    PaddleOCR 3.x supports device="gpu:0". Some older releases used use_gpu=True.
    This function tries the newer API first, then falls back for older installs.
    """
    common_kwargs = dict(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    try:
        return PaddleOCR(device=device, **common_kwargs)
    except TypeError:
        use_gpu = device.startswith("gpu")
        return PaddleOCR(use_gpu=use_gpu, **common_kwargs)


def crop_bottom_percent(image_path: str, percent: float = 0.065) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, _ = img.shape[:2]
    crop_h = max(1, int(h * percent))
    cropped = img[h - crop_h : h, :]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_crop.jpg")
    tmp_path = tmp.name
    tmp.close()

    ok = cv2.imwrite(tmp_path, cropped)
    if not ok:
        cleanup_temp_files([tmp_path])
        raise ValueError(f"Failed to write cropped image: {tmp_path}")

    return tmp_path


def extract_texts_from_prediction(prediction_result) -> List[str]:
    texts: List[str] = []

    if isinstance(prediction_result, list):
        for item in prediction_result:
            if isinstance(item, dict):
                for t in item.get("rec_texts") or []:
                    if t is not None:
                        texts.append(str(t).strip())
    elif isinstance(prediction_result, dict):
        for t in prediction_result.get("rec_texts") or []:
            if t is not None:
                texts.append(str(t).strip())

    return [t for t in texts if t]


def run_ocr_one(ocr: PaddleOCR, record: dict, image_path: str, crop_percent: float) -> Dict:
    file_id = str(record.get("file_id", "")).strip()
    file_name = record.get("file_name") or f"{file_id}.jpg"
    file_url = record.get("file_url") or record.get("web_url")
    path_value = record.get("path") or ""
    cropped_path: Optional[str] = None

    try:
        cropped_path = crop_bottom_percent(image_path, crop_percent)
        prediction = ocr.predict(cropped_path)
        texts = extract_texts_from_prediction(prediction)
        return {
            "status": "ok",
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "path": path_value,
            "ocr_texts": texts,
        }
    except Exception as exc:
        return {
            "status": "error",
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "path": path_value,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        cleanup_temp_files([cropped_path] if cropped_path else [])


def process_downloaded_batch(
    ocr: PaddleOCR,
    downloaded: List[Tuple[dict, str]],
    results_file: str,
    crop_percent: float,
    quiet: bool,
) -> Tuple[int, int]:
    ok_count = 0
    error_count = 0

    for idx, (record, image_path) in enumerate(downloaded, start=1):
        result = run_ocr_one(ocr, record, image_path, crop_percent)
        append_result(results_file, result)

        if result["status"] == "ok":
            ok_count += 1
            log(
                f"  OCR {idx}/{len(downloaded)} ok file_id={result.get('file_id')} "
                f"lines={len(result.get('ocr_texts', []))}",
                quiet,
            )
        else:
            error_count += 1
            log(
                f"  OCR {idx}/{len(downloaded)} failed file_id={result.get('file_id')} "
                f"error={result.get('error')}",
                quiet,
            )

    return ok_count, error_count


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Box trail-camera images and run PaddleOCR with GPU support."
    )
    parser.add_argument("--input-file", required=True, help="Input JSON list of Box image records.")
    parser.add_argument("--results-file", required=True, help="Output JSONL results file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--crop-percent", type=float, default=0.065)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu:0"],
        help="Use auto on Colab; it selects gpu:0 when Paddle has CUDA support.",
    )
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_json_list(args.input_file)
    processed_ids = set() if args.reprocess else load_processed_file_ids(args.results_file)

    if args.device == "auto":
        device = "gpu:0" if paddle_gpu_available() else "cpu"
    else:
        device = args.device

    log(f"Paddle compiled with CUDA: {paddle_gpu_available()}", args.quiet)
    if paddle is not None:
        try:
            log(f"Current Paddle device before OCR init: {paddle.device.get_device()}", args.quiet)
        except Exception:
            pass
    log(f"Selected OCR device: {device}", args.quiet)

    client = build_client()
    ocr = build_ocr(device=device, lang=args.lang)

    total = len(records)
    processed = 0
    skipped = 0
    failed = 0
    batch: List[dict] = []

    log(f"Loaded {total} records", args.quiet)
    log(f"Skipping {len(processed_ids)} already-successful file_id values", args.quiet)
    log(f"Batch size: {args.batch_size}", args.quiet)
    log(f"Download workers: {args.download_workers}", args.quiet)

    def handle_batch(batch_records: List[dict]) -> None:
        nonlocal processed, failed
        log(f"Downloading batch of {len(batch_records)} images...", args.quiet)
        downloaded = download_images_parallel(
            client=client,
            records=batch_records,
            max_workers=args.download_workers,
        )
        if not downloaded:
            return

        paths = [path for _, path in downloaded]
        try:
            log(f"Running GPU OCR on {len(downloaded)} images...", args.quiet)
            ok_count, error_count = process_downloaded_batch(
                ocr=ocr,
                downloaded=downloaded,
                results_file=args.results_file,
                crop_percent=args.crop_percent,
                quiet=args.quiet,
            )
            processed += ok_count
            failed += error_count
            log(f"Progress | processed_ok={processed} skipped={skipped} failed={failed}", args.quiet)
        finally:
            cleanup_temp_files(paths)

    for record in records:
        if args.limit and (processed + len(batch)) >= args.limit:
            break

        file_id = str(record.get("file_id", "")).strip()
        if not file_id:
            failed += 1
            append_result(args.results_file, {"status": "error", "error": "missing_file_id", "source_record": record})
            continue

        if file_id in processed_ids:
            skipped += 1
            continue

        batch.append(record)
        if len(batch) >= args.batch_size:
            handle_batch(batch)
            batch = []

    if batch:
        handle_batch(batch)

    log(
        f"\nDone | processed_ok={processed} | skipped={skipped} | failed={failed} | "
        f"results_file={args.results_file}",
        args.quiet,
    )


if __name__ == "__main__":
    main()
