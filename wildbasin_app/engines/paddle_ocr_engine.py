from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import cv2
from paddleocr import PaddleOCR

try:
    import paddle
except Exception:  # pragma: no cover
    paddle = None

from .common import (
    CancelCallback,
    LogCallback,
    TokenCallback,
    append_result,
    build_box_client,
    check_cancelled,
    cleanup_temp_files,
    download_images_parallel,
    emit_log,
    load_json_list,
    load_processed_file_ids,
)


def paddle_gpu_available() -> bool:
    if paddle is None:
        return False

    try:
        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False


def build_ocr(device: str, lang: str = "en") -> PaddleOCR:
    common_kwargs = dict(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    try:
        return PaddleOCR(device=device, **common_kwargs)
    except TypeError:
        return PaddleOCR(
            use_gpu=device.startswith("gpu"),
            **common_kwargs,
        )


def crop_bottom_percent(
    image_path: str,
    percent: float = 0.065,
) -> str:
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")

    if os.path.getsize(image_path) == 0:
        raise ValueError(f"Image file is empty: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            "OpenCV failed to read image. It may be corrupt, unsupported, "
            f"or not actually an image: {image_path}"
        )

    height, _ = image.shape[:2]
    crop_height = max(1, int(height * percent))
    cropped = image[height - crop_height:height, :]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_crop.jpg",
    )
    temp_path = temp_file.name
    temp_file.close()

    if not cv2.imwrite(temp_path, cropped):
        cleanup_temp_files([temp_path])
        raise ValueError(
            f"Failed to write cropped image: {temp_path}"
        )

    return temp_path


def extract_texts_from_prediction(prediction_result) -> List[str]:
    texts: List[str] = []

    if isinstance(prediction_result, list):
        for item in prediction_result:
            if isinstance(item, dict):
                for text in item.get("rec_texts") or []:
                    if text is not None:
                        texts.append(str(text).strip())

    elif isinstance(prediction_result, dict):
        for text in prediction_result.get("rec_texts") or []:
            if text is not None:
                texts.append(str(text).strip())

    return [text for text in texts if text]


def run_ocr_one(
    ocr: PaddleOCR,
    record: dict,
    image_path: str,
    crop_percent: float,
) -> Dict:
    file_id = str(record.get("file_id", "")).strip()
    file_name = record.get("file_name") or f"{file_id}.jpg"
    file_url = record.get("file_url") or record.get("web_url")
    path_value = record.get("path") or ""
    cropped_path: Optional[str] = None

    try:
        cropped_path = crop_bottom_percent(
            image_path,
            crop_percent,
        )
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
        cleanup_temp_files(
            [cropped_path] if cropped_path else []
        )


def run_paddle_ocr(
    *,
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str,
    input_file: str,
    results_file: str,
    batch_size: int = 128,
    download_workers: int = 32,
    crop_percent: float = 0.065,
    limit: int = 0,
    reprocess: bool = False,
    device: str = "auto",
    lang: str = "en",
    log_callback: Optional[LogCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    token_callback: Optional[TokenCallback] = None,
) -> dict:
    """
    Importable equivalent of batch-box-paddle-ocr-gpu.py.

    On macOS, device="auto" falls back to CPU because CUDA is unavailable.
    """
    records = load_json_list(input_file)
    processed_ids = (
        set()
        if reprocess
        else load_processed_file_ids(results_file)
    )

    if device == "auto":
        selected_device = (
            "gpu:0"
            if paddle_gpu_available()
            else "cpu"
        )
    else:
        selected_device = device

    emit_log(
        f"Paddle compiled with CUDA: {paddle_gpu_available()}",
        log_callback,
    )
    emit_log(
        f"Selected OCR device: {selected_device}",
        log_callback,
    )

    check_cancelled(should_cancel)

    client = build_box_client(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
        token_callback=token_callback,
    )

    emit_log("Loading PaddleOCR models...", log_callback)
    ocr = build_ocr(
        device=selected_device,
        lang=lang,
    )
    emit_log("PaddleOCR models loaded.", log_callback)

    processed = 0
    skipped = 0
    failed = 0
    download_failed = 0
    batch: List[dict] = []

    emit_log(f"Loaded {len(records)} records", log_callback)
    emit_log(
        f"Skipping {len(processed_ids)} already-successful file_id values",
        log_callback,
    )
    emit_log(f"Batch size: {batch_size}", log_callback)
    emit_log(
        f"Download workers: {download_workers}",
        log_callback,
    )

    def handle_batch(batch_records: List[dict]) -> None:
        nonlocal processed, failed, download_failed

        check_cancelled(should_cancel)

        emit_log(
            f"Downloading OCR batch of {len(batch_records)} images...",
            log_callback,
        )

        downloaded, current_download_failures = download_images_parallel(
            client=client,
            records=batch_records,
            max_workers=download_workers,
            should_cancel=should_cancel,
            log_callback=log_callback,
        )
        download_failed += current_download_failures

        if not downloaded:
            return

        paths = [path for _, path in downloaded]

        try:
            emit_log(
                f"Running OCR on {len(downloaded)} images...",
                log_callback,
            )

            for index, (record, image_path) in enumerate(
                downloaded,
                start=1,
            ):
                check_cancelled(should_cancel)

                result = run_ocr_one(
                    ocr,
                    record,
                    image_path,
                    crop_percent,
                )
                append_result(results_file, result)

                if result["status"] == "ok":
                    processed += 1
                    emit_log(
                        f"  OCR {index}/{len(downloaded)} ok "
                        f"file_id={result.get('file_id')} "
                        f"lines={len(result.get('ocr_texts', []))}",
                        log_callback,
                    )
                else:
                    failed += 1
                    emit_log(
                        f"  OCR {index}/{len(downloaded)} failed "
                        f"file_id={result.get('file_id')} "
                        f"error={result.get('error')}",
                        log_callback,
                    )

            emit_log(
                f"OCR progress | processed_ok={processed} | "
                f"skipped={skipped} | failed={failed} | "
                f"download_failed={download_failed}",
                log_callback,
            )

        finally:
            cleanup_temp_files(paths)

    for record in records:
        check_cancelled(should_cancel)

        if limit and (processed + len(batch)) >= limit:
            break

        file_id = str(record.get("file_id", "")).strip()

        if not file_id:
            failed += 1
            append_result(
                results_file,
                {
                    "status": "error",
                    "error": "missing_file_id",
                    "source_record": record,
                },
            )
            continue

        if file_id in processed_ids:
            skipped += 1
            continue

        batch.append(record)

        if len(batch) >= batch_size:
            handle_batch(batch)
            batch = []

    if batch:
        handle_batch(batch)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "download_failed": download_failed,
        "results_file": results_file,
        "device": selected_device,
    }

    emit_log(
        "OCR done | "
        f"processed_ok={processed} | "
        f"skipped={skipped} | "
        f"failed={failed} | "
        f"download_failed={download_failed} | "
        f"results={results_file}",
        log_callback,
    )

    return summary
