from __future__ import annotations

from typing import Optional

from speciesnet import DEFAULT_MODEL, SpeciesNet

from .common import (
    CancelCallback,
    LogCallback,
    TokenCallback,
    PipelineCancelled,
    append_result,
    build_box_client,
    check_cancelled,
    cleanup_temp_files,
    download_images_parallel,
    emit_log,
    load_json_list,
    load_processed_file_ids,
)


def run_speciesnet(
    *,
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str,
    input_file: str,
    results_file: str,
    model_name: str = DEFAULT_MODEL,
    run_mode: str = "single_thread",
    batch_size: int = 512,
    download_workers: int = 32,
    limit: int = 0,
    reprocess: bool = False,
    log_callback: Optional[LogCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    token_callback: Optional[TokenCallback] = None,
) -> dict:
    """
    Importable equivalent of batch-box-run-speciesnet.py.
    """
    records = load_json_list(input_file)
    processed_ids = (
        set()
        if reprocess
        else load_processed_file_ids(results_file)
    )

    if limit:
        records = records[:limit]

    records = [
        record
        for record in records
        if str(record.get("file_id", "")).strip() not in processed_ids
    ]

    emit_log(
        f"Loaded {len(records)} records to process",
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

    emit_log("Loading SpeciesNet model...", log_callback)
    model = SpeciesNet(
        model_name,
        components="all",
    )
    emit_log("SpeciesNet model loaded.", log_callback)

    processed = 0
    failed = 0
    download_failed = 0

    for start_idx in range(0, len(records), batch_size):
        check_cancelled(should_cancel)

        end_idx = min(start_idx + batch_size, len(records))
        chunk = records[start_idx:end_idx]

        emit_log(
            f"Downloading SpeciesNet batch {start_idx:,} - {end_idx:,}",
            log_callback,
        )

        downloaded, batch_download_failures = download_images_parallel(
            client=client,
            records=chunk,
            max_workers=download_workers,
            should_cancel=should_cancel,
            log_callback=log_callback,
        )

        download_failed += batch_download_failures

        if not downloaded:
            emit_log(
                "No images downloaded successfully for this batch.",
                log_callback,
            )
            continue

        batch_records = [record for record, _ in downloaded]
        batch_paths = [path for _, path in downloaded]

        emit_log(
            f"Running SpeciesNet on {len(batch_paths)} images...",
            log_callback,
        )

        try:
            check_cancelled(should_cancel)

            prediction_dict = model.predict(
                filepaths=batch_paths,
                run_mode=run_mode,
                progress_bars=False,
            )

            predictions = (prediction_dict or {}).get("predictions", [])

            if len(predictions) != len(batch_records):
                emit_log(
                    "Warning: SpeciesNet returned a different number of "
                    "predictions than downloaded images.",
                    log_callback,
                )

            for record, prediction_entry in zip(
                batch_records,
                predictions,
            ):
                check_cancelled(should_cancel)

                file_id = str(record.get("file_id", "")).strip()
                file_name = record.get("file_name") or f"{file_id}.jpg"
                file_url = record.get("file_url") or record.get("web_url")

                append_result(
                    results_file,
                    {
                        "status": "ok",
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_url": file_url,
                        "prediction": prediction_entry,
                    },
                )
                processed += 1

            unmatched = len(batch_records) - len(predictions)
            if unmatched > 0:
                failed += unmatched

            emit_log(
                f"SpeciesNet progress | processed={processed:,} | "
                f"failed={failed:,} | download_failed={download_failed:,}",
                log_callback,
            )

        except PipelineCancelled:
            raise
        except Exception as exc:
            failed += len(batch_paths)
            emit_log(
                f"SpeciesNet batch failed: {type(exc).__name__}: {exc}",
                log_callback,
            )
        finally:
            cleanup_temp_files(batch_paths)

    summary = {
        "processed": processed,
        "failed": failed,
        "download_failed": download_failed,
        "results_file": results_file,
    }

    emit_log(
        "SpeciesNet done | "
        f"processed={processed:,} | "
        f"failed={failed:,} | "
        f"download_failed={download_failed:,} | "
        f"results={results_file}",
        log_callback,
    )

    return summary
