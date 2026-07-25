import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List

from speciesnet import DEFAULT_MODEL, SpeciesNet


# Credentials and client setup delegated to box_downloads

from box_downloads import (
    download_images_parallel,
    store_tokens,
    build_client,
    load_json_list,
    load_processed_file_ids,
    append_result,
    cleanup_temp_files,
)

# All utility functions imported from box_downloads

def process_batch(
    model: SpeciesNet,
    batch_records: List[dict],
    batch_paths: List[str],
    results_file: str,
    run_mode: str,
):
    prediction_dict = model.predict(
        filepaths=batch_paths,
        run_mode=run_mode,
        progress_bars=False,
    )

    predictions = (prediction_dict or {}).get("predictions", [])

    for record, prediction_entry in zip(batch_records, predictions):
        file_id = str(record.get("file_id", "")).strip()
        file_name = record.get("file_name") or f"{file_id}.jpg"
        file_url = record.get("file_url") or record.get("web_url")

        result = {
            "status": "ok",
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "prediction": prediction_entry
        }

        append_result(results_file, result)

def main():

    parser = argparse.ArgumentParser(
        description="Run SpeciesNet on Box images using batched inference."
    )

    parser.add_argument("--input-file", default="box_images.json")
    parser.add_argument("--results-file", default="speciesnet_results.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument(
        "--run-mode",
        default="single_thread",
        choices=[
            "single_thread",
            "multi_thread",
            "multi_process",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--download-workers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
    )

    args = parser.parse_args()

    records = load_json_list(
        args.input_file
    )

    processed_ids = (
        set()
        if args.reprocess
        else load_processed_file_ids(
            args.results_file
        )
    )

    if args.limit:
        records = records[: args.limit]

    records = [
        r
        for r in records
        if str(
            r.get("file_id", "")
        ).strip()
        not in processed_ids
    ]

    print(
        f"Loaded {len(records)} records "
        f"to process"
    )

    client = build_client()

    model = SpeciesNet(
        args.model,
        components="all",
    )

    processed = 0
    failed = 0

    batch_size = args.batch_size

    for start_idx in range(
        0,
        len(records),
        batch_size,
    ):

        end_idx = min(
            start_idx + batch_size,
            len(records),
        )

        chunk = records[
            start_idx:end_idx
        ]

        print(
            "\n"
            f"Downloading batch "
            f"{start_idx:,}"
            f" - "
            f"{end_idx:,}"
        )

        downloaded = (
            download_images_parallel(
                client=client,
                records=chunk,
                max_workers=args.download_workers,
            )
        )

        if not downloaded:

            print(
                "No images downloaded "
                "successfully."
            )

            continue

        batch_records = [
            r
            for r, _ in downloaded
        ]

        batch_paths = [
            p
            for _, p in downloaded
        ]

        print(
            f"Running SpeciesNet "
            f"on {len(batch_paths)} images..."
        )

        try:

            process_batch(
                model=model,
                batch_records=batch_records,
                batch_paths=batch_paths,
                results_file=args.results_file,
                run_mode=args.run_mode,
            )

            processed += len(
                batch_paths
            )

            print(
                f"Processed "
                f"{processed:,} images"
            )

        except Exception as e:

            failed += len(
                batch_paths
            )

            print(
                "Batch failed:"
            )

            print(e)

        finally:

            cleanup_temp_files(
                batch_paths
            )

    print(
        "\nDone\n"
        f"Processed: "
        f"{processed:,}\n"
        f"Failed: "
        f"{failed:,}\n"
        f"Results: "
        f"{args.results_file}"
    )


if __name__ == "__main__":
    main()