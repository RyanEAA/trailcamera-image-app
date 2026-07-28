from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Set

from boxsdk.exception import BoxAPIException, BoxOAuthException

from .common import (
    CancelCallback,
    LogCallback,
    TokenCallback,
    build_box_client,
    check_cancelled,
    emit_log,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _load_existing_records(output_file: str) -> Dict[str, dict]:
    if not os.path.exists(output_file):
        return {}

    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, list):
        return {}

    return {
        str(item["file_id"]): item
        for item in data
        if isinstance(item, dict) and item.get("file_id")
    }


def _flush_pending(
    state: dict,
    *,
    log_callback: Optional[LogCallback],
    force: bool = False,
) -> None:
    pending = state["pending_records"]
    if not pending and not force:
        return

    new_count = len(pending)

    for record in pending:
        state["records_by_id"][str(record["file_id"])] = record

    if pending:
        state["pending_records"] = []
        state["batches_written"] += 1

    output_path = Path(state["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            list(state["records_by_id"].values()),
            handle,
            indent=4,
        )

    if new_count:
        emit_log(
            f"Flushed batch {state['batches_written']}: "
            f"{new_count} new records written | "
            f"total logged={len(state['records_by_id'])}",
            log_callback,
        )


def collect_box_image_records(
    *,
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str,
    folder_id: str,
    output_file: str,
    batch_size: int = 100,
    log_callback: Optional[LogCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    token_callback: Optional[TokenCallback] = None,
) -> dict:
    """
    Crawl a Box folder tree and write image metadata to a JSON list.

    This is the importable equivalent of box-get-urls.py.
    """
    if not folder_id:
        raise ValueError("Box folder ID is required.")

    client = build_box_client(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
        token_callback=token_callback,
    )

    records_by_id = _load_existing_records(output_file)
    state = {
        "output_file": output_file,
        "batch_size": max(1, batch_size),
        "records_by_id": records_by_id,
        "pending_records": [],
        "batches_written": 0,
    }

    seen_file_ids: Set[str] = set(records_by_id.keys())
    stats = {
        "files_searched": 0,
        "images_found": 0,
        "already_logged": 0,
    }

    def crawl(current_folder_id: str, parent_path: str = "", depth: int = 0) -> None:
        check_cancelled(should_cancel)

        indent = "  " * depth
        folder_label = parent_path if parent_path else "/"
        emit_log(
            f"{indent}Entering folder: {folder_label} "
            f"(id={current_folder_id})",
            log_callback,
        )

        items = client.folder(folder_id=current_folder_id).get_items(limit=1000)

        for item in items:
            check_cancelled(should_cancel)

            if item.type == "file":
                stats["files_searched"] += 1

                if not item.name.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                if item.id in seen_file_ids:
                    stats["already_logged"] += 1
                    continue

                seen_file_ids.add(item.id)
                file = client.file(item.id).get(fields=["id", "name"])
                file_url = f"https://app.box.com/file/{file.id}"

                state["pending_records"].append(
                    {
                        "file_name": file.name,
                        "file_id": file.id,
                        "path": parent_path,
                        "file_url": file_url,
                        "direct_download_url": (
                            f"https://api.box.com/2.0/files/{file.id}/content"
                        ),
                        "preview_url": (
                            f"https://app.box.com/file/{file.id}/preview"
                        ),
                    }
                )
                stats["images_found"] += 1

                emit_log(
                    f"{indent}  Image matched: {file.name} | "
                    f"files searched={stats['files_searched']} | "
                    f"images found={stats['images_found']}",
                    log_callback,
                )

                if len(state["pending_records"]) >= state["batch_size"]:
                    _flush_pending(
                        state,
                        log_callback=log_callback,
                    )

            elif item.type == "folder":
                crawl(
                    str(item.id),
                    f"{parent_path}/{item.name}",
                    depth + 1,
                )

        emit_log(
            f"{indent}Exiting folder: {folder_label} | "
            f"files searched={stats['files_searched']} | "
            f"images found={stats['images_found']}",
            log_callback,
        )

    try:
        crawl(str(folder_id))
        _flush_pending(
            state,
            log_callback=log_callback,
            force=True,
        )

    except BoxOAuthException as exc:
        raise RuntimeError(
            "Box authentication failed. Re-authenticate with Box."
        ) from exc
    except BoxAPIException as exc:
        if exc.status == 401:
            raise RuntimeError(
                "Box returned 401 invalid_token. Re-authenticate with Box."
            ) from exc
        raise

    emit_log(
        "Box crawl complete | "
        f"new images logged={stats['images_found']} | "
        f"already logged skipped={stats['already_logged']} | "
        f"total records={len(state['records_by_id'])}",
        log_callback,
    )

    return stats
