from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth


LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]
TokenCallback = Callable[[str, str], None]


class PipelineCancelled(Exception):
    """Raised when the user cancels a running pipeline."""


def check_cancelled(should_cancel: Optional[CancelCallback]) -> None:
    if should_cancel is not None and should_cancel():
        raise PipelineCancelled("Processing cancelled by user.")


def emit_log(message: str, log_callback: Optional[LogCallback] = None) -> None:
    if log_callback is not None:
        log_callback(message)
    else:
        print(message)


def build_box_client(
    *,
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str = "",
    token_callback: Optional[TokenCallback] = None,
) -> Client:
    """
    Build a Box client from credentials already held by the desktop app.

    Unlike the old helper module, this does not read credentials from .env.
    """
    if not access_token:
        raise ValueError("Box access token is missing.")

    if refresh_token and client_id and client_secret:
        def store_tokens(new_access_token: str, new_refresh_token: str) -> None:
            if token_callback is not None:
                token_callback(new_access_token, new_refresh_token or "")

        auth = OAuth2(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            store_tokens=store_tokens,
        )
    else:
        auth = DeveloperTokenAuth(
            get_new_token_callback=lambda: access_token
        )

    return Client(auth)


def load_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    return data


def load_processed_file_ids(
    results_file: str,
    status: str = "ok",
) -> Set[str]:
    processed: Set[str] = set()

    if not os.path.exists(results_file):
        return processed

    with open(results_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            file_id = str(record.get("file_id", "")).strip()
            if file_id and record.get("status") == status:
                processed.add(file_id)

    return processed


def append_result(results_file: str, result: Dict) -> None:
    output_path = Path(results_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")


def cleanup_temp_files(paths: List[Optional[str]]) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def download_box_file_to_temp(
    client: Client,
    file_id: str,
    file_name: str,
    retries: int = 3,
    backoff: float = 0.5,
) -> str:
    suffix = Path(file_name).suffix or ".jpg"
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name

        try:
            with temp_file:
                client.file(file_id).download_to(temp_file)
            return temp_path
        except Exception as exc:
            last_error = exc
            cleanup_temp_files([temp_path])

            if attempt + 1 < retries:
                time.sleep(backoff * (2 ** attempt))

    if last_error is None:
        raise RuntimeError(f"Could not download Box file {file_id}")

    raise last_error


def download_images_parallel(
    *,
    client: Client,
    records: List[dict],
    max_workers: int = 32,
    should_cancel: Optional[CancelCallback] = None,
    log_callback: Optional[LogCallback] = None,
) -> Tuple[List[Tuple[dict, str]], int]:
    """
    Download records concurrently.

    Returns:
        (successful_downloads, failure_count)
    """
    downloaded: List[Tuple[dict, str]] = []
    failures = 0

    check_cancelled(should_cancel)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}

        for record in records:
            check_cancelled(should_cancel)

            file_id = str(record.get("file_id", "")).strip()
            file_name = record.get("file_name") or f"{file_id}.jpg"

            future = executor.submit(
                download_box_file_to_temp,
                client,
                file_id,
                file_name,
            )
            future_map[future] = (record, file_id)

        for future in as_completed(future_map):
            check_cancelled(should_cancel)

            record, file_id = future_map[future]
            try:
                temp_path = future.result()
                downloaded.append((record, temp_path))
            except Exception as exc:
                failures += 1
                emit_log(
                    f"Download failed: file_id={file_id} error={exc}",
                    log_callback,
                )

    return downloaded, failures
