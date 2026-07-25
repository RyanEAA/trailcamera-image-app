import os
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import find_dotenv, load_dotenv, set_key
from boxsdk import Client
from boxsdk import OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ENV_PATH = find_dotenv(usecwd=True) or ".env"


def store_tokens(access_token: str, refresh_token: str):
    """Store access and refresh tokens to .env file."""
    set_key(ENV_PATH, "ACCESS_TOKEN", access_token)
    if refresh_token:
        set_key(ENV_PATH, "REFRESH_TOKEN", refresh_token)


def build_client() -> Client:
    """Build Box SDK client from .env credentials."""
    if not ACCESS_TOKEN:
        raise ValueError("ACCESS_TOKEN is missing. Set it in your .env file.")

    if REFRESH_TOKEN and CLIENT_ID and CLIENT_SECRET:
        auth = OAuth2(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            access_token=ACCESS_TOKEN,
            refresh_token=REFRESH_TOKEN,
            store_tokens=store_tokens,
        )
    else:
        auth = DeveloperTokenAuth(get_new_token_callback=lambda: ACCESS_TOKEN)

    return Client(auth)


def load_json_list(path: str) -> List[dict]:
    """Load a JSON list from a file."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_processed_file_ids(results_file: str, status: str = "ok") -> Set[str]:
    """Load set of file_ids that have been processed (matching status)."""
    processed = set()

    if not os.path.exists(results_file):
        return processed

    with open(results_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("status") == status:
                    processed.add(str(record.get("file_id")))
            except json.JSONDecodeError:
                continue

    return processed


def append_result(results_file: str, result: Dict):
    """Append a result record as JSONL."""
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def cleanup_temp_files(paths: List[str]):
    """Remove temporary files created during processing."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def download_box_file_to_temp(client: Client, file_id: str, file_name: str, retries: int = 3, backoff: float = 0.5) -> str:
    """Download a Box file to a temporary file with simple retry/backoff.

    Returns the temporary file path on success. Raises the last exception on failure.
    """
    suffix = Path(file_name).suffix or ".jpg"

    attempt = 0
    last_exc = None

    while attempt < retries:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name

        try:
            with temp_file:
                client.file(file_id).download_to(temp_file)
            return temp_path
        except Exception as e:
            last_exc = e
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

            attempt += 1
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))

    # all attempts failed
    raise last_exc


def download_images_parallel(
    client: Client,
    records: List[dict],
    max_workers: int = 32,
    file_id_key: str = "file_id",
    file_name_key: str = "file_name",
) -> List[tuple]:
    """Download a list of records in parallel.

    Returns a list of (record, temp_path) for successful downloads. Failed downloads are logged to stdout.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}

        for record in records:
            file_id = str(record.get(file_id_key, "")).strip()
            file_name = record.get(file_name_key) or f"{file_id}.jpg"

            future = executor.submit(download_box_file_to_temp, client, file_id, file_name)
            future_map[future] = (record, file_id)

        for future in as_completed(future_map):
            record, fid = future_map[future]
            try:
                temp_path = future.result()
                results.append((record, temp_path))
            except Exception as e:
                print(f"Download failed: {fid} {e}")

    return results


def log(message: str, quiet: bool = False):
    if not quiet:
        print(message)
