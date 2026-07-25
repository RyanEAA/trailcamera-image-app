from dataclasses import dataclass
from pathlib import Path
import re

BOX_FOLDER_RE = re.compile(r"(?:app\.box\.com/(?:folder|s)/)?(\d+)")

def normalize_box_folder_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value

    match = BOX_FOLDER_RE.search(value)
    if match:
        return match.group(1)

    raise ValueError(
        "Enter a Box folder ID or a Box folder URL containing a numeric folder ID."
    )

@dataclass(frozen=True)
class PipelineConfig:
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str
    folder_id: str
    output_dir: Path
    url_filename: str
    speciesnet_filename: str
    ocr_filename: str
    run_url_scan: bool = True
    run_speciesnet: bool = True
    run_ocr: bool = True
    speciesnet_batch_size: int = 128
    ocr_batch_size: int = 128
    download_workers: int = 32
    ocr_crop_percent: float = 0.065

    @property
    def url_file(self) -> Path:
        return self.output_dir / self.url_filename

    @property
    def speciesnet_file(self) -> Path:
        return self.output_dir / self.speciesnet_filename

    @property
    def ocr_file(self) -> Path:
        return self.output_dir / self.ocr_filename
