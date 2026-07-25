# Wild Basin Image Processor — macOS starter

This is the first desktop-app scaffold for the Wild Basin Box/SpeciesNet/PaddleOCR pipeline.

## What is implemented

- PySide6 desktop GUI.
- Box API fields: Client ID, Client Secret, Access Token, Refresh Token.
- Box folder ID or Box folder URL input.
- Output folder chooser.
- User-defined filenames for:
  - Box image-record JSON
  - SpeciesNet JSONL
  - PaddleOCR JSONL
- Selectable pipeline stages.
- Advanced batch/download/OCR crop controls.
- Live stdout/stderr streaming into the GUI.
- Cancel button.
- Open-results-folder button.
- The processing scripts under `engine/` are copied from the supplied project:
  - `batch-box-run-speciesnet.py`
  - `batch-box-paddle-ocr-gpu.py`
  - `box-get-urls.py`
  - `box_downloads.py`

## Important current limitation

This is a development milestone, not yet a signed `.app`/`.dmg`.

The current GUI passes Box credentials to the child scripts through their environment.
The existing scripts may refresh OAuth tokens by writing `.env`; secure long-lived token
storage and refresh-token handoff will be implemented before release.

On macOS, the existing PaddleOCR `--device auto` logic will fall back to CPU when CUDA
is unavailable. Mac-specific acceleration and dependency compatibility still need to
be tested before packaging.

## Run during development

Use the same Python version/environment that successfully installs SpeciesNet and PaddleOCR.

```bash
python -m pip install -r requirements-desktop.txt
python app.py
```

## Recommended development order

1. Launch GUI locally.
2. Test Box crawl with a very small folder.
3. Test SpeciesNet stage with a small limit/sample.
4. Test PaddleOCR on the same sample.
5. Add OAuth browser flow and secure macOS Keychain storage.
6. Add detailed progress parsing/resume UI.
7. Resolve Apple Silicon dependency/model issues.
8. Build `.app`.
9. Sign with Developer ID and notarize.
10. Package as `.dmg`.
