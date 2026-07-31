# Wild Basin Trail Camera Image Processor

A cross-platform desktop application for downloading trail camera images from Box, running SpeciesNet and PaddleOCR, and exporting results for import into the Wild Basin Django website.

The application was built using **PySide6** and is intended to allow researchers to process images without installing Python or running scripts manually.

---

# Features

## Box Authentication

- Browser-based OAuth login
- Automatic access token refresh
- Displays authenticated user
- Copy access/refresh tokens
- No manual OAuth setup required

---

## Box Processing

- Crawl Box folders recursively
- Resume previous crawls
- Skip already processed files
- Download images in batches
- Configurable batch size

Outputs:

- Image URL JSON

---

## SpeciesNet

Runs SpeciesNet classification directly inside the application.

Features

- Batch processing
- Resume support
- Progress reporting
- Automatic cleanup of temporary downloads

Outputs

- SpeciesNet JSONL

---

## PaddleOCR

Runs PaddleOCR on downloaded images.

Features

- CPU support
- Configurable bottom crop
- Resume support
- Batch processing

Outputs

- OCR JSONL

---

## GUI

- Native PySide6 interface
- Live console output
- Cancel processing
- Open output folder
- Adjustable batch sizes
- Scrollable interface
- Pipeline stage selection

---

# Project Structure

```
app.py

wildbasin_app/

    auth/
        box_oauth.py

    engines/
        box_urls.py
        speciesnet_engine.py
        paddle_ocr_engine.py
        common.py

    gui/
        main_window.py

    pipeline/
        runner.py
```

The old scripts are still kept under

```
engine/
```

as reference implementations.

---

# Development Environment

Python

```
3.9.x
```

Recommended platform

- macOS Apple Silicon

Main packages

```
PySide6
boxsdk
speciesnet
torch
torchvision
onnx
onnx2torch
paddlepaddle
paddleocr
paddlex
```

---

# Running from Source

Create a virtual environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

Install dependencies.

Run

```bash
python app.py
```

---

# Building the macOS Application

The application is currently packaged using **PyInstaller**.

Example build command

```bash
python -m PyInstaller \
  --clean \
  --noconfirm \
  --windowed \
  --onedir \
  --name "Trail Camera Image Processor" \
  --collect-all boxsdk \
  --collect-all paddle \
  --collect-all paddleocr \
  --collect-all paddlex \
  --collect-all speciesnet \
  --collect-all onnx2torch \
  --collect-all onnx \
  --collect-all torch \
  --collect-all torchvision \
  --collect-all imagesize \
  --collect-all pyclipper \
  --collect-all pypdfium2 \
  --collect-all bidi \
  --collect-all shapely \
  --collect-all cv2 \
  --copy-metadata cloudpathlib \
  --copy-metadata imagesize \
  --copy-metadata opencv-contrib-python \
  --copy-metadata pyclipper \
  --copy-metadata pypdfium2 \
  --copy-metadata python-bidi \
  --copy-metadata shapely \
  --recursive-copy-metadata paddlex \
  --recursive-copy-metadata paddleocr \
  app.py
```

The generated application is placed in

```
dist/
```

---

# Building on Another Computer

The project can be built on another machine **provided the environment matches**.

Requirements

- macOS (Apple Silicon recommended)
- Python 3.9.x
- Xcode Command Line Tools
- Homebrew
- OpenBLAS

Install OpenBLAS

```bash
brew install openblas
```

Clone the repository

```bash
git clone <repository>
cd trailcamera-image-app
```

Create a virtual environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

Install all project dependencies.

Run

```bash
python app.py
```

Once the application works correctly from source, run the PyInstaller build command above.

---

# Current Status

✅ Box OAuth

✅ Box crawler

✅ SpeciesNet

✅ PaddleOCR

✅ Resume processing

✅ Native GUI

✅ PyInstaller packaging

⬜ Code signing

⬜ Apple notarization

⬜ DMG creation

⬜ Windows/Linux builds (future)

---

# Future Improvements

- Automatic application updates
- Keychain credential storage
- Progress bars
- Background downloads
- Parallel OCR processing
- Better error dialogs
- Signed macOS installer