#!/bin/bash

set -e

rm -rf build
rm -rf dist
rm -f "Trail Camera Image Processor.spec"


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