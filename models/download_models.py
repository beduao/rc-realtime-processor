"""Baixa os modelos ONNX (YuNet + SFace) do OpenCV Zoo para esta pasta.

Uso:  python models/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def _progress(block, block_size, total):
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        sys.stdout.write(f"\r   {pct}%")
        sys.stdout.flush()


def download(name: str, url: str):
    dest = DEST / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] {name} já existe ({dest.stat().st_size // 1024} KB)")
        return
    print(f"[baixando] {name}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\r[ok] {name} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    for name, url in MODELS.items():
        download(name, url)
    print("Modelos prontos em:", DEST)
