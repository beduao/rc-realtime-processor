"""Valida a conexão RTSP com a câmera: conecta, imprime a resolução e salva 1 frame.

Uso:  python scripts/test_camera.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from core.camera import Camera  # noqa: E402
from core.config import load_config, project_path  # noqa: E402


def main() -> int:
    cfg = load_config()
    print("Conectando em:", cfg.camera.rtsp_url)
    cam = Camera(cfg.camera.rtsp_url, cfg.camera.reconnect_delay_seconds).start()
    try:
        frame = cam.read_wait(timeout=15)
    finally:
        # mantém o cam vivo até salvarmos o frame
        pass

    if frame is None:
        print("FALHA: nenhum frame em 15s. Verifique IP/usuário/senha/porta e a rede.")
        cam.stop()
        return 1

    h, w = frame.shape[:2]
    print(f"OK! Frame recebido: {w}x{h}")

    out_dir = project_path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "test_frame.jpg"
    cv2.imwrite(str(out), frame)
    print("Frame salvo em:", out)

    cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
