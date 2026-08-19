"""Valida a conexão com a câmera: conecta, imprime a resolução e salva 1 frame.

Funciona para câmera IP (RTSP), webcam USB (`camera.rtsp_url: 0`) e arquivo de
vídeo — o backend é escolhido automaticamente.

Uso:  python scripts/test_camera.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from core.camera import camera_from_config  # noqa: E402
from core.config import load_config_or_exit, project_path  # noqa: E402


def main() -> int:
    cfg = load_config_or_exit()
    print("Conectando em:", cfg.camera.rtsp_url)
    cam = camera_from_config(cfg).start()
    print("Backend:", "V4L2 (dispositivo local)" if cam.is_local_device else "FFmpeg (RTSP/arquivo)")
    try:
        frame = cam.read_wait(timeout=15)
    finally:
        # mantém o cam vivo até salvarmos o frame
        pass

    if frame is None:
        if cam.is_local_device:
            print("FALHA: não consegui abrir o dispositivo.")
            print("  Webcam USB e câmera CSI aceitam UM processo por vez. Se o")
            print("  facial-worker estiver rodando, ele está com o dispositivo:")
            print("    sudo systemctl stop facial-worker")
            print("  Se não estiver, confira o dispositivo e as permissões:")
            print("    ls /dev/video*  |  v4l2-ctl --list-devices  |  groups | grep video")
        else:
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
