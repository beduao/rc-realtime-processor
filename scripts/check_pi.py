"""Diagnóstico completo no Raspberry Pi: ambiente, modelos, câmera e desempenho.

Uso:
    python scripts/check_pi.py                # tudo
    python scripts/check_pi.py --no-camera    # só ambiente e modelos
    python scripts/check_pi.py --seconds 20   # tempo do teste de desempenho

Ele responde as perguntas que importam antes de culpar o código:
  - o OpenCV instalado consegue carregar o YuNet 2023mar?
  - a câmera entrega frames, em que resolução e a quantos fps?
  - quantos frames por segundo o Pi consegue REALMENTE reconhecer?
  - o gargalo é o decode do vídeo, a detecção ou o embedding?
"""

import argparse
import os
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.camera import camera_from_config  # noqa: E402
from core.config import live_image_path, load_config_or_exit, project_path  # noqa: E402
from core.database import Database  # noqa: E402

OK = "  [ok] "
BAD = "  [XX] "
WARN = "  [!!] "


def head(title: str):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def _worker_rodando() -> bool:
    """Há um `python .../worker.py` no ar? (ele segura a câmera USB)"""
    try:
        from scripts.monitor import achar_worker
        return achar_worker({}) is not None
    except Exception:                                   # noqa: BLE001
        pass
    try:
        import subprocess
        return subprocess.run(["systemctl", "is-active", "--quiet", "facial-worker"],
                              timeout=5).returncode == 0
    except Exception:                                   # noqa: BLE001
        return False


def _read_first_line(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.readline().strip().replace("\x00", "")
    except OSError:
        return "?"


def check_environment() -> bool:
    head("Ambiente")
    print(f"      modelo:  {_read_first_line('/proc/device-tree/model')}")
    print(f"      python:  {sys.version.split()[0]}  ({sys.executable})")
    print(f"      opencv:  {cv2.__version__}  threads={cv2.getNumThreads()}")
    print(f"      numpy:   {np.__version__}")

    good = True
    major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
    if (major, minor) < (4, 8):
        print(BAD + f"OpenCV {cv2.__version__} < 4.8 — o YuNet 2023mar não carrega. "
                    "Instale: pip install 'opencv-contrib-python-headless>=4.9,<5'")
        good = False
    else:
        print(OK + "versão do OpenCV compatível com o YuNet 2023mar")

    for attr in ("FaceDetectorYN", "FaceRecognizerSF"):
        if hasattr(cv2, attr):
            print(OK + f"cv2.{attr} disponível")
        else:
            print(BAD + f"cv2.{attr} ausente — instale o pacote *contrib*")
            good = False

    build = cv2.getBuildInformation()
    ffmpeg_line = next((ln.strip() for ln in build.splitlines() if "FFMPEG" in ln), "")
    if "YES" in ffmpeg_line.upper():
        print(OK + "OpenCV com FFmpeg (RTSP funciona)")
    else:
        print(WARN + f"suporte a FFmpeg incerto: '{ffmpeg_line or 'não reportado'}'")

    # memória e disco: as duas causas silenciosas de morte no Pi
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mem[k] = int(v.split()[0]) // 1024
    print(f"      RAM:     {mem.get('MemTotal', 0)} MB total / "
          f"{mem.get('MemAvailable', 0)} MB disponível / "
          f"{mem.get('SwapTotal', 0)} MB swap")
    if mem.get("MemAvailable", 0) < 200:
        print(WARN + "menos de 200 MB livres. Desligue o desktop gráfico: "
                     "sudo raspi-config -> System Options -> Boot -> Console")

    usage = shutil.disk_usage(str(project_path(".")))
    free_mb = usage.free // (1024 * 1024)
    print(f"      disco:   {free_mb} MB livres")
    if free_mb < 300:
        print(WARN + "pouco espaço. Rode: python scripts/cleanup_snapshots.py --days 7")

    # throttling térmico / subtensão: fonte fraca é clássico em Pi com câmera IP
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        flags = int(out.split("=")[1], 16) if "=" in out else 0
        if flags == 0:
            print(OK + "sem histórico de subtensão/throttling")
        else:
            print(WARN + f"vcgencmd {out} — houve subtensão ou throttling térmico. "
                         "Fonte fraca derruba o stream RTSP de forma intermitente.")
        temp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        print(f"      temp:    {temp.replace('temp=', '')}")
    except Exception:
        pass

    return good


def check_config_and_models() -> bool:
    head("Configuração e modelos")
    cfg = load_config_or_exit()
    good = True

    url = str(cfg.camera.rtsp_url)
    safe = url
    if "@" in url and "//" in url:
        prefix, rest = url.split("//", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        safe = f"{prefix}//{user}:***@{host}"
    print(f"      rtsp:    {safe}")

    if "SENHA_CODIFICADA" in url or "SUA_SENHA" in url:
        print(BAD + "a URL RTSP ainda é o exemplo — edite o config.yaml")
        good = False
    if "subtype=0" in url:
        print(WARN + "você está no stream principal (subtype=0). No Pi 3B use subtype=1.")
    for ch in "@#":
        creds = url.split("//", 1)[-1].split("@")[0] if "@" in url else ""
        if ch in creds:
            print(WARN + f"a senha parece conter '{ch}' sem codificar "
                         f"(@ = %40, # = %23) — isso quebra a URL RTSP")

    print(f"      live:    {live_image_path(cfg)}")
    print(f"      banco:   {project_path(cfg.storage.db_path)}")

    for label, rel in (("detector", cfg.models.detector),
                       ("reconhec.", cfg.models.recognizer)):
        p = project_path(rel)
        if p.exists():
            print(OK + f"{label}: {p.name} ({p.stat().st_size // 1024} KB)")
        else:
            print(BAD + f"{label}: {p} não existe — rode python models/download_models.py")
            good = False

    if not good:
        return False

    try:
        from core.face_engine import FaceEngine
        t0 = time.time()
        engine = FaceEngine(cfg)
        print(OK + f"engine carregada em {time.time() - t0:.1f}s")
    except Exception as exc:                                   # noqa: BLE001
        print(BAD + f"falha ao carregar a engine: {exc}")
        return False

    # sanidade: detecta em uma imagem sintética (não deve estourar exceção)
    dummy = np.full((240, 320, 3), 127, dtype=np.uint8)
    try:
        engine.detect(dummy)
        print(OK + "detecção executou sem erro")
    except Exception as exc:                                   # noqa: BLE001
        print(BAD + f"detect() falhou: {exc}")
        return False

    db = Database(cfg.storage.db_path)
    people = db.list_people()
    total_emb = sum(p["embeddings"] for p in people)
    print(f"      galeria: {len(people)} pessoas / {total_emb} embeddings")
    if not people:
        print(WARN + "nenhuma pessoa cadastrada: tudo vai sair como 'Desconhecido'")
    print(f"      eventos: {len(db.list_events(limit=500))} (últimos 500 consultados)")
    return True


def check_camera_and_speed(seconds: float) -> bool:
    head(f"Câmera e desempenho ({seconds:.0f}s)")
    cfg = load_config_or_exit()
    cam = camera_from_config(cfg).start()
    try:
        t0 = time.time()
        frame = cam.read_wait(timeout=20)
        if frame is None:
            if cam.is_local_device:
                # Causa nº 1 com webcam USB: dispositivo V4L2 é EXCLUSIVO.
                ocupado = _worker_rodando()
                print(BAD + "não consegui abrir a câmera.")
                if ocupado:
                    print("       O serviço facial-worker está rodando e segura o "
                          "dispositivo.")
                    print("       Webcam USB e câmera CSI só aceitam UM processo por vez:")
                    print("         sudo systemctl stop facial-worker")
                    print("         .venv/bin/python scripts/check_pi.py")
                    print("         sudo systemctl start facial-worker")
                    print("       (para acompanhar SEM parar nada, use scripts/monitor.py)")
                else:
                    print("       Confira o dispositivo e as permissões:")
                    print("         ls /dev/video*        # existe?")
                    print("         v4l2-ctl --list-devices")
                    print("         groups | grep video   # seu usuário está no grupo?")
            else:
                print(BAD + "nenhum frame em 20s. Verifique IP, usuário, senha, "
                            "porta 554 e a rede.")
                print("       Teste rápido:  ffprobe -rtsp_transport tcp '<sua_url>'")
            return False
        h, w = frame.shape[:2]
        print(OK + f"primeiro frame em {time.time() - t0:.1f}s — {w}x{h}")
        if w > 1024:
            print(WARN + f"{w}px de largura é muito para o Pi 3B. Use o substream "
                         "(subtype=1) e configure-o para 640x480.")

        from core.face_engine import FaceEngine
        threads = int(cfg.worker.get("opencv_threads", 0) or 0)
        if threads > 0:
            cv2.setNumThreads(threads)
        engine = FaceEngine(cfg)
        db = Database(cfg.storage.db_path)
        gallery = db.load_gallery()

        det_ms, emb_ms, faces_total = [], [], 0
        seq, frames = 0, 0
        deadline = time.time() + seconds
        while time.time() < deadline:
            f, seq = cam.read_new(seq)
            if f is None:
                time.sleep(0.01)
                continue
            frames += 1
            t = time.time()
            faces = engine.detect(f)
            det_ms.append((time.time() - t) * 1000)
            faces_total += len(faces)
            for face in faces:
                t = time.time()
                vec = engine.embed(f, face)
                engine.match(vec, gallery.matrix, gallery.ids)
                emb_ms.append((time.time() - t) * 1000)

        elapsed = seconds
        fps_cam = cam.frames_received / elapsed
        print(f"      câmera:  {fps_cam:.1f} fps recebidos "
              f"({cam.frames_received} frames, {cam.reconnects} reconexões)")
        if det_ms:
            print(f"      detecção: {statistics.median(det_ms):.1f} ms/frame "
                  f"(mediana de {len(det_ms)} medições)")
        if emb_ms:
            print(f"      embedding: {statistics.median(emb_ms):.1f} ms/rosto "
                  f"({faces_total} rostos vistos)")
        else:
            print("      embedding: nenhum rosto no enquadramento durante o teste")

        per_frame = (statistics.median(det_ms) if det_ms else 0) + \
                    (statistics.median(emb_ms) if emb_ms else 0)
        if per_frame:
            print(f"      capacidade: ~{1000 / per_frame:.1f} reconhecimentos/s "
                  f"({per_frame:.1f} ms por frame com 1 rosto)")
            step = max(1, int(cfg.worker.process_every_n_frames))
            mi = float(cfg.worker.get("min_interval_seconds", 0) or 0)
            alvo = min(fps_cam / step if step else fps_cam, (1 / mi) if mi else 1e9)
            print(f"      config atual pede ~{alvo:.1f}/s "
                  f"(process_every_n_frames={step}, min_interval_seconds={mi})")
            if alvo > 1000 / per_frame * 1.2:
                print(WARN + "o config pede mais do que este Pi entrega: aumente "
                             "worker.min_interval_seconds ou process_every_n_frames")
            else:
                print(OK + "config compatível com a capacidade medida")
        if fps_cam < 3:
            print(WARN + "menos de 3 fps chegando: rede/fonte de energia instável, "
                         "ou o substream está em H.265 (o Pi 3B não decodifica bem). "
                         "Configure o substream como H.264.")
        return True
    finally:
        cam.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico do Pi.")
    ap.add_argument("--no-camera", action="store_true", help="não testa a câmera")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="duração do teste de desempenho (padrão 15s)")
    args = ap.parse_args()

    results = [check_environment(), check_config_and_models()]
    if not args.no_camera and results[1]:
        results.append(check_camera_and_speed(args.seconds))

    head("Resultado")
    if all(results):
        print(OK + "tudo verificado. Se os serviços estiverem ativos, está operando.")
        print("       sudo systemctl status facial-worker facial-api")
        return 0
    print(BAD + "há problemas acima. Corrija de cima para baixo — os primeiros "
                "itens causam os de baixo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
