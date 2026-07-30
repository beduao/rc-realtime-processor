"""Worker de reconhecimento (roda no Raspberry Pi — ou no Mac na Fase 1).

Loop contínuo:
  câmera -> detecta rostos -> embedding -> compara com a galeria ->
  grava evento + snapshot (respeitando o cooldown) e atualiza o preview ao vivo.

Tudo é feito em CPU. Ajustes para Pi 3B ficam no config.yaml:
  worker.process_every_n_frames   (agora conta frames REAIS recebidos)
  worker.min_interval_seconds     (teto de processamentos por segundo)
  worker.event_cooldown_seconds
  worker.opencv_threads
  models.detect_width
  storage.live_path               (aponte para /dev/shm no Pi: poupa o cartão SD)
"""

import os
import time

import cv2

from core.camera import camera_from_config
from core.config import live_image_path, load_config
from core.database import Database
from core.draw import draw_face
from core.face_engine import FaceEngine
from core.storage import SnapshotStore

GALLERY_RELOAD_SECONDS = 10.0   # recarrega cadastros novos sem reiniciar
LIVE_WRITE_SECONDS = 0.5        # frequência de atualização do preview ao vivo
LIVE_JPEG_QUALITY = 70          # menor = menos CPU e menos escrita em disco
STATS_EVERY_SECONDS = 300.0     # log periódico de saúde (aparece no journalctl)


def _write_live(path, image):
    """Escreve o preview de forma atômica (arquivo temporário + rename).

    Sem isso a API pode servir um JPEG cortado, porque ela lê o arquivo no
    mesmo instante em que o worker está escrevendo.

    Codificamos em memória com `imencode` em vez de usar `imwrite` num arquivo
    ".tmp": o OpenCV escolhe o formato pela EXTENSÃO, então um nome temporário
    terminado em .tmp faz o imwrite falhar.
    """
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
    if not ok:
        return
    tmp = str(path) + ".part"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(tmp, str(path))


def main():
    cfg = load_config()

    threads = int(cfg.worker.get("opencv_threads", 0) or 0)
    if threads > 0:
        # No Pi 3B (4 núcleos) deixar 1 núcleo livre para o decode do RTSP
        # normalmente resulta em latência menor do que usar os 4 na DNN.
        cv2.setNumThreads(threads)

    engine = FaceEngine(cfg)
    db = Database(cfg.storage.db_path)
    store = SnapshotStore(cfg.storage.snapshots_dir)
    live_path = live_image_path(cfg)
    live_path.parent.mkdir(parents=True, exist_ok=True)

    cam = camera_from_config(cfg).start()

    cooldown = float(cfg.worker.event_cooldown_seconds)
    step = max(1, int(cfg.worker.process_every_n_frames))
    min_interval = float(cfg.worker.get("min_interval_seconds", 0.0) or 0.0)
    annotate = bool(cfg.worker.draw_annotations)
    threshold = engine.cosine_threshold

    gallery = db.load_gallery()
    last_reload = time.time()
    last_live = 0.0
    last_proc = 0.0
    last_stats = time.time()
    last_seen: dict[str, float] = {}
    seq = 0
    frames_seen = 0
    processed = 0
    proc_time = 0.0

    print(f"[worker] iniciado. opencv={cv2.__version__} threads={cv2.getNumThreads()} "
          f"limiar={threshold} cooldown={cooldown}s 1 a cada {step} frames "
          f"(intervalo mínimo {min_interval}s). Galeria: {len(gallery.ids)} embeddings.",
          flush=True)

    while True:
        frame, seq = cam.read_new(seq)
        if frame is None:
            time.sleep(0.02)          # nada novo: não queime CPU
            continue

        frames_seen += 1
        now = time.time()

        # descarte por contagem de frames reais e por teto de taxa
        if frames_seen % step != 0 or (min_interval and now - last_proc < min_interval):
            continue
        last_proc = now

        if now - last_reload > GALLERY_RELOAD_SECONDS:
            gallery = db.load_gallery()
            last_reload = now

        t0 = time.time()
        try:
            faces = engine.detect(frame)
            live = frame.copy() if annotate else None

            for face in faces:
                vec = engine.embed(frame, face)
                pid, score = engine.match(vec, gallery.matrix, gallery.ids)

                if pid is not None and score >= threshold:
                    name = gallery.names.get(pid, "?")
                    known = True
                    key = f"known:{pid}"
                else:
                    name, pid, known, key = "Desconhecido", None, False, "unknown"

                if live is not None:
                    draw_face(live, face, name, score, known)

                # cooldown: não duplicar a mesma identidade em janela curta
                if now - last_seen.get(key, 0.0) < cooldown:
                    continue
                last_seen[key] = now

                snap_frame = live if live is not None else frame
                snapshot = store.save(snap_frame, name)
                db.add_event(pid, name, score, snapshot, known)
                print(f"[evento] {name} (score={score:.3f}) -> {snapshot}", flush=True)

            # atualiza preview ao vivo (consumido pela API em /live.jpg)
            if live is not None and now - last_live > LIVE_WRITE_SECONDS:
                _write_live(live_path, live)
                last_live = now
        except Exception as exc:                       # noqa: BLE001
            # um frame corrompido ou erro transitório não deve matar o serviço
            print(f"[erro] frame ignorado: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.2)
            continue

        processed += 1
        proc_time += time.time() - t0

        if now - last_stats > STATS_EVERY_SECONDS:
            fps_in = frames_seen / (now - last_stats)
            avg = (proc_time / processed * 1000) if processed else 0.0
            print(f"[stats] {fps_in:.1f} fps da câmera | {processed} processados "
                  f"| {avg:.0f} ms/frame | {cam.stats()}", flush=True)
            last_stats, frames_seen, processed, proc_time = now, 0, 0, 0.0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] encerrado.")
