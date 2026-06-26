"""Worker de reconhecimento (roda no Pi — ou no Mac na Fase 1).

Loop contínuo:
  câmera -> detecta rostos -> embedding -> compara com a galeria ->
  grava evento + snapshot (respeitando o cooldown) e atualiza data/live.jpg.

Tudo é feito em CPU. As mitigações para Pi 3B estão no config.yaml:
  worker.process_every_n_frames, worker.event_cooldown_seconds,
  models.detect_width.
"""

import time

import cv2

from core.camera import Camera
from core.config import load_config, project_path
from core.database import Database
from core.draw import draw_face
from core.face_engine import FaceEngine
from core.storage import SnapshotStore

GALLERY_RELOAD_SECONDS = 10.0   # recarrega cadastros novos sem reiniciar
LIVE_WRITE_SECONDS = 0.5        # frequência de atualização do preview ao vivo


def main():
    cfg = load_config()
    engine = FaceEngine(cfg)
    db = Database(cfg.storage.db_path)
    store = SnapshotStore(cfg.storage.snapshots_dir)
    live_path = project_path("data/live.jpg")
    live_path.parent.mkdir(parents=True, exist_ok=True)

    cam = Camera(cfg.camera.rtsp_url, cfg.camera.reconnect_delay_seconds).start()

    cooldown = float(cfg.worker.event_cooldown_seconds)
    step = max(1, int(cfg.worker.process_every_n_frames))
    annotate = bool(cfg.worker.draw_annotations)
    threshold = engine.cosine_threshold

    gallery = db.load_gallery()
    last_reload = time.time()
    last_live = 0.0
    last_seen: dict[str, float] = {}
    frame_idx = 0

    print(f"[worker] iniciado. limiar={threshold} cooldown={cooldown}s "
          f"1 a cada {step} frames. Galeria: {len(gallery.ids)} embeddings.")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.1)
            continue

        frame_idx += 1
        if frame_idx % step != 0:
            time.sleep(0.005)
            continue

        now = time.time()
        if now - last_reload > GALLERY_RELOAD_SECONDS:
            gallery = db.load_gallery()
            last_reload = now

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

            snap_frame = live if live is not None else frame.copy()
            snapshot = store.save(snap_frame, name)
            db.add_event(pid, name, score, snapshot, known)
            print(f"[evento] {name} (score={score:.3f}) -> {snapshot}")

        # atualiza preview ao vivo (consumido pela API em /live.jpg)
        if live is not None and now - last_live > LIVE_WRITE_SECONDS:
            cv2.imwrite(str(live_path), live)
            last_live = now

        time.sleep(0.005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] encerrado.")
