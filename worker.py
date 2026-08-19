"""Worker de captura/reconhecimento (roda no Raspberry Pi).

Dois modos, escolhidos em `worker.mode` no config.yaml:

  mode: "realtime"  (padrão, comportamento original)
      câmera -> detecta -> reconhece NA HORA -> grava evento + snapshot.
      Bom para fluxo esparso. Custa ~285 ms por rosto, então com várias pessoas
      no frame ele não acompanha o vídeo.

  mode: "captura"   (para grupos / chamada de presença)
      câmera -> detecta -> RASTREIA -> guarda os melhores recortes de cada
      pessoa. NÃO reconhece. A detecção custa ~57 ms por frame independente de
      quantos rostos haja, então a captura acompanha o vídeo mesmo com o
      corredor cheio.
      O reconhecimento roda depois: `python scripts/recognize_batch.py`
      (ou pelo timer facial-batch.timer). Como cada pessoa tem vários
      recortes, a decisão é por votação — bem mais robusta que um frame só.

Ajustes para Pi 3B ficam no config.yaml (veja o preset em config.pi.example.yaml).
"""

import argparse
import json
import os
import signal
import threading
import time

import cv2

from core.camera import camera_from_config
from core.config import (frame_image_path, live_image_path, load_config_or_exit,
                         project_path, reload_config)
from core.database import Database
from core.draw import draw_face
from core.face_engine import FaceEngine
from core.storage import SnapshotStore

GALLERY_RELOAD_SECONDS = 10.0   # recarrega cadastros novos sem reiniciar
LIVE_WRITE_SECONDS = 0.5        # frequência de atualização do preview ao vivo
LIVE_JPEG_QUALITY = 70          # menor = menos CPU e menos escrita em disco
FRAME_WRITE_SECONDS = 0.5       # frame limpo publicado para o cadastro da API
FRAME_JPEG_QUALITY = 95         # alto: dele saem os embeddings do cadastro
STATS_EVERY_SECONDS = 300.0     # log periódico de saúde (aparece no journalctl)

# Encerramento limpo. Sem isto, o SIGTERM do `systemctl stop/restart` mata o
# processo na hora e o `finally` do modo captura nunca roda — as trilhas de quem
# estava em cena naquele momento seriam PERDIDAS a cada reinício do serviço.
PARAR = threading.Event()


def _pedir_parada(signum, _frame):
    print(f"[worker] sinal {signum} recebido, encerrando com calma...", flush=True)
    PARAR.set()


def instalar_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _pedir_parada)
        except ValueError:
            pass    # fora da thread principal (ex.: rodando em teste)


MODOS = ("realtime", "captura")
MODE_CHECK_SECONDS = 10.0   # padrão; `worker.mode_check_seconds` sobrescreve


def modo_do_config(padrao: str = "realtime") -> str:
    """Lê `worker.mode` do disco, sem cache. Valor inválido não derruba o worker."""
    try:
        modo = str((reload_config().get("worker") or {}).get("mode", padrao))
    except Exception as exc:                            # noqa: BLE001
        # config.yaml sendo editado neste exato instante, por exemplo
        print(f"[worker] não consegui reler o config ({exc}); mantendo {padrao}",
              flush=True)
        return padrao
    modo = modo.strip().lower()
    if modo not in MODOS:
        print(f"[worker] worker.mode inválido no config ({modo!r}); mantendo {padrao}",
              flush=True)
        return padrao
    return modo


def publicar_status(live_path, modo: str, extra: dict = None):
    """Publica o estado do worker ao lado do preview (tmpfs, não gasta o SD).

    É assim que a API sabe o modo REAL em execução — que pode diferir do
    config.yaml quando o modo foi fixado por --mode na linha de comando.
    """
    dados = {"mode": modo, "updated_at": time.time(), "pid": os.getpid()}
    dados.update(extra or {})
    caminho = live_path.with_name("facial-status.json")
    tmp = str(caminho) + ".part"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dados, fh)
        os.replace(tmp, str(caminho))
    except OSError:
        pass    # status é informativo; falhar aqui não pode parar o worker


def _escrever_jpeg(path, image, qualidade):
    """Escreve um JPEG de forma atômica (arquivo temporário + rename).

    Sem isso a API pode servir uma imagem cortada, porque ela lê o arquivo no
    mesmo instante em que o worker está escrevendo.

    Codificamos em memória com `imencode` em vez de usar `imwrite` num arquivo
    ".tmp": o OpenCV escolhe o formato pela EXTENSÃO, então um nome temporário
    terminado em .tmp faz o imwrite falhar.
    """
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), qualidade])
    if not ok:
        return
    tmp = str(path) + ".part"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(tmp, str(path))


def _write_live(path, image):
    """Preview anotado, consumido pela API em /live.jpg."""
    _escrever_jpeg(path, image, LIVE_JPEG_QUALITY)


def _write_frame(path, image):
    """Frame LIMPO para o cadastro da API.

    Publicado sempre, nos dois modos e independente de `draw_annotations`:
    é o que permite cadastrar pessoas com o worker rodando, já que webcam USB
    não aceita dois processos abrindo o dispositivo.
    """
    _escrever_jpeg(path, image, FRAME_JPEG_QUALITY)


def intervalo_checagem(cfg) -> float:
    return float(cfg.worker.get("mode_check_seconds", MODE_CHECK_SECONDS)
                 or MODE_CHECK_SECONDS)


def _setup(cfg):
    """Parte comum aos dois modos."""
    threads = int(cfg.worker.get("opencv_threads", 0) or 0)
    if threads > 0:
        # No Pi 3B (4 núcleos) deixar 1 núcleo livre para o decode do RTSP
        # normalmente resulta em latência menor do que usar os 4 na DNN.
        cv2.setNumThreads(threads)

    engine = FaceEngine(cfg)
    db = Database(cfg.storage.db_path)
    live_path = live_image_path(cfg)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    cam = camera_from_config(cfg).start()
    return engine, db, live_path, cam


# --------------------------------------------------------------------------- #
# Modo 1 — reconhecimento em tempo real
# --------------------------------------------------------------------------- #
def loop_realtime(cfg, engine, db, live_path, cam, modo_fixo=False):
    store = SnapshotStore(cfg.storage.snapshots_dir)
    cooldown = float(cfg.worker.event_cooldown_seconds)
    step = max(1, int(cfg.worker.process_every_n_frames))
    min_interval = float(cfg.worker.get("min_interval_seconds", 0.0) or 0.0)
    annotate = bool(cfg.worker.draw_annotations)
    threshold = engine.cosine_threshold

    frame_path = frame_image_path(cfg)
    gallery = db.load_gallery()
    last_reload = last_stats = last_mode = time.time()
    last_live = last_proc = last_frame = 0.0
    last_seen: dict[str, float] = {}
    seq = frames_seen = processed = 0
    proc_time = 0.0
    publicar_status(live_path, "realtime", {"fixo": modo_fixo})

    print(f"[worker] modo=realtime opencv={cv2.__version__} "
          f"threads={cv2.getNumThreads()} limiar={threshold} cooldown={cooldown}s "
          f"1 a cada {step} frames (intervalo mínimo {min_interval}s). "
          f"Galeria: {len(gallery.ids)} embeddings.", flush=True)

    while not PARAR.is_set():
        frame, seq = cam.read_new(seq)
        if frame is None:
            time.sleep(0.02)
            continue

        frames_seen += 1
        now = time.time()

        # Publica o frame limpo antes de qualquer descarte, para o cadastro da
        # API ter preview fluido mesmo quando o processamento está represado.
        if now - last_frame > FRAME_WRITE_SECONDS:
            _write_frame(frame_path, frame)
            last_frame = now

        if not modo_fixo and now - last_mode > intervalo_checagem(cfg):
            last_mode = now
            novo = modo_do_config("realtime")
            if novo != "realtime":
                print(f"[worker] modo alterado no config: realtime -> {novo}",
                      flush=True)
                return novo

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
                    name, known, key = gallery.names.get(pid, "?"), True, f"known:{pid}"
                else:
                    name, pid, known, key = "Desconhecido", None, False, "unknown"

                if live is not None:
                    draw_face(live, face, name, score, known)

                if now - last_seen.get(key, 0.0) < cooldown:
                    continue
                last_seen[key] = now

                snapshot = store.save(live if live is not None else frame, name)
                db.add_event(pid, name, score, snapshot, known)
                print(f"[evento] {name} (score={score:.3f}) -> {snapshot}", flush=True)

            if live is not None and now - last_live > LIVE_WRITE_SECONDS:
                _write_live(live_path, live)
                last_live = now
        except Exception as exc:                       # noqa: BLE001
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
            publicar_status(live_path, "realtime",
                            {"fixo": modo_fixo, "fps": round(fps_in, 1)})
            last_stats, frames_seen, processed, proc_time = now, 0, 0, 0.0

    return None


# --------------------------------------------------------------------------- #
# Modo 2 — captura com rastreamento (reconhecimento fica para o lote)
# --------------------------------------------------------------------------- #
def loop_captura(cfg, engine, db, live_path, cam, modo_fixo=False):
    from core.tracker import FaceTracker

    tcfg = cfg.get("tracking") or {}
    tracker = FaceTracker(
        iou_threshold=float(tcfg.get("iou_threshold", 0.3)),
        max_missing_frames=int(tcfg.get("max_missing_frames", 8)),
        crops_per_track=int(tcfg.get("crops_per_track", 3)),
        min_track_frames=int(tcfg.get("min_track_frames", 2)),
    )
    crops_base = project_path(tcfg.get("crops_dir", "data/tracks"))
    crops_base.mkdir(parents=True, exist_ok=True)
    annotate = bool(cfg.worker.draw_annotations)

    frame_path = frame_image_path(cfg)
    last_stats = last_mode = time.time()
    last_live = last_frame = 0.0
    seq = frames_seen = frame_idx = trilhas = 0
    det_time = 0.0
    trocar_para = None
    publicar_status(live_path, "captura", {"fixo": modo_fixo})

    print(f"[worker] modo=captura opencv={cv2.__version__} "
          f"threads={cv2.getNumThreads()} "
          f"recortes/trilha={tracker.crops_per_track} "
          f"iou={tracker.iou_threshold} "
          f"encerra após {tracker.max_missing_frames} frames sem ver. "
          f"O reconhecimento roda em scripts/recognize_batch.py.", flush=True)

    def gravar(trs):
        nonlocal trilhas
        for tr in trs:
            dia = time.strftime("%Y%m%d", time.localtime(tr.started_at))
            pasta = crops_base / dia
            pasta.mkdir(parents=True, exist_ok=True)
            marca = time.strftime("%H%M%S", time.localtime(tr.started_at))
            registros = []
            for i, (qualidade, recorte, face_local) in enumerate(tr.crops):
                nome = f"{marca}_t{tr.id}_{i}.jpg"
                caminho = pasta / nome
                if not cv2.imwrite(str(caminho), recorte,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                    continue
                registros.append({
                    "path": f"{dia}/{nome}",
                    "quality": qualidade,
                    "face": json.dumps([round(float(v), 3) for v in face_local]),
                })
            if not registros:
                continue
            db.add_track(tr.started_at, tr.ended_at, tr.frames, registros)
            trilhas += 1
            print(f"[trilha {tr.id}] {tr.frames} frames em {tr.duracao:.1f}s, "
                  f"{len(registros)} recorte(s) guardado(s)", flush=True)

    try:
        while not PARAR.is_set():
            frame, seq = cam.read_new(seq)
            if frame is None:
                time.sleep(0.02)
                continue

            frames_seen += 1
            frame_idx += 1
            now = time.time()

            if now - last_frame > FRAME_WRITE_SECONDS:
                _write_frame(frame_path, frame)
                last_frame = now

            if not modo_fixo and now - last_mode > intervalo_checagem(cfg):
                last_mode = now
                novo = modo_do_config("captura")
                if novo != "captura":
                    print(f"[worker] modo alterado no config: captura -> {novo}",
                          flush=True)
                    trocar_para = novo
                    break      # o finally abaixo salva quem está em cena

            t0 = time.time()
            try:
                faces = engine.detect(frame)
                det_time += time.time() - t0
                gravar(tracker.update(faces, frame, frame_idx, now))

                if annotate and now - last_live > LIVE_WRITE_SECONDS:
                    live = frame.copy()
                    for face in faces:
                        draw_face(live, face, "capturando", score=float(face[14]),
                                  known=True)
                    _write_live(live_path, live)
                    last_live = now
            except Exception as exc:                   # noqa: BLE001
                print(f"[erro] frame ignorado: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.2)
                continue

            if now - last_stats > STATS_EVERY_SECONDS:
                fps_in = frames_seen / (now - last_stats)
                avg = (det_time / frames_seen * 1000) if frames_seen else 0.0
                pendentes = db.count_tracks_by_status().get("pendente", 0)
                print(f"[stats] {fps_in:.1f} fps | {avg:.0f} ms/detecção | "
                      f"{trilhas} trilhas gravadas | {len(tracker.ativos)} em cena "
                      f"| {tracker.descartadas} descartadas | "
                      f"{pendentes} pendentes de reconhecimento | {cam.stats()}",
                      flush=True)
                publicar_status(live_path, "captura",
                                {"fixo": modo_fixo, "fps": round(fps_in, 1),
                                 "pendentes": pendentes})
                last_stats, frames_seen, trilhas, det_time = now, 0, 0, 0.0
    finally:
        # Não perde quem estava em cena — vale tanto para parada do serviço
        # quanto para troca de modo em execução.
        em_cena = tracker.flush()
        if em_cena:
            print(f"[worker] salvando {len(em_cena)} trilha(s) em cena", flush=True)
        gravar(em_cena)

    return trocar_para


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Worker de captura/reconhecimento facial.",
        epilog="Sem --mode, o modo vem do config.yaml e pode ser trocado em "
               "execução (o worker relê a cada 10s, sem precisar reiniciar).")
    ap.add_argument("--mode", choices=MODOS,
                    help="fixa o modo, ignorando o config.yaml e desligando a "
                         "troca automática. Útil para teste pontual.")
    args = ap.parse_args(argv)

    instalar_handlers()
    cfg = load_config_or_exit()

    modo_fixo = args.mode is not None
    if modo_fixo:
        modo = args.mode
    else:
        modo = str(cfg.worker.get("mode", "realtime")).strip().lower()
        if modo not in MODOS:
            raise SystemExit(
                f"worker.mode inválido: {modo!r} (use {' ou '.join(MODOS)})")

    # Modelo (37 MB) e câmera são carregados UMA vez e reaproveitados entre as
    # trocas de modo — por isso alternar não custa reinício.
    engine, db, live_path, cam = _setup(cfg)
    origem = "--mode" if modo_fixo else "config.yaml"
    print(f"[worker] modo inicial: {modo} (de {origem})", flush=True)

    try:
        while not PARAR.is_set():
            laco = loop_captura if modo == "captura" else loop_realtime
            proximo = laco(cfg, engine, db, live_path, cam, modo_fixo=modo_fixo)
            if proximo is None or proximo == modo:
                break
            modo = proximo
    finally:
        cam.stop()
    print("[worker] encerrado.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] encerrado.")
