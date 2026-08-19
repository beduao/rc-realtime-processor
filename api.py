"""API HTTP (FastAPI) — roda junto do worker (no Pi, ou no Mac na Fase 1).

Cadastro INTERATIVO por sessão (preview + captura individual das amostras):

    POST /enroll/start    {name}                 -> abre a câmera e cria a sessão
    GET  /enroll/preview                          -> frame ao vivo (com caixa do rosto)
    POST /enroll/capture  {session_id}            -> captura 1 amostra (1 embedding)
    POST /enroll/sample/delete {session_id,index} -> remove uma amostra
    GET  /enroll/status   ?session_id=            -> amostras já capturadas
    POST /enroll/finish   {session_id}            -> grava a pessoa + embeddings
    POST /enroll/cancel   {session_id}            -> descarta a sessão

Outras rotas: /people (GET/DELETE), /events, /snapshots, /live.jpg, /health.

Observações:
- As sessões ficam em memória -> rode o uvicorn com 1 worker (padrão).
- O worker mantém o stream contínuo; a API abre uma 2ª conexão só enquanto
  existe sessão de cadastro ativa (e fecha quando não há nenhuma).

Subir com:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from core.camera import Camera, camera_from_config
from core.config import (frame_image_path, live_image_path, load_config,
                         project_path)
from core.database import Database
from core.draw import crop_face, draw_face
from core.face_engine import FaceEngine
from core.storage import SnapshotStore

cfg = load_config()
engine = FaceEngine(cfg)
db = Database(cfg.storage.db_path)
store = SnapshotStore(cfg.storage.snapshots_dir)

SNAP_BASE = project_path(cfg.storage.snapshots_dir).resolve()
LIVE_PATH = live_image_path(cfg)
STATUS_PATH = LIVE_PATH.with_name("facial-status.json")
FRAME_PATH = frame_image_path(cfg)
FRAME_MAX_AGE = 5.0        # acima disso o frame do worker é velho demais
ENROLL_TARGET = int(cfg.enroll.frames_to_capture)  # amostras sugeridas por pessoa

app = FastAPI(title="Reconhecimento Facial — API")

# ---- estado das sessões de cadastro --------------------------------------- #
_lock = threading.Lock()
_sessions: dict[str, dict] = {}   # session_id -> {name, samples:[{index,embedding,snapshot}]}
_enroll_cam: Camera | None = None


def _frame_do_worker():
    """Último frame limpo publicado pelo worker, ou None se não houver/estiver velho."""
    try:
        if time.time() - FRAME_PATH.stat().st_mtime > FRAME_MAX_AGE:
            return None
    except OSError:
        return None
    return cv2.imread(str(FRAME_PATH))


def _get_camera() -> Camera:
    global _enroll_cam
    with _lock:
        if _enroll_cam is None:
            _enroll_cam = camera_from_config(cfg).start()
        return _enroll_cam


def frame_para_cadastro():
    """Frame para preview e captura do cadastro.

    Prioriza o frame publicado pelo worker. Isso não é otimização: webcam USB e
    câmera CSI são dispositivos V4L2 EXCLUSIVOS, então com o worker rodando a
    API simplesmente não consegue abrir a câmera. Consumindo o frame dele, o
    cadastro funciona sem precisar parar o reconhecimento.

    Se o worker estiver parado (frame ausente ou velho), abre a câmera
    diretamente — que é o caminho de sempre, e o único quando só a API roda.
    """
    img = _frame_do_worker()
    if img is not None:
        return img, "worker"
    cam = _get_camera()
    return cam.read(), "camera"


def _maybe_close_camera():
    global _enroll_cam
    with _lock:
        if not _sessions and _enroll_cam is not None:
            _enroll_cam.stop()
            _enroll_cam = None


def _public_samples(sess: dict):
    return [{"index": s["index"], "snapshot_url": f"/snapshots/{s['snapshot']}"}
            for s in sess["samples"]]


# ---- modelos de request --------------------------------------------------- #
class StartReq(BaseModel):
    name: str


class SessionReq(BaseModel):
    session_id: str


class SampleDelReq(BaseModel):
    session_id: str
    index: int


# ---- saúde ---------------------------------------------------------------- #
@app.get("/health")
def health():
    """Saúde do sistema — útil para checar o Pi remotamente pelo navegador."""
    live_age = None
    if LIVE_PATH.exists():
        live_age = round(time.time() - LIVE_PATH.stat().st_mtime, 1)

    # Modo REAL em execução, publicado pelo worker. Difere do config.yaml
    # quando o worker foi iniciado com --mode.
    worker_mode, tracks_pending = None, None
    try:
        with open(STATUS_PATH, encoding="utf-8") as fh:
            st = json.load(fh)
        if time.time() - st.get("updated_at", 0) < 600:
            worker_mode = st.get("mode")
            tracks_pending = st.get("pendentes")
    except (OSError, ValueError):
        pass

    return {
        "ok": True,
        "enroll_target": ENROLL_TARGET,
        "opencv": cv2.__version__,
        "people": len(db.list_people()),
        "worker_live_age_seconds": live_age,   # None ou muito alto => worker parado
        "worker_mode": worker_mode,            # None => worker parado ou antigo
        "tracks_pending": tracks_pending,      # só no modo captura
        "enroll_session_active": bool(_sessions),
        # "worker" => cadastro usa o frame publicado (funciona com o worker no ar);
        # "camera" => a API abre a câmera (só possível com o worker parado)
        "enroll_source": "worker" if _frame_do_worker() is not None else "camera",
    }


# ---- cadastro interativo -------------------------------------------------- #
@app.post("/enroll/start")
def enroll_start(req: StartReq):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Nome é obrigatório.")
    # Só abre a câmera se o worker NÃO estiver publicando frames — com webcam
    # USB, tentar abrir enquanto ele roda falha (dispositivo exclusivo).
    if _frame_do_worker() is None:
        _get_camera()
    sid = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[sid] = {"name": name, "samples": []}
    return {"session_id": sid, "name": name, "target": ENROLL_TARGET}


@app.get("/enroll/preview")
def enroll_preview():
    """Frame ao vivo (limpo) com a caixa do rosto detectado desenhada."""
    if not _sessions and _enroll_cam is None and _frame_do_worker() is None:
        raise HTTPException(404, "Nenhuma sessão de cadastro ativa.")
    frame, origem = frame_para_cadastro()
    if frame is None:
        raise HTTPException(
            503, "Ainda sem imagem. Se a câmera é USB e o worker está parado, "
                 "aguarde alguns segundos; se ele está rodando, verifique "
                 "'sudo systemctl status facial-worker'.")
    preview = frame.copy()
    for face in engine.detect(frame):
        draw_face(preview, face, "rosto", score=float(face[14]), known=True)
    ok, buf = cv2.imencode(".jpg", preview)
    if not ok:
        raise HTTPException(500, "Falha ao codificar o preview.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/enroll/capture")
def enroll_capture(req: SessionReq):
    """Captura UMA amostra a partir do frame atual (1 embedding por chamada)."""
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Sessão inválida ou expirada.")
    frame, origem = frame_para_cadastro()
    if frame is None:
        return {"ok": False, "message": "Ainda sem imagem da câmera."}

    face = engine.best_face(engine.detect(frame))
    if face is None:
        return {"ok": False, "message": "Nenhum rosto detectado. Ajuste a posição."}

    vec = engine.embed(frame, face)
    index = len(sess["samples"])
    crop = crop_face(frame, face)
    snapshot = store.save(crop, f"{sess['name']}_s{index + 1}", subdir=f"enroll/{req.session_id}")
    sess["samples"].append({"index": index, "embedding": vec, "snapshot": snapshot})
    return {"ok": True, "count": len(sess["samples"]), "target": ENROLL_TARGET,
            "samples": _public_samples(sess)}


@app.post("/enroll/sample/delete")
def enroll_delete(req: SampleDelReq):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Sessão inválida ou expirada.")
    if 0 <= req.index < len(sess["samples"]):
        sess["samples"].pop(req.index)
        for i, s in enumerate(sess["samples"]):  # reindexa
            s["index"] = i
    return {"ok": True, "count": len(sess["samples"]), "samples": _public_samples(sess)}


@app.get("/enroll/status")
def enroll_status(session_id: str):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(404, "Sessão inválida ou expirada.")
    return {"name": sess["name"], "count": len(sess["samples"]),
            "target": ENROLL_TARGET, "samples": _public_samples(sess)}


@app.post("/enroll/finish")
def enroll_finish(req: SessionReq):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "Sessão inválida ou expirada.")
    if not sess["samples"]:
        raise HTTPException(422, "Capture pelo menos uma amostra antes de concluir.")
    person_id = db.add_person(sess["name"])
    for s in sess["samples"]:
        db.add_embedding(person_id, s["embedding"])
    captured = len(sess["samples"])
    name = sess["name"]
    with _lock:
        _sessions.pop(req.session_id, None)
    _maybe_close_camera()
    return {"id": person_id, "name": name, "captured": captured}


@app.post("/enroll/cancel")
def enroll_cancel(req: SessionReq):
    with _lock:
        _sessions.pop(req.session_id, None)
    store.remove_dir(f"enroll/{req.session_id}")
    _maybe_close_camera()
    return {"ok": True}


# ---- pessoas / eventos / mídia -------------------------------------------- #
@app.get("/people")
def people():
    return db.list_people()


@app.delete("/people/{person_id}")
def delete_person(person_id: int):
    db.delete_person(person_id)
    return {"deleted": person_id}


@app.get("/events")
def events(limit: int = Query(50, ge=1, le=500)):
    return db.list_events(limit)


@app.get("/snapshots/{path:path}")
def snapshot(path: str):
    full = (SNAP_BASE / path).resolve()
    if not str(full).startswith(str(SNAP_BASE)) or not full.is_file():
        raise HTTPException(404, "Snapshot não encontrado.")
    return FileResponse(str(full), media_type="image/jpeg")


@app.get("/live.jpg")
def live():
    if not LIVE_PATH.exists():
        raise HTTPException(404, "Ainda não há preview ao vivo (o worker está rodando?).")
    return FileResponse(str(LIVE_PATH), media_type="image/jpeg")
