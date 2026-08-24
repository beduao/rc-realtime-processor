"""API HTTP (FastAPI) — roda junto do worker (no Pi, ou no seu computador na Fase 1).

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
import secrets
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
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

# --------------------------------------------------------------------------- #
# Autenticação opcional por token.
#
# Sem `api.token` no config.yaml, nada muda — a API segue aberta, como sempre.
# Definindo o token, TODAS as rotas passam a exigi-lo, exceto /health (que
# precisa ficar aberta para monitoramento e para o painel detectar a API).
# É opcional porque ligar de repente trancaria o painel; e existe porque um
# endpoint que devolve nomes de crianças por HTTP não deveria ficar sem
# nenhuma barreira quando for consumido por outro sistema.
# --------------------------------------------------------------------------- #
API_TOKEN = str((cfg.get("api") or {}).get("token") or "").strip()


ROTAS_ABERTAS = {"/health", "/docs", "/openapi.json", "/redoc"}


def exigir_token(request: Request,
                 x_api_token: str = Header(default=""),
                 authorization: str = Header(default="")):
    if not API_TOKEN or request.url.path in ROTAS_ABERTAS:
        return
    fornecido = x_api_token or ""
    if not fornecido and authorization.lower().startswith("bearer "):
        fornecido = authorization[7:]
    # compare_digest evita vazar informação pelo tempo de comparação
    if not secrets.compare_digest(fornecido, API_TOKEN):
        raise HTTPException(
            401, "Token ausente ou inválido. Envie o cabeçalho "
                 "'X-API-Token: <token>' ou 'Authorization: Bearer <token>'.")


app = FastAPI(title="Reconhecimento Facial — API",
              dependencies=[Depends(exigir_token)])

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


AMOSTRAS_SUBDIR = "amostras"          # snapshots/amostras/<person_id>/


def _nitidez(imagem) -> float:
    """Variância do Laplaciano: quanto maior, mais nítido o recorte."""
    try:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        return round(float(cv2.Laplacian(cinza, cv2.CV_64F).var()), 1)
    except cv2.error:
        return 0.0


def _guardar_amostra(person_id: int, crop, rotulo: str) -> str:
    """Salva o recorte na pasta definitiva da pessoa e devolve o caminho relativo."""
    return store.save(crop, rotulo, subdir=f"{AMOSTRAS_SUBDIR}/{person_id}")


def _mover_para_amostras(origem, person_id: int) -> str:
    """Move um recorte já salvo para a pasta da pessoa. Devolve o novo caminho."""
    destino_dir = SNAP_BASE / AMOSTRAS_SUBDIR / str(person_id)
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / origem.name
    shutil.move(str(origem), str(destino))
    return f"{AMOSTRAS_SUBDIR}/{person_id}/{origem.name}"


def _redundancia(amostras: list[dict]) -> list[dict]:
    """Para cada amostra, a maior similaridade com as OUTRAS da mesma pessoa.

    Amostra quase idêntica a outra não acrescenta informação ao cadastro — ela
    infla a contagem sem melhorar o reconhecimento. Como os vetores estão
    normalizados, o produto interno já é o cosseno.
    """
    saida = []
    for i, a in enumerate(amostras):
        melhor, parceiro = 0.0, None
        for j, b in enumerate(amostras):
            if i == j:
                continue
            sim = float(np.dot(a["vec"], b["vec"]))
            if sim > melhor:
                melhor, parceiro = sim, b["id"]
        saida.append({
            "id": a["id"],
            "snapshot_url": (f"/snapshots/{a['snapshot_path']}"
                             if a["snapshot_path"] else None),
            "created_at": a["created_at"],
            "quality": a["quality"],
            "similaridade_maxima": round(melhor, 3) if parceiro else None,
            "parecida_com": parceiro,
        })
    return saida


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
    sess["samples"].append({"index": index, "embedding": vec, "snapshot": snapshot,
                            "quality": _nitidez(crop)})
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

    # Move os recortes da pasta temporária da sessão para a pasta definitiva da
    # pessoa. Assim cada embedding fica ligado à sua foto (é o que permite
    # revisar as amostras depois) e apagar a pessoa remove as imagens dela.
    for s in sess["samples"]:
        destino = None
        try:
            origem = SNAP_BASE / s["snapshot"]
            destino = _mover_para_amostras(origem, person_id)
        except OSError as exc:
            print(f"[api] não movi o recorte da amostra: {exc}", flush=True)
        db.add_embedding(person_id, s["embedding"], destino, s.get("quality"))

    store.remove_dir(f"enroll/{req.session_id}")
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
    """Apaga a pessoa, seus embeddings E as fotos das amostras dela.

    Antes as imagens ficavam no disco depois da exclusão. Para dado biométrico
    isso é problema: 'excluir' precisa excluir de fato.
    """
    caminhos = db.delete_person(person_id)
    removidas = 0
    for rel in caminhos:
        arq = (SNAP_BASE / rel).resolve()
        if str(arq).startswith(str(SNAP_BASE)) and arq.is_file():
            arq.unlink(missing_ok=True)
            removidas += 1
    pasta = (SNAP_BASE / AMOSTRAS_SUBDIR / str(person_id)).resolve()
    if str(pasta).startswith(str(SNAP_BASE)) and pasta.is_dir():
        shutil.rmtree(pasta, ignore_errors=True)
    return {"deleted": person_id, "fotos_removidas": removidas,
            "aviso": "O histórico de reconhecimentos (eventos) foi mantido."}


class RenameReq(BaseModel):
    name: str


@app.patch("/people/{person_id}")
def rename_person(person_id: int, req: RenameReq):
    nome = req.name.strip()
    if not nome:
        raise HTTPException(400, "Nome não pode ficar vazio.")
    if db.get_person(person_id) is None:
        raise HTTPException(404, "Pessoa não encontrada.")
    db.rename_person(person_id, nome)
    return {"id": person_id, "name": nome}


@app.get("/people/{person_id}/samples")
def person_samples(person_id: int):
    """Amostras da pessoa, com nitidez e indicação de redundância."""
    pessoa = db.get_person(person_id)
    if pessoa is None:
        raise HTTPException(404, "Pessoa não encontrada.")
    amostras = db.list_embeddings(person_id)
    return {"person": pessoa, "samples": _redundancia(amostras),
            "sem_foto": sum(1 for a in amostras if not a["snapshot_path"])}


@app.post("/people/{person_id}/samples")
def add_sample(person_id: int):
    """Acrescenta uma amostra a quem já está cadastrado, do frame atual.

    É a forma mais eficaz de reduzir falso positivo: mais amostras da mesma
    pessoa, em ângulos e luz variados, elevam o score dos acertos e permitem
    subir o limiar sem perdê-la.
    """
    pessoa = db.get_person(person_id)
    if pessoa is None:
        raise HTTPException(404, "Pessoa não encontrada.")

    frame, origem = frame_para_cadastro()
    if frame is None:
        return {"ok": False, "message": "Ainda sem imagem da câmera."}
    face = engine.best_face(engine.detect(frame))
    if face is None:
        return {"ok": False, "message": "Nenhum rosto detectado. Ajuste a posição."}

    vec = engine.embed(frame, face)
    crop = crop_face(frame, face)
    n = db.count_embeddings(person_id) + 1
    caminho = _guardar_amostra(person_id, crop, f"{pessoa['name']}_s{n}")
    eid = db.add_embedding(person_id, vec, caminho, _nitidez(crop))
    _maybe_close_camera()
    return {"ok": True, "id": eid, "origem": origem,
            "snapshot_url": f"/snapshots/{caminho}",
            "total": db.count_embeddings(person_id)}


@app.delete("/embeddings/{embedding_id}")
def delete_sample(embedding_id: int):
    """Remove UMA amostra, com trava para não deixar a pessoa sem nenhuma.

    Pessoa sem embedding nunca mais seria reconhecida, mas continuaria na lista
    de cadastrados — um estado silenciosamente quebrado.
    """
    amostra = db.get_embedding(embedding_id)
    if amostra is None:
        raise HTTPException(404, "Amostra não encontrada.")
    if db.count_embeddings(amostra["person_id"]) <= 1:
        raise HTTPException(
            409, "Esta é a última amostra da pessoa. Adicione outra antes de "
                 "remover, ou exclua a pessoa inteira.")

    db.delete_embedding(embedding_id)
    if amostra["snapshot_path"]:
        arq = (SNAP_BASE / amostra["snapshot_path"]).resolve()
        if str(arq).startswith(str(SNAP_BASE)) and arq.is_file():
            arq.unlink(missing_ok=True)
    return {"deleted": embedding_id,
            "restantes": db.count_embeddings(amostra["person_id"])}


@app.get("/events")
def events(limit: int = Query(50, ge=1, le=500)):
    return db.list_events(limit)


def _janela(dia: str, inicio: str, fim: str):
    """Converte dia + horas em uma janela de timestamps no fuso LOCAL do Pi.

    O fuso importa: o consumidor precisa saber a que 'dia' os dados se referem,
    e comparar horário local com UTC produziria chamada de outro dia perto da
    meia-noite. Devolvemos também os limites em ISO para o outro sistema poder
    conferir o que foi consultado.
    """
    if dia:
        try:
            d = time.strptime(dia, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Parâmetro 'dia' inválido. Use AAAA-MM-DD.")
        base = (d.tm_year, d.tm_mon, d.tm_mday)
    else:
        agora = time.localtime()
        base = (agora.tm_year, agora.tm_mon, agora.tm_mday)

    def hora(texto, padrao):
        if not texto:
            return padrao
        try:
            h, _, m = texto.partition(":")
            h, m = int(h), int(m or 0)
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError
            return h, m
        except ValueError:
            raise HTTPException(400, f"Horário inválido: {texto!r}. Use HH:MM.")

    h0, m0 = hora(inicio, (0, 0))
    h1, m1 = hora(fim, (23, 59))
    t0 = time.mktime((*base, h0, m0, 0, 0, 0, -1))
    t1 = time.mktime((*base, h1, m1, 59, 0, 0, -1))
    if t1 <= t0:
        raise HTTPException(400, "'fim' precisa ser depois de 'inicio'.")
    return t0, t1


@app.get("/attendance")
def attendance(dia: str = Query("", description="AAAA-MM-DD; vazio = hoje"),
               inicio: str = Query("", description="HH:MM; vazio = 00:00"),
               fim: str = Query("", description="HH:MM; vazio = 23:59")):
    """Chamada do período, para consumo por outro sistema.

    Pontos de atenção para quem integra:

    `completo`  — false quando ainda há trilhas aguardando reconhecimento. A
      chamada está INCOMPLETA nesse caso: quem consumir e marcar falta vai
      marcar falta de quem talvez esteja na fila. Sempre verifique este campo
      antes de gravar ausência.

    `nao_identificados` — pessoas cadastradas que não foram vistas na janela.
      NÃO é o mesmo que ausente: pode ser falha de captura, criança que passou
      fora do enquadramento, ou reconhecimento que não atingiu o limiar. A
      decisão de transformar isso em falta é do outro sistema, e deveria passar
      por conferência humana.

    `person_id` é o identificador interno deste sistema. Casar por nome é
      frágil (homônimos, acentuação, digitação); para integração de verdade,
      convém guardar a matrícula do aluno aqui e casar por ela.
    """
    t0, t1 = _janela(dia, inicio, fim)
    presentes = db.attendance(t0, t1)
    todos = db.list_people()
    vistos = {l["person_id"] for l in presentes}
    pendentes = db.count_tracks_by_status().get("pendente", 0)

    return {
        "periodo": {
            "dia": time.strftime("%Y-%m-%d", time.localtime(t0)),
            "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
            "fim": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
            "fuso": time.strftime("%Z", time.localtime(t0)),
        },
        "completo": pendentes == 0,
        "trilhas_pendentes": pendentes,
        "total_cadastrados": len(todos),
        "total_presentes": len(presentes),
        "presentes": [
            {
                "person_id": l["person_id"],
                "nome": l["name"],
                "primeira_vez": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                              time.localtime(l["primeira"])),
                "ultima_vez": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                            time.localtime(l["ultima"])),
                "passagens": l["passagens"],
                "melhor_score": round(l["melhor_score"], 3),
                "fontes": (l["fontes"] or "").split(","),
            }
            for l in presentes
        ],
        "nao_identificados": [
            {"person_id": p["id"], "nome": p["name"]}
            for p in todos if p["id"] not in vistos
        ],
        "aviso": ("'nao_identificados' não significa ausente — confira antes de "
                  "registrar falta." if len(presentes) < len(todos) else None),
    }


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
