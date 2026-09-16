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

import ipaddress
import json
import secrets
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from core.camera import Camera, camera_from_config
from core.config import (frame_image_path, live_image_path, load_config,
                         project_path)
from core.database import (Database, InepDuplicado, inep_suspeito,
                          normalizar_inep)
from core.draw import crop_face, draw_face
from core.face_engine import FaceEngine
from core.imagem import ler as ler_imagem
from core.storage import SnapshotStore

cfg = load_config()
engine = FaceEngine(cfg)
db = Database(cfg.storage.db_path)
store = SnapshotStore(cfg.storage.snapshots_dir)

SNAP_BASE = project_path(cfg.storage.snapshots_dir).resolve()
# Os recortes do modo captura ficam em OUTRA base, então a exclusão precisa
# resolver cada caminho contra a base correta.
TRACKS_BASE = project_path(
    (cfg.get("tracking") or {}).get("crops_dir", "data/tracks")).resolve()
LIVE_PATH = live_image_path(cfg)
STATUS_PATH = LIVE_PATH.with_name("facial-status.json")
FRAME_PATH = frame_image_path(cfg)
FRAME_MAX_AGE = 5.0        # acima disso o frame do worker é velho demais
ENROLL_TARGET = int(cfg.enroll.frames_to_capture)  # amostras sugeridas por pessoa

# --------------------------------------------------------------------------- #
# Autenticação por EXPOSIÇÃO, não por flag.
#
# A regra é decidida por requisição, olhando de onde ela veio:
#
#   cliente em 127.0.0.1/::1  -> libera. Nada fora da máquina alcança isso,
#                                então exigir credencial não protegeria nada e
#                                só atrapalharia quem testa tudo local.
#   cliente remoto, com token -> valida.
#   cliente remoto, SEM token configurado -> 503. RECUSA.
#
# A última linha é a inversão que importa. Antes, token vazio significava "API
# aberta", e a proteção dependia de alguém lembrar de preencher o config. Agora
# o esquecimento resulta em porta fechada, não em porta escancarada.
#
# Amarrar a regra ao IP de origem, e não ao `api.host` do config, é proposital:
# o bind real vem da linha de comando do uvicorn (`--host`), então o config pode
# discordar da realidade. O IP de origem não mente.
#
# Por que não "recusar subir sem token": no Pi a API roda sob systemd com
# Restart=always. Morrer na inicialização produziria um laço de reinício com a
# causa escondida no journal — e um serviço morto é pior de diagnosticar que um
# 503 com mensagem explicando o que falta.
#
# ATENÇÃO para o futuro: se algum dia entrar um proxy reverso na frente, todas
# as requisições passarão a chegar de 127.0.0.1 e esta regra liberaria tudo. Aí
# é obrigatório usar `--forwarded-allow-ips` no uvicorn e ler o IP real.
# --------------------------------------------------------------------------- #
def _carregar_tokens(bloco) -> dict:
    """Monta {nome: token} aceitando as duas formas de configuração.

    `api.token: "abc"`            -> {"padrao": "abc"}      (compatibilidade)
    `api.tokens: {painel: "x"}`   -> {"painel": "x"}        (por consumidor)

    Token por consumidor existe porque vazamento é o cenário realista, não
    força bruta: com um token por consumidor você revoga o suspeito sem
    derrubar a integração com o sistema da escola. E o log passa a dizer QUEM
    chamou, que é o que permite auditar depois.
    """
    bloco = bloco or {}
    tokens = {}
    unico = str(bloco.get("token") or "").strip()
    if unico:
        tokens["padrao"] = unico
    for nome, valor in (bloco.get("tokens") or {}).items():
        valor = str(valor or "").strip()
        if valor:
            tokens[str(nome)] = valor
    return tokens


API_TOKENS = _carregar_tokens(cfg.get("api"))

# Rotas liberadas mesmo para cliente remoto autenticado-ou-não.
# Só /health: o painel a usa para detectar se a API está viva antes de ter
# token, e o monitoramento precisa dela. Devolve contagem de pessoas, nunca
# nomes. /docs, /openapi.json e /redoc SAÍRAM daqui: não vazam dado, mas
# entregam o mapa completo da API para quem estiver na rede.
ROTAS_ABERTAS = {"/health"}

DICA_TOKEN = (
    "Gere um com:  python -c \"import secrets;print(secrets.token_urlsafe(32))\"  "
    "e coloque em api.token no config.yaml do Pi e no do computador do painel."
)


def endereco_local(host: str) -> bool:
    """True se o endereço é loopback — inalcançável de fora da máquina.

    Usa `ipaddress` em vez de uma lista fixa {"127.0.0.1", "::1"} porque a
    lista deixava dois casos legítimos de fora:

      ::ffff:127.0.0.1  conexão IPv4 local chegando por socket IPv6 (uvicorn
                        em dual-stack). `is_loopback` devolve False para essa
                        forma, então é preciso desembrulhar o `ipv4_mapped`
                        antes de perguntar.
      127.0.0.2         todo o 127.0.0.0/8 é loopback, não só o .1.

    Qualquer um dos dois cairia como "remoto" e tomaria 503 em teste local —
    justamente a fricção que a regra por exposição existe para evitar.

    Endereço que não é IP (o TestClient usa "testclient") não é local: falha
    para o lado fechado, que é o certo.
    """
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapeado = getattr(addr, "ipv4_mapped", None)
    if mapeado is not None:
        addr = mapeado
    return addr.is_loopback


def cliente_local(request: Request) -> bool:
    return endereco_local(request.client.host if request.client else "")


def autorizar(request: Request):
    """Devolve (consumidor, erro). `erro` é (status, detalhe) ou None."""
    if cliente_local(request):
        return "local", None
    if request.url.path in ROTAS_ABERTAS:
        return None, None
    if not API_TOKENS:
        return None, (503, "Esta API está acessível pela rede e nenhum token "
                           "foi configurado, então o acesso remoto está "
                           f"recusado. {DICA_TOKEN}")

    fornecido = request.headers.get("x-api-token", "")
    if not fornecido:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            fornecido = auth[7:]

    # compare_digest em TODOS os candidatos, sem interromper no primeiro acerto:
    # sair mais cedo faria o tempo de resposta revelar quantos tokens existem.
    consumidor = None
    for nome, valor in API_TOKENS.items():
        if secrets.compare_digest(fornecido, valor):
            consumidor = nome
    if consumidor is None:
        return None, (401, "Token ausente ou inválido. Envie o cabeçalho "
                           "'X-API-Token: <token>' ou "
                           "'Authorization: Bearer <token>'.")
    return consumidor, None


app = FastAPI(title="Reconhecimento Facial — API")


# Middleware, e não `dependencies=[Depends(...)]` no app: as rotas /docs,
# /openapi.json e /redoc são registradas pelo FastAPI no nível do Starlette e
# NÃO executam dependências do router. Verificado em teste — com a dependência,
# as três respondiam 200 para cliente remoto sem token, entregando o mapa
# completo da API a quem estivesse na rede. O middleware roda antes do
# roteamento e cobre tudo com um mecanismo só.
@app.middleware("http")
async def autenticacao(request: Request, call_next):
    consumidor, erro = autorizar(request)
    if erro:
        status, detalhe = erro
        # Mesmo formato que o FastAPI usa em HTTPException, para o painel poder
        # ler `detail` sem tratar dois casos.
        return JSONResponse({"detail": detalhe}, status_code=status)
    request.state.consumidor = consumidor
    return await call_next(request)

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
    # imagem.ler, não cv2.imread: com caminho não-ASCII o imread devolve None
    # em silêncio, e o sintoma era a API dizer "ainda sem imagem" enquanto o
    # worker publicava o frame normalmente.
    return ler_imagem(FRAME_PATH)


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

    # Plano B é abrir a câmera aqui. Legítimo com RTSP, que aceita várias
    # conexões — e CONDENADO com webcam USB ou CSI, que são exclusivas: com o
    # worker rodando, a API disputa o dispositivo e perde quase sempre.
    #
    # Foi exatamente esse o bug do cadastro que exigia ~15 cliques para
    # capturar uma amostra. Silenciosamente competir produzia sucesso
    # esporádico, o que é muito pior que falhar: parecia instabilidade da
    # câmera, e escondeu por semanas que o frame do worker não estava sendo
    # alcançado (no Pi porque não era publicado; no Windows porque o imread
    # não lia caminho com acento).
    if _fonte_e_dispositivo_local() and _worker_vivo():
        return None, "conflito"

    cam = _get_camera()
    return cam.read(), "camera"


def _porque_sem_imagem(origem: str) -> str:
    """Explica a ausência de imagem conforme a causa, em vez de um texto só."""
    if origem == "conflito":
        return ("A câmera é local (webcam/CSI) e o worker está com o "
                "dispositivo, que aceita só um processo. A API deveria estar "
                "lendo o frame publicado por ele, e não está — verifique se o "
                f"arquivo {FRAME_PATH.name} está sendo atualizado na pasta "
                f"{FRAME_PATH.parent}. Alternativa: pare o worker durante o "
                "cadastro.")
    return ("Ainda sem imagem da câmera. Se o worker acabou de subir, aguarde "
            "alguns segundos; se está parado, verifique se ele está rodando.")


def _fonte_e_dispositivo_local() -> bool:
    """True quando a câmera é webcam/CSI, que só aceita UM processo."""
    from core.camera import _parse_source
    return _parse_source(cfg.camera.rtsp_url)[1] != cv2.CAP_FFMPEG


def _worker_vivo() -> bool:
    """O worker publicou status recentemente?"""
    try:
        with open(STATUS_PATH, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return False
    return time.time() - float(st.get("updated_at", 0)) < 60


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
    # ID INEP do aluno. Opcional no cadastro para não travar o
    # piloto, mas sem ele o sistema da escola não consegue casar.
    inep_id: str = ""


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
        _sessions[sid] = {"name": name, "samples": [],
                          "inep_id": normalizar_inep(req.inep_id)}
    return {"session_id": sid, "name": name, "target": ENROLL_TARGET}


@app.get("/enroll/preview")
def enroll_preview():
    """Frame ao vivo (limpo) com a caixa do rosto detectado desenhada."""
    if not _sessions and _enroll_cam is None and _frame_do_worker() is None:
        raise HTTPException(404, "Nenhuma sessão de cadastro ativa.")
    frame, origem = frame_para_cadastro()
    if frame is None:
        raise HTTPException(503, _porque_sem_imagem(origem))
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
        return {"ok": False, "message": _porque_sem_imagem(origem)}

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
    try:
        person_id = db.add_person(sess["name"], sess.get("inep_id"))
    except InepDuplicado as exc:
        raise HTTPException(409, str(exc)) from None

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


def _apagar_arquivo(base, rel: str) -> bool:
    """Remove um arquivo dentro de `base`, recusando caminho que escape dela."""
    if not rel:
        return False
    alvo = (base / rel).resolve()
    if not str(alvo).startswith(str(base)) or not alvo.is_file():
        return False
    alvo.unlink(missing_ok=True)
    return True


@app.delete("/people/{person_id}")
def delete_person(person_id: int,
                  anonimizar: bool = Query(
                      False, description="preserva as passagens sem identificar "
                                         "quem passou, em vez de apagá-las")):
    """Exclusão completa: pessoa, embeddings, histórico e TODAS as imagens.

    Antes isso deixava para trás os eventos, os snapshots das passagens e os
    recortes das trilhas — ou seja, imagem de rosto continuava no disco depois
    de um pedido de exclusão. Para dado biométrico de criança, 'excluir' precisa
    excluir de fato.

    Com `?anonimizar=true`, as linhas de passagem ficam com `person_id` nulo e
    nome neutro: a contagem de "alguém passou às 7:42" sobrevive para
    estatística, sem identificar. As IMAGENS são apagadas nos dois modos, porque
    a foto do rosto é justamente o dado que identifica.
    """
    if db.get_person(person_id) is None:
        raise HTTPException(404, "Pessoa não encontrada.")

    arquivos = db.delete_person(person_id, anonimizar=anonimizar)

    removidas = sum(_apagar_arquivo(SNAP_BASE, r) for r in arquivos["snapshots"])
    removidos_recortes = sum(_apagar_arquivo(TRACKS_BASE, r)
                             for r in arquivos["tracks"])

    pasta = (SNAP_BASE / AMOSTRAS_SUBDIR / str(person_id)).resolve()
    if str(pasta).startswith(str(SNAP_BASE)) and pasta.is_dir():
        shutil.rmtree(pasta, ignore_errors=True)

    return {
        "deleted": person_id,
        "modo": "anonimizado" if anonimizar else "apagado",
        "fotos_removidas": removidas,
        "recortes_removidos": removidos_recortes,
        "historico": ("preservado sem identificação" if anonimizar
                      else "apagado junto com a pessoa"),
    }


class RenameReq(BaseModel):
    name: str | None = None
    inep_id: str | None = None


@app.patch("/people/{person_id}")
def update_person(person_id: int, req: RenameReq):
    """Altera nome e/ou ID INEP. Campo ausente fica como está.

    O ID INEP é a identificação única do aluno no Censo Escolar, e é por ele
    que o sistema de gestão da escola casa os registros. Casar por nome é
    frágil: homônimos, acentuação e digitação divergente quebram a associação.

    Enviar `inep_id: ""` limpa o campo.
    """
    if db.get_person(person_id) is None:
        raise HTTPException(404, "Pessoa não encontrada.")

    resposta = {"id": person_id}

    if req.name is not None:
        nome = req.name.strip()
        if not nome:
            raise HTTPException(400, "Nome não pode ficar vazio.")
        db.rename_person(person_id, nome)
        resposta["name"] = nome

    if req.inep_id is not None:
        try:
            gravado = db.set_person_inep(person_id, req.inep_id)
        except InepDuplicado as exc:
            raise HTTPException(409, str(exc)) from None
        resposta["inep_id"] = gravado
        aviso = inep_suspeito(gravado or "")
        if aviso:
            resposta["aviso"] = f"ID INEP {aviso}"

    return resposta


@app.get("/people/by-inep/{inep_id}")
def person_by_inep(inep_id: str):
    """Busca aluno pelo ID INEP — o caminho de entrada para outro sistema."""
    pessoa = db.person_by_inep(inep_id)
    if pessoa is None:
        raise HTTPException(404, "Nenhum aluno cadastrado com este ID INEP.")
    return pessoa


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
        return {"ok": False, "message": _porque_sem_imagem(origem)}
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


def _montar_chamada(dia: str = "", inicio: str = "", fim: str = "") -> dict:
    """Monta a chamada. Função comum, NÃO rota.

    Existe separada porque a rota tem `Query(...)` nos parâmetros — chamá-la
    como função Python passaria os objetos de dependência em vez dos valores
    padrão. O fechamento precisa deste resumo, então ele chama daqui.
    """

    t0, t1 = _janela(dia, inicio, fim)
    dia_iso = time.strftime("%Y-%m-%d", time.localtime(t0))

    automatico = {l["person_id"]: l for l in db.attendance(t0, t1)}
    # Foto da melhor detecção de cada pessoa, para a conferência humana poder
    # olhar o rosto antes de confirmar.
    fotos = db.melhores_fotos(t0, t1)
    correcoes = db.attendance_overrides(dia_iso)
    fechamento = db.attendance_closure(dia_iso)
    todos = db.list_people()
    pendentes = db.count_tracks_by_status().get("pendente", 0)

    presentes, ausentes = [], []
    n_marcados_presentes = n_marcados_ausentes = 0

    for p in todos:
        pid = p["id"]
        auto = automatico.get(pid)
        corr = correcoes.get(pid)
        detectado = auto is not None
        # A correção manual, quando existe, prevalece sobre o automático.
        presente = bool(corr["presente"]) if corr else detectado

        if corr and corr["presente"] and not detectado:
            origem = "manual_presente"          # falso negativo do reconhecimento
            n_marcados_presentes += 1
        elif corr and not corr["presente"] and detectado:
            origem = "manual_ausente"           # falso positivo do reconhecimento
            n_marcados_ausentes += 1
        elif detectado:
            origem = "automatico"
        else:
            origem = "nao_identificado"

        linha = {"person_id": pid, "inep_id": p.get("inep_id"),
                 "nome": p["name"], "origem": origem,
                 "detectado_pelo_sistema": detectado}
        if corr:
            linha["correcao"] = {"motivo": corr["motivo"] or None,
                                 "autor": corr["autor"] or None,
                                 "em": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                     time.localtime(corr["created_at"]))}
        if auto:
            linha.update({
                "primeira_vez": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                              time.localtime(auto["primeira"])),
                "ultima_vez": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                            time.localtime(auto["ultima"])),
                "passagens": auto["passagens"],
                "melhor_score": round(auto["melhor_score"], 3),
                "fontes": (auto["fontes"] or "").split(","),
            })
            # As duas origens guardam imagem em bases diferentes, então a URL
            # precisa sair daqui pronta — quem consome não deveria ter que
            # saber dessa separação interna.
            foto = fotos.get(pid)
            if foto and foto["foto"]:
                base = "/snapshots/" if foto["fonte"] == "realtime" else "/tracks/"
                linha["foto_url"] = f"{base}{foto['foto']}"
                linha["foto_score"] = round(foto["score"], 3)
        (presentes if presente else ausentes).append(linha)

    presentes.sort(key=lambda x: (x.get("primeira_vez") or "~", x["nome"]))
    ausentes.sort(key=lambda x: x["nome"].lower())

    return {
        "periodo": {
            "dia": dia_iso,
            "inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
            "fim": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
            "fuso": time.strftime("%Z", time.localtime(t0)),
        },
        "completo": pendentes == 0,
        "trilhas_pendentes": pendentes,
        # `conferida` é mais forte que `completo`: significa que uma PESSOA
        # revisou e fechou. Quem for gravar falta deveria exigir isto.
        "conferida": fechamento is not None,
        "fechamento": ({"em": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                            time.localtime(fechamento["closed_at"])),
                        "autor": fechamento["autor"] or None}
                       if fechamento else None),
        "total_cadastrados": len(todos),
        "total_presentes": len(presentes),
        # Quem está sem ID INEP não pode ser casado pelo sistema da escola.
        # Exposto aqui para o problema aparecer antes de virar falta errada.
        "sem_inep": sum(1 for p in todos if not p.get("inep_id")),
        "correcoes": {
            # Estes dois números medem o acerto do reconhecimento no dia:
            # presentes marcados à mão = o sistema deixou passar;
            # ausentes marcados à mão  = o sistema identificou errado.
            "marcados_presentes": n_marcados_presentes,
            "marcados_ausentes": n_marcados_ausentes,
            "total": len(correcoes),
        },
        "presentes": presentes,
        "ausentes": ausentes,
        "aviso": (None if fechamento else
                  "Chamada NÃO conferida. Quem não foi identificado pode ter "
                  "passado sem ser detectado — revise antes de registrar falta."),
    }


@app.get("/attendance")
def attendance(dia: str = Query("", description="AAAA-MM-DD; vazio = hoje"),
               inicio: str = Query("", description="HH:MM; vazio = 00:00"),
               fim: str = Query("", description="HH:MM; vazio = 23:59")):
    """Chamada do período, para consumo por outro sistema.

    Pontos de atenção para quem integra:

    `conferida` — true só depois que uma PESSOA revisou e fechou a chamada.
      É o campo mais forte da resposta: só uma chamada conferida deveria
      alimentar registro de falta.

    `completo` — false quando ainda há trilhas aguardando reconhecimento. A
      chamada está INCOMPLETA nesse caso.

    `ausentes` — inclui quem o sistema não identificou. NÃO equivale a falta:
      pode ser falha de captura ou score abaixo do limiar. O campo `origem` de
      cada pessoa diz se a informação veio do reconhecimento ou de correção
      manual.

    `correcoes` — mede o acerto do reconhecimento no dia: `marcados_presentes`
      são falsos negativos (o sistema deixou passar) e `marcados_ausentes` são
      falsos positivos (identificou errado).

    `person_id` é o identificador interno deste sistema. Casar por nome é
      frágil (homônimos, acentuação, digitação); para integração de verdade,
      convém guardar a matrícula do aluno aqui e casar por ela.
    """
    return _montar_chamada(dia, inicio, fim)


@app.get("/detections")
def detections(person_id: int = Query(0, description="0 = todas as pessoas"),
               dia: str = Query("", description="AAAA-MM-DD; vazio = todos"),
               limit: int = Query(120, ge=1, le=1000)):
    """Detecções das DUAS origens, filtráveis por pessoa e dia.

    Difere de `/events`, que lê só a tabela `events` e portanto devolve vazio
    no modo captura — o modo usado na escola. Cada linha já vem com a URL da
    foto na rota certa (snapshots e recortes de trilha ficam em bases
    diferentes) e com o rótulo manual, se houver.
    """
    t0 = t1 = None
    if dia:
        t0, t1 = _janela(dia, "", "")
    linhas = db.deteccoes_de(person_id or None, t0, t1, limit)
    rotulos = db.detection_labels()

    saida = []
    for l in linhas:
        base = "/snapshots/" if l["fonte"] == "evento" else "/tracks/"
        saida.append({
            "fonte": l["fonte"],
            "id": int(l["id"]),
            # `id` colide entre as origens; a chave é o par.
            "chave": f"{l['fonte']}:{int(l['id'])}",
            "person_id": l["person_id"],
            "nome": l["name"],
            "score": round(float(l["score"] or 0.0), 3),
            "ts": l["ts"],
            "quando": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                    time.localtime(l["ts"])),
            "is_known": bool(l["is_known"]),
            "foto_url": f"{base}{l['foto']}" if l["foto"] else None,
            "rotulo": rotulos.get(f"{l['fonte']}:{int(l['id'])}"),
        })
    return {"total": len(saida), "deteccoes": saida}


class RotuloReq(BaseModel):
    fonte: str
    detection_id: int
    rotulo: str = "errado"
    autor: str = ""


@app.post("/detections/label")
def rotular_deteccao(req: RotuloReq):
    """Marca uma detecção como 'certo' ou 'errado'.

    NÃO altera a presença — decisão deliberada. Rotular mede o acerto do
    reconhecimento; quem esteve na escola é decidido na chamada. Se uma
    revisão de fotos mexesse na frequência do aluno, ninguém confiaria em
    nenhuma das duas.
    """
    try:
        db.set_detection_label(req.fonte, req.detection_id, req.rotulo,
                               req.autor)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "chave": f"{req.fonte}:{req.detection_id}",
            "rotulo": req.rotulo,
            "observacao": "A presença na chamada não foi alterada."}


@app.delete("/detections/label")
def remover_rotulo(fonte: str = Query(...), detection_id: int = Query(...)):
    removidos = db.remove_detection_label(fonte, detection_id)
    return {"ok": True, "removidos": removidos}


class OverrideReq(BaseModel):
    dia: str
    person_id: int
    presente: bool
    motivo: str = ""
    autor: str = ""


def _validar_dia(dia: str) -> str:
    try:
        time.strptime(dia, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Campo 'dia' inválido. Use AAAA-MM-DD.")
    return dia


@app.post("/attendance/override")
def set_override(req: OverrideReq):
    """Corrige manualmente a presença de uma pessoa num dia.

    Não altera o que o reconhecimento detectou — grava a discordância à parte.
    Isso preserva a evidência e, de bônus, produz a medida de acerto do sistema:
    cada correção é um erro dele, com o tipo identificado.
    """
    dia = _validar_dia(req.dia)
    if db.get_person(req.person_id) is None:
        raise HTTPException(404, "Pessoa não encontrada.")
    if db.attendance_closure(dia) is not None:
        raise HTTPException(
            409, "A chamada deste dia está fechada. Reabra antes de corrigir: "
                 "DELETE /attendance/close?dia=" + dia)
    db.set_attendance_override(dia, req.person_id, req.presente,
                               req.motivo.strip(), req.autor.strip())
    return {"ok": True, "dia": dia, "person_id": req.person_id,
            "presente": req.presente}


@app.delete("/attendance/override")
def del_override(dia: str = Query(...), person_id: int = Query(...)):
    """Desfaz a correção: volta a valer o que o reconhecimento disse."""
    dia = _validar_dia(dia)
    if db.attendance_closure(dia) is not None:
        raise HTTPException(409, "A chamada deste dia está fechada.")
    return {"removidas": db.remove_attendance_override(dia, person_id)}


class CloseReq(BaseModel):
    dia: str
    autor: str = ""


@app.post("/attendance/close")
def close_attendance(req: CloseReq):
    """Fecha a chamada do dia, marcando que uma pessoa a conferiu."""
    dia = _validar_dia(req.dia)
    pendentes = db.count_tracks_by_status().get("pendente", 0)
    if pendentes:
        raise HTTPException(
            409, f"Há {pendentes} trilha(s) aguardando reconhecimento. Rode o "
                 "lote antes de fechar, senão a chamada fecha incompleta: "
                 "python scripts/recognize_batch.py")

    resumo = _montar_chamada(dia=dia)
    db.close_attendance(dia, req.autor.strip(), resumo["total_presentes"],
                        resumo["correcoes"]["total"])
    return {"ok": True, "dia": dia, "presentes": resumo["total_presentes"],
            "correcoes": resumo["correcoes"]["total"]}


@app.delete("/attendance/close")
def reopen_attendance(dia: str = Query(...)):
    """Reabre a chamada para nova conferência."""
    return {"reaberta": db.reopen_attendance(_validar_dia(dia))}


@app.get("/snapshots/{path:path}")
def snapshot(path: str):
    return _servir_imagem(SNAP_BASE, path, "Snapshot não encontrado.")


@app.get("/tracks/{path:path}")
def track_crop(path: str):
    """Recorte guardado pelo modo captura.

    Os recortes ficam em `tracking.crops_dir`, base DIFERENTE da de snapshots.
    Sem esta rota eles não tinham como ser exibidos, e tanto a chamada quanto
    os relatórios de calibração e de recall montavam URLs sob /snapshots/ que
    devolviam 404 para tudo que viesse do modo captura — justamente o modo
    usado em produção.
    """
    return _servir_imagem(TRACKS_BASE, path, "Recorte não encontrado.")


def _servir_imagem(base, path: str, erro: str):
    full = (base / path).resolve()
    # is_relative_to compara COMPONENTES de caminho. Um `startswith` de string
    # deixaria passar diretório irmão de nome parecido ("snapshots-privado" ao
    # lado de "snapshots"). O `.resolve()` barra o "../../etc/passwd".
    if not full.is_relative_to(base) or not full.is_file():
        raise HTTPException(404, erro)
    return FileResponse(str(full), media_type="image/jpeg")


@app.get("/live.jpg")
def live():
    if not LIVE_PATH.exists():
        raise HTTPException(404, "Ainda não há preview ao vivo (o worker está rodando?).")
    return FileResponse(str(LIVE_PATH), media_type="image/jpeg")
