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
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               Response)
from pydantic import BaseModel

from core.camera import Camera, camera_from_config
from core.config import (frame_image_path, live_image_path, load_config,
                         project_path)
from core.database import (Database, MatriculaDuplicada,
                          normalizar_matricula)
from core.draw import crop_face, draw_face
from core.face_engine import FaceEngine
from core.imagem import LARGURAS_MINIATURA, caminho_miniatura, gerar_miniatura
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

# Nitidez mínima de uma amostra de cadastro — a variância do Laplaciano do
# recorte, o mesmo número que a aba Pessoas mostra como "nitidez". 0 desliga.
#
# Por que avisar: rosto borrado ou mal visível vira um embedding genérico,
# parecido com o de qualquer outro rosto borrado, e na galeria ele "atrai"
# capturas ruins de OUTRAS pessoas. Foi a causa da confusão lais -> bia: três
# amostras dela tinham nitidez 171, 257 e 274, e nenhuma amostra boa do piloto
# ficou abaixo de 407. O 300 sai daí — dados da webcam do PC. A escala muda
# com câmera e resolução, então confira as amostras boas no Pi antes de
# confiar no padrão.
_nitidez_cfg = cfg.enroll.get("nitidez_minima")
NITIDEZ_MINIMA = 300.0 if _nitidez_cfg is None else float(_nitidez_cfg)

# Status do worker mais velho que isto = worker parado. Ele republica a cada
# 10s (worker.STATUS_EVERY_SECONDS); 30s tolera duas publicações perdidas.
WORKER_VIVO_SEGUNDOS = 30
# Sessão de cadastro sem uso por este tempo expira. Sem prazo, quem fechasse
# o navegador no meio deixava a sessão — e a câmera que ela segura — para
# sempre na memória da API.
SESSAO_EXPIRA_SEGUNDOS = 15 * 60
# Câmera aberta pela API, sem sessão e sem uso por este tempo, é fechada.
CAMERA_OCIOSA_SEGUNDOS = 60
VARREDURA_SEGUNDOS = 30    # intervalo da faxina do cadastro

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
# `/session` fica aberta por necessidade lógica: é a rota que autentica, então
# exigir autenticação nela impediria qualquer login. Ela valida o token por
# conta própria, com compare_digest, e o GET não revela nada — só se a sessão
# atual vale e se token é exigido nesta origem.
ROTAS_ABERTAS = {"/health", "/session"}

# Nome do cookie de sessão da interface web.
COOKIE_SESSAO = "facial_sessao"

# Prefixo dos arquivos da interface. Fica aberto porque HTML, CSS e JS não
# contêm dado de ninguém — é só o código da tela. Todo dado continua atrás
# das rotas protegidas, e sem cookie válido a tela carrega vazia pedindo o
# token. Proteger os estáticos impediria a própria tela de login de carregar.
PREFIXO_PAINEL = "/painel"

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
    if request.url.path.startswith(PREFIXO_PAINEL):
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
    if not fornecido:
        # Cookie, para a interface web. O navegador o envia sozinho, então o
        # JavaScript nunca precisa conhecer o token — diferente de guardá-lo em
        # localStorage, onde qualquer script da página o leria. É HttpOnly.
        fornecido = request.cookies.get(COOKIE_SESSAO, "")

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


@asynccontextmanager
async def _ciclo_de_vida(_app):
    """Mantém a faxina do cadastro rodando enquanto a API estiver no ar."""
    parar = threading.Event()
    threading.Thread(target=_faxina_periodica, args=(parar,), daemon=True,
                     name="faxina-cadastro").start()
    try:
        yield
    finally:
        parar.set()


app = FastAPI(title="Reconhecimento Facial — API", lifespan=_ciclo_de_vida)


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
# session_id -> {name, matricula, ultimo_uso, samples:[{index,embedding,snapshot}]}
_sessions: dict[str, dict] = {}
_enroll_cam: Camera | None = None
_camera_usada_em = 0.0     # último frame que a API tirou da própria câmera


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
    if not _pode_abrir_camera():
        return None, "conflito"

    global _camera_usada_em
    cam = _get_camera()
    _camera_usada_em = time.time()
    frame = cam.read()
    if frame is None:
        # Câmera recém-aberta ainda sem o primeiro frame. Em "adicionar
        # amostra" ela é aberta no próprio clique; sem esta espera o primeiro
        # clique respondia "sem imagem" e só o segundo funcionava.
        frame = cam.read_wait(timeout=2.0)
    return frame, "camera"


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


def _ler_status() -> dict:
    """Último status publicado pelo worker, ou {} se não houver/estiver ilegível."""
    try:
        with open(STATUS_PATH, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def _worker_vivo() -> bool:
    """O worker está no ar — e, portanto, dono da câmera?

    O worker republica o status a cada 10s por uma thread própria, chegue
    frame ou não. O limite antigo era 60s com publicação a cada 5 minutos, e
    a resposta era "parado" na maior parte do tempo.

    `encerrado` é o aviso que o worker deixa ao sair, para a API não ter que
    esperar o status envelhecer.
    """
    st = _ler_status()
    if not st or st.get("encerrado"):
        return False
    try:
        return time.time() - float(st.get("updated_at", 0)) < WORKER_VIVO_SEGUNDOS
    except (TypeError, ValueError):
        return False


def _pode_abrir_camera() -> bool:
    """A API pode abrir a câmera sem tomá-la do worker?

    RTSP aceita várias conexões, então sempre pode. Webcam e CSI aceitam um
    processo só: com o worker no ar, abrir aqui é disputar o dispositivo.
    """
    return not (_fonte_e_dispositivo_local() and _worker_vivo())


def _soltar_camera() -> None:
    """Fecha a câmera da API. O stop() fica fora do lock: ele espera a thread
    de captura terminar, e segurar o lock nesse tempo travaria as rotas."""
    global _enroll_cam
    with _lock:
        cam, _enroll_cam = _enroll_cam, None
    if cam is not None:
        cam.stop()


def _maybe_close_camera():
    with _lock:
        livre = not _sessions and _enroll_cam is not None
    if livre:
        _soltar_camera()


def _sessao(session_id: str) -> dict:
    """Sessão de cadastro viva, já marcada como usada agora. 404 se não há."""
    with _lock:
        sess = _sessions.get(session_id)
        if sess is not None:
            sess["ultimo_uso"] = time.time()
    if sess is None:
        raise HTTPException(
            404, "Sessão de cadastro inválida ou expirada (expira após "
                 f"{SESSAO_EXPIRA_SEGUNDOS // 60} minutos sem uso). Cancele e "
                 "comece de novo.")
    return sess


def _varrer_cadastro(agora: float | None = None) -> dict:
    """Faxina do cadastro: solta a câmera, expira sessões, limpa pastas órfãs.

    Roda a cada VARREDURA_SEGUNDOS numa thread (ver `_ciclo_de_vida`). É
    função à parte para o teste poder chamá-la direto, sem esperar.

    Regras, da mais importante para a menos:

      1. Câmera local aberta pela API + worker no ar -> solta NA HORA. Era o
         que parava a captura em silêncio: a API abria a webcam com o worker
         parado, o worker voltava e não conseguia mais abrir o dispositivo.
         Ele tenta reabrir a cada poucos segundos, então soltar aqui basta.
      2. Sessão sem uso há SESSAO_EXPIRA_SEGUNDOS expira, levando a pasta
         temporária com as fotos dela.
      3. Sem sessão e sem uso há CAMERA_OCIOSA_SEGUNDOS, a câmera fecha.
         Não fecha logo após "adicionar amostra" de propósito: o preview da
         aba Pessoas continua usando-a enquanto a tela está aberta.
      4. Pastas `enroll/<sessão>` sem sessão viva e mais velhas que o prazo
         são apagadas. Sobram quando a API reinicia no meio de um cadastro, e
         são fotos de rosto sem dono.
    """
    agora = time.time() if agora is None else agora

    motivo = None
    with _lock:
        aberta = _enroll_cam is not None
    if aberta:
        if not _pode_abrir_camera():
            motivo = "worker no ar"
        else:
            with _lock:
                ociosa = (not _sessions
                          and agora - _camera_usada_em > CAMERA_OCIOSA_SEGUNDOS)
            if ociosa:
                motivo = "ociosa"
        if motivo:
            _soltar_camera()

    with _lock:
        expiradas = [sid for sid, s in _sessions.items()
                     if agora - s.get("ultimo_uso", agora) > SESSAO_EXPIRA_SEGUNDOS]
        for sid in expiradas:
            _sessions.pop(sid, None)
        vivas = set(_sessions)
    for sid in expiradas:
        store.remove_dir(f"enroll/{sid}")
    if expiradas:
        _maybe_close_camera()

    orfas = 0
    pasta = SNAP_BASE / "enroll"
    if pasta.is_dir():
        for d in pasta.iterdir():
            if not d.is_dir() or d.name in vivas:
                continue
            try:
                velha = agora - d.stat().st_mtime > SESSAO_EXPIRA_SEGUNDOS
            except OSError:
                continue
            if velha:
                store.remove_dir(f"enroll/{d.name}")
                orfas += 1

    if motivo or expiradas or orfas:
        print(f"[api] faxina do cadastro: câmera "
              f"{'fechada (' + motivo + ')' if motivo else 'sem mudança'}, "
              f"{len(expiradas)} sessão(ões) expirada(s), {orfas} pasta(s) "
              "órfã(s) removida(s)", flush=True)
    return {"camera_fechada": motivo, "expiradas": expiradas, "orfas": orfas}


def _faxina_periodica(parar: threading.Event) -> None:
    while not parar.wait(VARREDURA_SEGUNDOS):
        try:
            _varrer_cadastro()
        except Exception as exc:                           # noqa: BLE001
            # Faxina falhando não pode derrubar a API nem parar de tentar.
            print(f"[api] faxina do cadastro falhou: {type(exc).__name__}: "
                  f"{exc}", flush=True)


def _public_samples(sess: dict):
    return [{"index": s["index"], "snapshot_url": f"/snapshots/{s['snapshot']}",
             "nitidez": s.get("quality"),
             "nitidez_baixa": _nitidez_baixa(s.get("quality"))}
            for s in sess["samples"]]


def _nitidez_baixa(valor):
    """True/False conforme NITIDEZ_MINIMA; None quando não dá para dizer —
    amostra antiga, gravada antes de a nitidez existir, ou aviso desligado."""
    if valor is None or NITIDEZ_MINIMA <= 0:
        return None
    return float(valor) < NITIDEZ_MINIMA


def _aviso_nitidez(nitidez):
    if not _nitidez_baixa(nitidez):
        return None
    return (f"Amostra com nitidez baixa ({nitidez:.0f}; o mínimo é "
            f"{NITIDEZ_MINIMA:.0f}). Rosto borrado ou mal visível atrapalha "
            "mais do que ajuda: o sistema passa a confundir outras pessoas com "
            "esta. Remova a amostra e capture de novo, com a pessoa parada, de "
            "frente e com o rosto iluminado.")


def _avisos_da_amostra(n_rostos: int, nitidez):
    """Avisos de uma captura num texto só, ou None se não há nada a dizer.

    Avisa e NÃO recusa: o número de nitidez depende da câmera, e recusar com
    um limiar mal calibrado travaria o cadastro inteiro. A amostra fica, a
    pessoa vê o aviso e a miniatura marcada, e decide.
    """
    avisos = [a for a in (_aviso_multiplos_rostos(n_rostos),
                          _aviso_nitidez(nitidez)) if a]
    return " ".join(avisos) or None


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
            "nitidez_baixa": _nitidez_baixa(a["quality"]),
            "similaridade_maxima": round(melhor, 3) if parceiro else None,
            "parecida_com": parceiro,
        })
    return saida


# ---- modelos de request --------------------------------------------------- #
class StartReq(BaseModel):
    name: str
    # Matrícula do aluno. Opcional no cadastro para não travar o
    # piloto, mas sem ela o sistema da escola não consegue casar.
    matricula: str = ""


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
    vivo = _worker_vivo()
    st = _ler_status() if vivo else {}

    return {
        "ok": True,
        "enroll_target": ENROLL_TARGET,
        "opencv": cv2.__version__,
        "people": len(db.list_people()),
        # Pelo batimento do worker, não pela idade do live.jpg: o preview só
        # é escrito com `draw_annotations` ligado e quando chega frame.
        "worker_ativo": vivo,
        "worker_live_age_seconds": live_age,
        "worker_mode": st.get("mode"),         # None => worker parado
        "tracks_pending": st.get("pendentes"),  # só no modo captura
        "enroll_session_active": bool(_sessions),
        # "worker" => cadastro usa o frame publicado (funciona com o worker no ar);
        # "camera" => a API abre a câmera (só possível com o worker parado)
        "enroll_source": "worker" if _frame_do_worker() is not None else "camera",
    }


# ---- cadastro interativo -------------------------------------------------- #
@app.post("/enroll/start")
def enroll_start(req: StartReq):
    global _camera_usada_em
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Nome é obrigatório.")
    # Só abre a câmera se o worker não estiver publicando frames E não estiver
    # com o dispositivo. Frame atrasado não quer dizer worker parado: com a
    # câmera travada ele continua vivo e dono da webcam.
    if _frame_do_worker() is None and _pode_abrir_camera():
        _get_camera()
        _camera_usada_em = time.time()
    sid = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[sid] = {"name": name, "samples": [],
                          "matricula": normalizar_matricula(req.matricula),
                          "ultimo_uso": time.time()}
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
    rostos = engine.detect(frame)
    escolhido = engine.best_face(rostos)

    # Distingue o rosto que SERÁ capturado dos demais.
    #
    # Antes todos saíam com a mesma caixa verde e o rótulo "rosto", enquanto a
    # captura usa `best_face` — o MAIOR do quadro. Com duas pessoas em cena,
    # não havia como saber qual seria gravada, e quem cadastra costuma estar
    # mais perto da câmera que a pessoa sendo cadastrada. O resultado foi uma
    # galeria inteira com o rosto errado, descoberta só semanas depois, pela
    # calibração, como 75 confusões do mesmo par.
    for face in rostos:
        eh_alvo = escolhido is not None and face is escolhido
        draw_face(preview, face,
                  "SERA CAPTURADO" if eh_alvo else "ignorado",
                  score=float(face[14]), known=eh_alvo)

    if len(rostos) > 1:
        cv2.putText(preview, f"{len(rostos)} rostos em cena",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", preview)
    if not ok:
        raise HTTPException(500, "Falha ao codificar o preview.")
    resp = Response(content=buf.tobytes(), media_type="image/jpeg")
    # A interface lê isto para avisar antes de a pessoa clicar em capturar.
    resp.headers["X-Rostos"] = str(len(rostos))
    return resp


@app.post("/enroll/capture")
def enroll_capture(req: SessionReq):
    """Captura UMA amostra a partir do frame atual (1 embedding por chamada)."""
    sess = _sessao(req.session_id)
    frame, origem = frame_para_cadastro()
    if frame is None:
        return {"ok": False, "message": _porque_sem_imagem(origem)}

    rostos = engine.detect(frame)
    face = engine.best_face(rostos)
    if face is None:
        return {"ok": False, "message": "Nenhum rosto detectado. Ajuste a posição."}

    vec = engine.embed(frame, face)
    index = len(sess["samples"])
    crop = crop_face(frame, face)
    nitidez = _nitidez(crop)
    snapshot = store.save(crop, f"{sess['name']}_s{index + 1}", subdir=f"enroll/{req.session_id}")
    sess["samples"].append({"index": index, "embedding": vec, "snapshot": snapshot,
                            "quality": nitidez})
    return {"ok": True, "count": len(sess["samples"]), "target": ENROLL_TARGET,
            "samples": _public_samples(sess),
            # Avisa depois do clique porque quem cadastra costuma estar
            # olhando para o botão, não para a imagem: pessoa errada em cena
            # (a primeira barreira é a marcação no preview) e rosto borrado.
            "aviso": _avisos_da_amostra(len(rostos), nitidez)}


def _aviso_multiplos_rostos(n: int):
    if n <= 1:
        return None
    return (f"Havia {n} rostos em cena e foi capturado o MAIOR — normalmente "
            "quem está mais perto da câmera. Confira a miniatura: se for a "
            "pessoa errada, remova a amostra e peça para os outros saírem de "
            "quadro. Galeria com o rosto de outra pessoa produz confusão "
            "sistemática, difícil de diagnosticar depois.")


@app.post("/enroll/sample/delete")
def enroll_delete(req: SampleDelReq):
    sess = _sessao(req.session_id)
    if 0 <= req.index < len(sess["samples"]):
        sess["samples"].pop(req.index)
        for i, s in enumerate(sess["samples"]):  # reindexa
            s["index"] = i
    return {"ok": True, "count": len(sess["samples"]), "samples": _public_samples(sess)}


@app.get("/enroll/status")
def enroll_status(session_id: str):
    sess = _sessao(session_id)
    return {"name": sess["name"], "count": len(sess["samples"]),
            "target": ENROLL_TARGET, "samples": _public_samples(sess)}


@app.post("/enroll/finish")
def enroll_finish(req: SessionReq):
    sess = _sessao(req.session_id)
    if not sess["samples"]:
        raise HTTPException(422, "Capture pelo menos uma amostra antes de concluir.")
    try:
        person_id = db.add_person(sess["name"], sess.get("matricula"))
    except MatriculaDuplicada as exc:
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
        sess = _sessions.pop(req.session_id, None)
    # Só apaga a pasta de uma sessão que EXISTIA. O id vem do cliente, e
    # usá-lo direto no caminho deixava `".."` apagar data/snapshots inteiro.
    # Cancelar sessão desconhecida segue respondendo ok: ela pode ter
    # expirado, e o efeito desejado — não existir mais — já vale.
    if sess is not None:
        store.remove_dir(f"enroll/{req.session_id}")
    _maybe_close_camera()
    return {"ok": True}


# ---- pessoas / eventos / mídia -------------------------------------------- #
@app.get("/people")
def people():
    return db.list_people()


def _apagar_arquivo(base, rel: str) -> bool:
    """Remove um arquivo dentro de `base` E as miniaturas dele.

    As miniaturas moram ao lado da original (`foto.mini128.jpg`) e são cópia
    reduzida do mesmo rosto. Apagar só a original deixava a foto da pessoa
    excluída no disco, em tamanho menor.

    `is_relative_to` compara componentes de caminho; o `startswith` de antes
    aceitava diretório irmão de nome parecido.
    """
    if not rel:
        return False
    alvo = (base / rel).resolve()
    if alvo == base or not alvo.is_relative_to(base):
        return False
    for largura in LARGURAS_MINIATURA:
        try:
            caminho_miniatura(alvo, largura).unlink(missing_ok=True)
        except OSError:
            pass
    if not alvo.is_file():
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
    if pasta != SNAP_BASE and pasta.is_relative_to(SNAP_BASE) and pasta.is_dir():
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
    matricula: str | None = None


@app.patch("/people/{person_id}")
def update_person(person_id: int, req: RenameReq):
    """Altera nome e/ou matrícula. Campo ausente fica como está.

    A matrícula é o identificador que o sistema de gestão da escola usa para
    casar os registros. Casar por nome é frágil: homônimos, acentuação e
    digitação divergente quebram a associação.

    Enviar `matricula: ""` limpa o campo.
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

    if req.matricula is not None:
        try:
            gravado = db.set_person_matricula(person_id, req.matricula)
        except MatriculaDuplicada as exc:
            raise HTTPException(409, str(exc)) from None
        resposta["matricula"] = gravado
        # Sem aviso de formato: o formato da matrícula não foi confirmado, e
        # a versão anterior deste campo avisava com base numa suposição minha
        # sobre o ID INEP que se mostrou errada. Validar o que não se conhece
        # transforma dado válido em dado recusado.

    return resposta


@app.get("/people/by-matricula/{matricula:path}")
def person_by_matricula(matricula: str):
    """Busca aluno pela matrícula — porta de entrada para outro sistema.

    `:path` no parâmetro porque o formato da matrícula não foi confirmado: se
    ela contiver barra (comum em "2026/0142"), a rota precisa aceitar.
    """
    pessoa = db.person_by_matricula(matricula)
    if pessoa is None:
        raise HTTPException(404, "Nenhum aluno cadastrado com esta matrícula.")
    return pessoa


@app.get("/people/{person_id}/samples")
def person_samples(person_id: int):
    """Amostras da pessoa, com nitidez e indicação de redundância."""
    pessoa = db.get_person(person_id)
    if pessoa is None:
        raise HTTPException(404, "Pessoa não encontrada.")
    amostras = db.list_embeddings(person_id)
    return {"person": pessoa, "samples": _redundancia(amostras),
            "sem_foto": sum(1 for a in amostras if not a["snapshot_path"]),
            "nitidez_minima": NITIDEZ_MINIMA or None}


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
    rostos = engine.detect(frame)
    face = engine.best_face(rostos)
    if face is None:
        return {"ok": False, "message": "Nenhum rosto detectado. Ajuste a posição."}

    vec = engine.embed(frame, face)
    crop = crop_face(frame, face)
    nitidez = _nitidez(crop)
    n = db.count_embeddings(person_id) + 1
    caminho = _guardar_amostra(person_id, crop, f"{pessoa['name']}_s{n}")
    eid = db.add_embedding(person_id, vec, caminho, nitidez)
    # A câmera, se foi a API que abriu, NÃO fecha aqui: o preview da aba
    # Pessoas segue usando-a. Quem fecha é a faxina (`_varrer_cadastro`),
    # depois de um minuto sem uso ou na hora em que o worker volta.
    return {"ok": True, "id": eid, "origem": origem,
            "snapshot_url": f"/snapshots/{caminho}",
            "total": db.count_embeddings(person_id),
            "nitidez": nitidez, "nitidez_baixa": _nitidez_baixa(nitidez),
            "aviso": _avisos_da_amostra(len(rostos), nitidez)}


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
    # Mesma rotina da exclusão de pessoa: leva as miniaturas junto.
    _apagar_arquivo(SNAP_BASE, amostra["snapshot_path"])
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
    # Só as pendências DESTE período. As do banco inteiro faziam qualquer dia
    # parecer incompleto enquanto houvesse alguém passando hoje.
    pendentes = db.count_pending_tracks(t0, t1)

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

        linha = {"person_id": pid, "matricula": p.get("matricula"),
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
        # Quem está sem matrícula não pode ser casado pelo sistema da escola.
        # Exposto aqui para o problema aparecer antes de virar falta errada.
        "sem_matricula": sum(1 for p in todos if not p.get("matricula")),
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


class LoginReq(BaseModel):
    token: str


@app.post("/session")
def abrir_sessao(req: LoginReq, request: Request, response: Response):
    """Troca o token por um cookie HttpOnly, para a interface web.

    Por que cookie e não guardar o token no navegador: em `localStorage`
    qualquer script da página lê o valor, e ele fica gravado no disco do
    usuário. Cookie HttpOnly não é acessível por JavaScript, então o token
    existe só no tráfego e na memória do navegador.

    `SameSite=Strict` porque nenhuma outra origem tem motivo para chamar esta
    API com a sessão do usuário. Sem `Secure` por enquanto: o projeto ainda
    roda em HTTP e um cookie Secure simplesmente não seria enviado. Quando o
    HTTPS entrar, esta flag tem que entrar junto — está anotado como pendência.
    """
    if not API_TOKENS:
        # Sem token configurado, quem chega pela rede já foi recusado antes.
        # Aqui só chega cliente local, que não precisa de sessão.
        return {"ok": True, "consumidor": "local",
                "observacao": "Acesso local dispensa token."}

    quem = None
    for nome, valor in API_TOKENS.items():
        if secrets.compare_digest(req.token or "", valor):
            quem = nome
    if quem is None:
        raise HTTPException(401, "Token inválido.")

    response.set_cookie(
        COOKIE_SESSAO, req.token, httponly=True, samesite="strict",
        max_age=12 * 3600, path="/")
    return {"ok": True, "consumidor": quem}


@app.delete("/session")
def fechar_sessao(response: Response):
    response.delete_cookie(COOKIE_SESSAO, path="/")
    return {"ok": True}


@app.get("/session")
def estado_sessao(request: Request):
    """Diz se a interface já está autenticada, sem revelar o token.

    A tela chama isto ao carregar para decidir se mostra o conteúdo ou o pedido
    de token. Precisa funcionar SEM sessão, daí estar em ROTAS_ABERTAS — e por
    isso não pode devolver nada sensível.
    """
    local = cliente_local(request)
    # `autorizar` devolveria "aberta, sem erro" para /session (ela está na
    # lista), então a checagem do cookie é feita aqui de propósito.
    cookie = request.cookies.get(COOKIE_SESSAO, "")
    quem = None
    for nome, valor in API_TOKENS.items():
        if secrets.compare_digest(cookie, valor):
            quem = nome

    if local:
        return {"autenticado": True, "consumidor": "local",
                "token_exigido": False, "motivo": None}
    if not API_TOKENS:
        return {"autenticado": False, "consumidor": None,
                "token_exigido": False,
                "motivo": ("Esta API está acessível pela rede e nenhum token "
                           f"foi configurado. {DICA_TOKEN}")}
    return {"autenticado": quem is not None, "consumidor": quem,
            "token_exigido": True, "motivo": None}


@app.get("/detections")
def detections(person_id: int = Query(
                   0, description="0 = todas; -1 = sem identidade "
                                  "(desconhecidos e marcados como não "
                                  "cadastrados)"),
               dia: str = Query("", description="AAAA-MM-DD; vazio = todos"),
               limit: int = Query(120, ge=1, le=1000)):
    """Detecções das DUAS origens, filtráveis pela identidade CORRIGIDA.

    Difere de `/events`, que lê só a tabela `events` e portanto devolve vazio
    no modo captura — o modo usado na escola.

    O filtro por pessoa usa a identidade efetiva: uma detecção marcada como
    "não é a fulana, é a beltrana" sai da lista da fulana e entra na da
    beltrana. É o que faz a revisão convergir — marcando erro, a lista de cada
    pessoa vai ficando só com o que é dela de fato.
    """
    t0 = t1 = None
    if dia:
        t0, t1 = _janela(dia, "", "")
    linhas = db.deteccoes_de(person_id or None, t0, t1, limit)
    nomes = {p["id"]: p["name"] for p in db.list_people()}

    saida = []
    for l in linhas:
        base = "/snapshots/" if l["fonte"] == "evento" else "/tracks/"
        efetivo = l["efetivo"]
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
            "rotulo": l["rotulo"],
            # Quem a pessoa realmente é, depois da correção humana.
            "pessoa_correta": l["pessoa_correta"],
            "efetivo_id": efetivo,
            "efetivo_nome": nomes.get(efetivo) if efetivo else None,
        })
    return {"total": len(saida), "deteccoes": saida}


class RotuloReq(BaseModel):
    fonte: str
    detection_id: int
    rotulo: str = "errado"
    autor: str = ""
    # Quem é de verdade, com rotulo='errado': o id da pessoa; -1 para "não é
    # ninguém cadastrado"; ausente (null) para "não sei de quem é".
    pessoa_correta: int | None = None


@app.post("/detections/label")
def rotular_deteccao(req: RotuloReq):
    """Marca uma detecção como 'certo' ou 'errado', e de quem ela é.

    Informar `pessoa_correta` é o que transforma o rótulo de "isto está
    errado" em par (detecção → identidade real). Só o segundo serve para
    medir: sem alvo, "errado" diz o que NÃO é e nada mais.

    NÃO altera a presença — decisão deliberada. Rotular mede o acerto do
    reconhecimento; quem esteve na escola é decidido na chamada. Se uma
    revisão de fotos mexesse na frequência do aluno, ninguém confiaria em
    nenhuma das duas.
    """
    try:
        db.set_detection_label(req.fonte, req.detection_id, req.rotulo,
                               req.autor, req.pessoa_correta)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "chave": f"{req.fonte}:{req.detection_id}",
            "rotulo": req.rotulo, "pessoa_correta": req.pessoa_correta,
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
    # Pendências só do dia que está sendo fechado. Contar as do banco inteiro
    # recusava o fechamento durante a aula toda: no modo captura sempre há
    # alguém que acabou de passar esperando o lote.
    t0, t1 = _janela(dia, "", "")
    pendentes = db.count_pending_tracks(t0, t1)
    if pendentes:
        raise HTTPException(
            409, f"Há {pendentes} trilha(s) deste dia aguardando "
                 "reconhecimento, e fechar agora fecharia a chamada "
                 "incompleta. O lote roda sozinho a cada 10 minutos: tente de "
                 "novo em alguns minutos. Para adiantar, no Pi: "
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
def snapshot(path: str, largura: int = Query(
        0, description="0 = original; 128 ou 320 = miniatura")):
    return _servir_imagem(SNAP_BASE, path, "Snapshot não encontrado.", largura)


@app.get("/tracks/{path:path}")
def track_crop(path: str, largura: int = Query(
        0, description="0 = original; 128 ou 320 = miniatura")):
    """Recorte guardado pelo modo captura.

    Os recortes ficam em `tracking.crops_dir`, base DIFERENTE da de snapshots.
    Sem esta rota eles não tinham como ser exibidos, e tanto a chamada quanto
    os relatórios de calibração e de recall montavam URLs sob /snapshots/ que
    devolviam 404 para tudo que viesse do modo captura — justamente o modo
    usado em produção.
    """
    return _servir_imagem(TRACKS_BASE, path, "Recorte não encontrado.", largura)


@app.get("/")
def raiz():
    """Manda quem abre a API no navegador para a interface."""
    return RedirectResponse(f"{PREFIXO_PAINEL}/")


def _montar_painel():
    """Serve a interface web pela própria API, se os arquivos existirem.

    Com isso o painel deixa de precisar de um processo próprio: eram três
    (API, worker, Streamlit) e passam a ser dois. Um processo a menos é uma
    coisa a menos para quebrar e para alguém ter que saber reiniciar — o que
    importa quando quem opera não é quem desenvolveu.

    Silenciosamente ignorado se a pasta não existir, para a API continuar
    subindo em instalação que só usa o Streamlit.
    """
    pasta = project_path("web")
    if not pasta.is_dir():
        return
    from fastapi.staticfiles import StaticFiles
    app.mount(PREFIXO_PAINEL, StaticFiles(directory=str(pasta), html=True),
              name="painel")


def _servir_imagem(base, path: str, erro: str, largura: int = 0):
    full = (base / path).resolve()
    # is_relative_to compara COMPONENTES de caminho. Um `startswith` de string
    # deixaria passar diretório irmão de nome parecido ("snapshots-privado" ao
    # lado de "snapshots"). O `.resolve()` barra o "../../etc/passwd".
    if not full.is_relative_to(base) or not full.is_file():
        raise HTTPException(404, erro)

    if largura:
        if largura not in LARGURAS_MINIATURA:
            raise HTTPException(
                400, f"Largura não suportada. Use uma de: "
                     f"{', '.join(map(str, LARGURAS_MINIATURA))}.")
        mini = gerar_miniatura(full, largura)
        if mini is not None:
            full = mini
        # Falhou a geração? Serve a original. Foto grande demais é melhor que
        # foto nenhuma numa tela de conferência.

    # Cache do navegador: a imagem de uma detecção NUNCA muda — o caminho
    # carrega data e hora. Sem isto, cada recarga da chamada rebaixa tudo, e
    # é justamente a primeira carga que esta otimização existe para encolher.
    return FileResponse(str(full), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/live.jpg")
def live():
    if not LIVE_PATH.exists():
        raise HTTPException(404, "Ainda não há preview ao vivo (o worker está rodando?).")
    return FileResponse(str(LIVE_PATH), media_type="image/jpeg")


# No fim do módulo, depois de todas as rotas: `mount` captura tudo sob o
# prefixo, então registrá-lo antes poderia sombrear rota declarada depois.
_montar_painel()
