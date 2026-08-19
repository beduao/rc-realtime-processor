"""Suíte de regressão do projeto — roda sem câmera e sem os modelos ONNX.

Uso:
    python tests/test_all.py            # tudo (~90s: alguns testes sobem o
                                        #  worker de verdade e esperam)
    python tests/test_all.py tracker    # só os testes cujo nome contém "tracker"
    python tests/test_all.py modo       # os de troca de modo

Não usa pytest de propósito: assim roda no Pi sem instalar nada além do que o
worker já precisa. Os testes que exigem os modelos ou a câmera real são pulados
com aviso, em vez de falhar.

Cobre principalmente as partes onde um erro seria SILENCIOSO: alinhamento dos
recortes, contagem de frames, votação, escrita atômica e retenção.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

TESTES = []
TMP = None


def teste(fn):
    TESTES.append(fn)
    return fn


# --------------------------------------------------------------------------- #
# apoio
# --------------------------------------------------------------------------- #
def face_em(x, y, s=100):
    """Linha do YuNet (15 valores) para um rosto na posição dada."""
    f = np.zeros(15, np.float32)
    f[:4] = [x, y, s, s]
    f[4:14] = [x + .30 * s, y + .32 * s, x + .70 * s, y + .32 * s,
               x + .50 * s, y + .52 * s, x + .35 * s, y + .75 * s,
               x + .65 * s, y + .75 * s]
    f[14] = 0.99
    return f


def video_sintetico(caminho, frames=60, tam=(640, 480)):
    vw = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"MJPG"), 15, tam)
    for i in range(frames):
        img = np.full((tam[1], tam[0], 3), 30, np.uint8)
        cv2.circle(img, (60 + i * 8, tam[1] // 2), 55, (200, 200, 200), -1)
        vw.write(img)
    vw.release()


def escrever_config(**over):
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml")))
    cfg["storage"]["db_path"] = os.path.join(TMP, "t.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(TMP, "snaps")
    cfg["storage"]["live_path"] = os.path.join(TMP, "live.jpg")
    cfg["tracking"]["crops_dir"] = os.path.join(TMP, "tracks")
    for k, v in over.items():
        secao, _, chave = k.partition(".")
        cfg[secao][chave] = v
    caminho = os.path.join(TMP, "config.yaml")
    yaml.safe_dump(cfg, open(caminho, "w"))
    os.environ["FACIAL_CONFIG"] = caminho
    import core.config as cc
    cc._cache = None
    return cfg


class EngineFalsa:
    """Substitui a FaceEngine: detecta um rosto que atravessa a cena."""
    cosine_threshold = 0.5

    def __init__(self, cfg=None, embed_proibido=False):
        self.n = 0
        self.embed_proibido = embed_proibido

    def detect(self, img):
        self.n += 1
        pos = self.n % 20
        return [face_em(40 + pos * 22, 190)] if pos < 12 else []

    def embed(self, img, face):
        if self.embed_proibido:
            raise AssertionError("este modo não deveria chamar embed()")
        v = np.ones(128, np.float32)
        v[0] = 2.0
        return v / np.linalg.norm(v)

    def match(self, vec, matriz, ids):
        if matriz is None or not len(ids):
            return None, 0.0
        s = matriz @ vec
        i = int(np.argmax(s))
        return ids[i], float(s[i])

    @staticmethod
    def best_face(faces):
        return max(faces, key=lambda f: float(f[2]) * float(f[3])) if faces else None


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@teste
def config_live_path_absoluto_e_fallback():
    from core.config import live_image_path, load_config
    escrever_config()
    cfg = load_config()
    assert str(live_image_path(cfg)) == os.path.join(TMP, "live.jpg")
    sem_storage = {k: v for k, v in cfg.items() if k != "storage"}
    assert live_image_path(sem_storage).name == "live.jpg"
    return "caminho absoluto respeitado; ausência de storage cai no padrão"


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
@teste
def camera_escolhe_backend_por_tipo_de_fonte():
    from core.camera import _parse_source, Camera
    casos = [("rtsp://a@1.2.3.4/x", cv2.CAP_FFMPEG, False),
             ("/tmp/v.avi", cv2.CAP_FFMPEG, False),
             (0, cv2.CAP_V4L2, True), ("0", cv2.CAP_V4L2, True),
             ("/dev/video0", cv2.CAP_V4L2, True)]
    for fonte, backend, local in casos:
        _, be = _parse_source(fonte)
        assert be == backend, (fonte, be)
        assert Camera(fonte).is_local_device == local, fonte
    return f"{len(casos)} fontes classificadas corretamente"


@teste
def camera_read_new_nao_repete_nem_copia_a_esmo():
    from core.camera import Camera
    v = os.path.join(TMP, "v1.avi")
    video_sintetico(v, frames=40)
    cam = Camera(v, 0.2).start()
    try:
        assert cam.read_wait(timeout=10) is not None, "nenhum frame lido"
        seq, novos, vazios = 0, 0, 0
        fim = time.time() + 2
        while time.time() < fim:
            f, seq = cam.read_new(seq)
            novos += f is not None
            vazios += f is None
            time.sleep(0.002)
        assert novos > 0 and vazios > novos, (novos, vazios)
        return f"{novos} frames novos, {vazios} chamadas sem novidade evitaram cópia"
    finally:
        cam.stop()


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
@teste
def tracker_iou_e_nitidez():
    from core.tracker import iou, sharpness
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [50, 50, 10, 10]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 10, 10]) - 5 / 15) < 1e-6
    nitido = np.zeros((80, 80), np.uint8)
    nitido[::4] = 255
    borrado = cv2.GaussianBlur(nitido, (9, 9), 5)
    assert sharpness(nitido) > sharpness(borrado) * 3
    return "IoU correto; nitidez separa nítido de borrado"


@teste
def tracker_alinhamento_do_recorte_equivale_ao_frame_inteiro():
    """A garantia mais importante do modo captura.

    Se o alinhamento a partir do recorte divergir do alinhamento a partir do
    frame, os embeddings da fase 2 seriam diferentes dos do modo em tempo real
    e o limiar calibrado não valeria — uma falha silenciosa.
    """
    from core.tracker import crop_with_landmarks
    GABARITO = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

    def alinhar(img, face):
        pts = np.asarray(face[4:14], np.float32).reshape(5, 2)
        M, _ = cv2.estimateAffinePartial2D(pts, GABARITO, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)

    frame = np.random.RandomState(7).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for rotulo, x, y in [("centro", 260, 180), ("borda esquerda", 6, 180),
                         ("topo", 260, 5), ("canto inferior", 520, 360)]:
        face = face_em(x, y)
        rec, local = crop_with_landmarks(frame, face, margin=0.4)
        dif = np.abs(alinhar(frame, face).astype(int) - alinhar(rec, local).astype(int)).mean()
        assert dif < 1.0, (rotulo, dif)

    # margem pequena DEVE degradar — confirma que 0.4 não é número mágico
    face = face_em(260, 180)
    rec, local = crop_with_landmarks(frame, face, margin=0.05)
    pior = np.abs(alinhar(frame, face).astype(int) - alinhar(rec, local).astype(int)).mean()
    assert pior > 5, pior
    return "idêntico ao pixel nas 4 posições; margem 0.05 degrada (diferença "
    f"{pior:.1f}), confirmando a necessidade da margem"


@teste
def tracker_agrupa_pessoas_e_descarta_deteccao_solitaria():
    from core.tracker import FaceTracker
    tr = FaceTracker(iou_threshold=0.3, max_missing_frames=3,
                     crops_per_track=3, min_track_frames=2)
    img = np.random.RandomState(1).randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for i in range(12):
        faces = [face_em(20 + i * 25, 150)]
        if i >= 4:
            faces.append(face_em(500 - (i - 4) * 20, 200))
        assert tr.update(faces, img, i, now=100 + i * 0.1) == []
    assert len(tr.ativos) == 2, tr.ativos
    encerradas = []
    for i in range(12, 18):
        encerradas += tr.update([], img, i, now=102 + i * 0.1)
    assert len(encerradas) == 2, len(encerradas)
    assert all(t.frames >= 2 and t.crops for t in encerradas)

    solo = FaceTracker(max_missing_frames=1, min_track_frames=2)
    solo.update([face_em(10, 10)], img, 0, now=1.0)
    assert solo.update([], img, 5, now=2.0) == [] and solo.descartadas == 1
    return "2 pessoas viraram 2 trilhas; detecção de 1 frame descartada"


@teste
def tracker_guarda_os_recortes_mais_nitidos():
    from core.tracker import FaceTracker
    tr = FaceTracker(max_missing_frames=1, crops_per_track=2, min_track_frames=1)
    for i in range(6):
        img = np.zeros((480, 640, 3), np.uint8)
        padrao = np.zeros((300, 300, 3), np.uint8)
        padrao[::3] = 255
        if i != 3:                       # só o frame 3 é nítido
            padrao = cv2.GaussianBlur(padrao, (15, 15), 8)
        img[100:400, 100:400] = padrao
        tr.update([face_em(150, 150)], img, i, now=i * 0.1)
    crops = tr.flush()[0].crops
    q = [c[0] for c in crops]
    assert q == sorted(q, reverse=True) and q[0] > q[-1] * 100, q
    return f"ordenados por qualidade; melhor {q[0]:.0f} vs pior {q[-1]:.1f}"


# --------------------------------------------------------------------------- #
# banco
# --------------------------------------------------------------------------- #
@teste
def banco_trilhas_presenca_e_retencao():
    escrever_config()
    from core.database import Database
    db = Database(os.path.join(TMP, "b.db"))
    ana, bruno = db.add_person("Ana"), db.add_person("Bruno")
    db.add_person("Carla")                      # não aparece

    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    crop = [{"path": "x.jpg", "quality": 1.0, "face": "[]"}]
    for pid, nome, h in [(ana, "Ana", 7.5), (ana, "Ana", 7.6), (bruno, "Bruno", 7.8)]:
        tid = db.add_track(meia + h * 3600, meia + h * 3600 + 1, 9, crop)
        db.resolve_track(tid, pid, nome, 0.8, "[]")
    db.add_track(meia + 8 * 3600, meia + 8 * 3600 + 1, 5, crop)   # pendente

    pres = db.attendance(meia, meia + 86400)
    assert len(pres) == 2, pres
    ana_linha = next(p for p in pres if p["person_id"] == ana)
    assert ana_linha["passagens"] == 2, ana_linha
    assert db.count_tracks_by_status() == {"processado": 3, "pendente": 1}
    assert db.reset_processed_tracks() == 3
    assert db.count_tracks_by_status().get("pendente") == 4

    # retenção de eventos
    db.add_event(ana, "Ana", 0.9, "a.jpg", 1)
    antigo = time.time() - 90 * 86400
    with db._connect() as con:                  # noqa: SLF001
        con.execute("INSERT INTO events(person_id,name,score,ts,snapshot_path,is_known)"
                    " VALUES(?,?,?,?,?,?)", (ana, "Ana", 0.9, antigo, "b.jpg", 1))
    assert db.count_events() == 2
    assert db.purge_events_before(time.time() - 30 * 86400) == 1
    assert db.count_events() == 1
    return "presença agrupa por pessoa (Ana: 2 passagens, 1 linha); retenção e VACUUM ok"


# --------------------------------------------------------------------------- #
# votação (fase 2)
# --------------------------------------------------------------------------- #
@teste
def votacao_um_recorte_ruim_e_voto_vencido():
    from pathlib import Path
    from scripts.recognize_batch import processar_trilha

    class Galeria:
        ids = [7, 9]
        names = {7: "Ana", 9: "Bruno"}
        matrix = None

    def engine(votos):
        seq = list(votos)

        class E:
            def embed(s, img, face):
                return np.zeros(128, np.float32)

            def match(s, v, m, ids):
                return seq.pop(0)
        return E()

    # recortes reais em disco para o imread funcionar
    base = Path(TMP) / "vot"
    base.mkdir(parents=True, exist_ok=True)
    crops = []
    for i in range(3):
        nome = f"c{i}.jpg"
        cv2.imwrite(str(base / nome), np.full((80, 80, 3), 120, np.uint8))
        crops.append({"path": nome, "quality": 1.0, "face": json.dumps([0.0] * 15)})

    casos = [
        ([(7, .80), (9, .55), (7, .76)], 7, "Ana", "2 votos Ana vencem 1 Bruno"),
        ([(9, .52), (7, .88), (7, .85)], 7, "Ana", "recorte ruim isolado não decide"),
        ([(7, .80), (9, .79), (None, .10)], None, "Desconhecido", "sem maioria"),
        ([(7, .31), (7, .28), (9, .37)], None, "Desconhecido", "todos sob o limiar"),
    ]
    for votos, pid_esp, nome_esp, rotulo in casos:
        pid, nome, score, _, status = processar_trilha(
            engine(votos), Galeria(), base, {"id": 1}, crops, 0.5, 2)
        assert pid == pid_esp and nome == nome_esp, (rotulo, pid, nome)
        assert status == "processado"
    # média dos vencedores, não o máximo
    pid, _, score, _, _ = processar_trilha(
        engine([(7, .80), (9, .55), (7, .76)]), Galeria(), base, {"id": 1}, crops, 0.5, 2)
    assert abs(score - 0.78) < 1e-6, score
    return f"{len(casos)} cenários de votação corretos; score = média dos vencedores"


# --------------------------------------------------------------------------- #
# worker — os dois modos
# --------------------------------------------------------------------------- #
def _rodar_worker(segundos=8, **over):
    """Roda o worker em thread e o PARA antes de devolver.

    Deixar a thread viva contaminava os testes seguintes: o worker antigo
    continuava publicando no arquivo de status, e o teste seguinte lia o estado
    do worker errado.
    """
    import importlib
    import core.face_engine as fe
    v = os.path.join(TMP, "w.avi")
    if not os.path.exists(v):
        video_sintetico(v, frames=80)
    escrever_config(**{**over, "camera.rtsp_url": v,
                       "camera.reconnect_delay_seconds": 0.2})
    proibir = over.get("worker.mode") == "captura"
    fe.FaceEngine = lambda cfg: EngineFalsa(cfg, embed_proibido=proibir)
    w = importlib.import_module("worker")
    importlib.reload(w)
    w.STATS_EVERY_SECONDS = 3.0
    w.LIVE_WRITE_SECONDS = 0.2
    w.PARAR.clear()
    # argv=[] explícito: sem isso o argparse do worker leria o filtro passado
    # para a suíte (ex.: `python tests/test_all.py modo`) e abortaria.
    th = threading.Thread(target=lambda: w.main([]), daemon=True)
    th.start()
    try:
        time.sleep(segundos)
    finally:
        w.PARAR.set()
        th.join(timeout=15)
        w.PARAR.clear()


@teste
def worker_modo_realtime_gera_eventos_e_preview_atomico():
    from core.database import Database
    escrever_config()
    db = Database(os.path.join(TMP, "t.db"))
    pid = db.add_person("Maria")
    db.add_embedding(pid, EngineFalsa().embed(None, None))
    _rodar_worker(8, **{"worker.mode": "realtime", "worker.min_interval_seconds": 0.05,
                        "worker.process_every_n_frames": 1,
                        "worker.event_cooldown_seconds": 0.3})
    ev = db.list_events(limit=100)
    assert ev, "nenhum evento gerado"
    assert all(e["name"] == "Maria" and e["is_known"] == 1 for e in ev)
    snap = os.path.join(TMP, "snaps", ev[0]["snapshot_path"])
    assert cv2.imread(snap) is not None, snap
    live = os.path.join(TMP, "live.jpg")
    assert cv2.imread(live) is not None, "preview ilegível (escrita não atômica?)"
    assert not os.path.exists(live + ".part"), "sobrou arquivo temporário"
    ts = sorted(e["ts"] for e in ev)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert all(g >= 0.29 for g in gaps), gaps
    return f"{len(ev)} eventos, cooldown respeitado, preview legível e sem .part"


@teste
def worker_modo_captura_rastreia_sem_reconhecer():
    from core.database import Database
    escrever_config()
    _rodar_worker(9, **{"worker.mode": "captura", "worker.draw_annotations": False,
                        "tracking.max_missing_frames": 2})
    db = Database(os.path.join(TMP, "t.db"))
    trilhas = db.pending_tracks()
    assert trilhas, "nenhuma trilha capturada"
    crops = db.track_crops(trilhas[0]["id"])
    assert 1 <= len(crops) <= 3, len(crops)
    for c in crops:
        p = os.path.join(TMP, "tracks", c["path"])
        assert cv2.imread(p) is not None, p
        assert len(json.loads(c["face"])) == 15, "landmarks não salvos"
    # a EngineFalsa levanta AssertionError se embed() for chamado; se houve
    # trilha e nenhum erro, a captura de fato não reconheceu
    return (f"{len(trilhas)} trilhas, {len(crops)} recortes com landmarks; "
            "embed() nunca chamado")


@teste
def worker_sigterm_salva_trilhas_em_andamento():
    """`systemctl stop/restart` não pode descartar quem está em cena.

    Prova: `max_missing_frames` altíssimo faz com que NENHUMA trilha termine
    naturalmente. Se algo aparecer no banco, só pode ter vindo do flush
    disparado pelo SIGTERM.
    """
    import signal as sig
    import subprocess
    from core.database import Database

    db_path = os.path.join(TMP, "sigterm.db")
    v = os.path.join(TMP, "sig.avi")
    video_sintetico(v, frames=80)
    cfg_path = os.path.join(TMP, "cfg_sigterm.yaml")
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml")))
    cfg["camera"]["rtsp_url"] = v
    cfg["camera"]["reconnect_delay_seconds"] = 0.2
    cfg["storage"]["db_path"] = db_path
    cfg["storage"]["live_path"] = os.path.join(TMP, "sig-live.jpg")
    cfg["tracking"]["crops_dir"] = os.path.join(TMP, "sig_tracks")
    cfg["tracking"]["max_missing_frames"] = 10 ** 6      # nada encerra sozinho
    cfg["tracking"]["min_track_frames"] = 1
    cfg["worker"]["mode"] = "captura"
    cfg["worker"]["draw_annotations"] = False
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    # engine falsa injetada por sitecustomize, já que roda em outro processo
    shim = os.path.join(TMP, "shim")
    os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, "sitecustomize.py"), "w") as fh:
        fh.write(
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {RAIZ!r})\n"
            "import core.face_engine as fe\n"
            "def _f(x, y, s=100):\n"
            "    import numpy as np\n"
            "    f = np.zeros(15, np.float32); f[:4] = [x, y, s, s]\n"
            "    f[4:14] = [x+30,y+32, x+70,y+32, x+50,y+52, x+35,y+75, x+65,y+75]\n"
            "    f[14] = 0.99; return f\n"
            "class E:\n"
            "    cosine_threshold = 0.5\n"
            "    def __init__(self, cfg=None): self.n = 0\n"
            "    def detect(self, img):\n"
            "        self.n += 1; return [_f(200, 190)]\n"
            "    def embed(self, i, f): raise AssertionError('captura não reconhece')\n"
            "    def match(self, *a): return None, 0.0\n"
            "fe.FaceEngine = E\n"
        )

    env = dict(os.environ, FACIAL_CONFIG=cfg_path,
               PYTHONPATH=shim + os.pathsep + RAIZ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "worker.py"], cwd=RAIZ, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        time.sleep(6)
        assert proc.poll() is None, f"worker morreu sozinho:\n{proc.stdout.read()}"

        db = Database(db_path)
        assert not db.pending_tracks(), \
            "alguma trilha terminou sozinha — o teste não provaria nada"

        proc.send_signal(sig.SIGTERM)
        saida = proc.communicate(timeout=25)[0]
        assert proc.returncode == 0, f"saída suja ({proc.returncode}):\n{saida}"

        trilhas = db.pending_tracks()
        assert trilhas, f"trilha em cena foi PERDIDA no SIGTERM:\n{saida}"
        assert "encerrando com calma" in saida, saida
        assert db.track_crops(trilhas[0]["id"]), "trilha salva sem recortes"
        return (f"{len(trilhas)} trilha(s) em cena preservada(s); "
                f"processo saiu com código 0")
    finally:
        if proc.poll() is None:
            proc.kill()


def _shim_engine(destino, proibir_embed=False):
    """sitecustomize que injeta a EngineFalsa num worker rodando em subprocesso."""
    os.makedirs(destino, exist_ok=True)
    guarda = "        raise AssertionError('captura nao reconhece')\n" if proibir_embed else ""
    with open(os.path.join(destino, "sitecustomize.py"), "w") as fh:
        fh.write(
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {RAIZ!r})\n"
            "import core.face_engine as fe\n"
            "def _f(x, y, s=100):\n"
            "    f = np.zeros(15, np.float32); f[:4] = [x, y, s, s]\n"
            "    f[4:14] = [x+30,y+32, x+70,y+32, x+50,y+52, x+35,y+75, x+65,y+75]\n"
            "    f[14] = 0.99; return f\n"
            "class E:\n"
            "    cosine_threshold = 0.5\n"
            "    def __init__(self, cfg=None): self.n = 0\n"
            "    def detect(self, img):\n"
            # rosto que ENTRA e SAI de cena, para as trilhas encerrarem sozinhas
            "        self.n += 1; pos = self.n % 20\n"
            "        return [_f(40 + pos * 22, 190)] if pos < 12 else []\n"
            "    def embed(self, i, f):\n"
            + guarda +
            "        v = np.ones(128, np.float32); v[0] = 2.0\n"
            "        return v / np.linalg.norm(v)\n"
            "    def match(self, vec, m, ids):\n"
            "        return (ids[0], 1.0) if m is not None and len(ids) else (None, 0.0)\n"
            "fe.FaceEngine = E\n"
        )


def _cfg_arquivo(nome, **over):
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml")))
    v = os.path.join(TMP, "modo.avi")
    if not os.path.exists(v):
        video_sintetico(v, frames=80)
    # Cada teste com sua pasta: o status é publicado ao lado do live.jpg, e
    # compartilhar o caminho fazia um teste ler o status do worker de outro.
    pasta = os.path.join(TMP, nome)
    os.makedirs(pasta, exist_ok=True)
    cfg["camera"]["rtsp_url"] = v
    cfg["camera"]["reconnect_delay_seconds"] = 0.2
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(pasta, "snaps")
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    cfg["tracking"]["crops_dir"] = os.path.join(pasta, "tracks")
    cfg["tracking"]["max_missing_frames"] = 2
    cfg["worker"]["draw_annotations"] = False
    cfg["worker"]["mode_check_seconds"] = 2   # deixa o teste rápido
    for k, val in over.items():
        sec, _, chave = k.partition(".")
        cfg[sec][chave] = val
    caminho = os.path.join(TMP, f"{nome}.yaml")
    yaml.safe_dump(cfg, open(caminho, "w"))
    return caminho


def _subir_worker(cfg_path, extra_args=(), proibir_embed=False):
    import subprocess
    shim = os.path.join(TMP, "shim_modo")
    _shim_engine(shim, proibir_embed)
    env = dict(os.environ, FACIAL_CONFIG=cfg_path,
               PYTHONPATH=shim + os.pathsep + RAIZ, PYTHONUNBUFFERED="1")
    return subprocess.Popen([sys.executable, "worker.py", *extra_args],
                            cwd=RAIZ, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


@teste
def modo_cli_fixa_e_ignora_o_config():
    """--mode tem precedência e desliga a troca automática."""
    import signal as sig
    cfg_path = _cfg_arquivo("cli", **{"worker.mode": "captura"})
    proc = _subir_worker(cfg_path, ["--mode", "realtime"])
    status_file = os.path.join(TMP, "cli", "facial-status.json")
    try:
        time.sleep(4)
        assert proc.poll() is None, proc.stdout.read()
        status = json.load(open(status_file))
        assert status["mode"] == "realtime", status
        assert status["fixo"] is True, status

        # config diz "captura", mas o CLI mandou: não pode trocar
        time.sleep(6)
        status = json.load(open(status_file))
        assert status["mode"] == "realtime", "trocou apesar de --mode"
        proc.send_signal(sig.SIGTERM)
        saida = proc.communicate(timeout=25)[0]
        assert "modo inicial: realtime (de --mode)" in saida, saida
        return "config pedia captura; --mode realtime prevaleceu e não trocou"
    finally:
        if proc.poll() is None:
            proc.kill()


@teste
def modo_troca_a_quente_sem_reiniciar():
    """Editar worker.mode no config troca o laço com o processo em execução."""
    import signal as sig
    from core.database import Database
    cfg_path = _cfg_arquivo("hot", **{"worker.mode": "captura",
                                      "tracking.max_missing_frames": 2})
    proc = _subir_worker(cfg_path)
    status_file = os.path.join(TMP, "hot", "facial-status.json")
    try:
        time.sleep(5)
        assert proc.poll() is None, proc.stdout.read()
        st = json.load(open(status_file))
        assert st["mode"] == "captura" and st["fixo"] is False, st

        db = Database(os.path.join(TMP, "hot", "dados.db"))
        trilhas_antes = len(db.pending_tracks())
        assert trilhas_antes > 0, "captura não gravou nada antes da troca"

        # cadastra alguém para o realtime ter o que reconhecer
        pid = db.add_person("Maria")
        db.add_embedding(pid, EngineFalsa().embed(None, None))

        # troca pelo mesmo caminho que a pessoa usaria
        cfg = yaml.safe_load(open(cfg_path))
        cfg["worker"]["mode"] = "realtime"
        cfg["worker"]["min_interval_seconds"] = 0.05
        cfg["worker"]["process_every_n_frames"] = 1
        cfg["worker"]["event_cooldown_seconds"] = 0.3
        yaml.safe_dump(cfg, open(cfg_path, "w"))

        time.sleep(7)
        st = json.load(open(status_file))
        assert st["mode"] == "realtime", f"não trocou: {st}"
        assert db.list_events(limit=5), "realtime não gerou eventos após a troca"
        assert len(db.pending_tracks()) >= trilhas_antes, \
            "trilhas pendentes sumiram na transição"

        proc.send_signal(sig.SIGTERM)
        saida = proc.communicate(timeout=25)[0]
        assert "captura -> realtime" in saida, saida
        assert proc.returncode == 0
        return ("captura -> realtime sem reiniciar; trilhas preservadas e "
                "eventos passaram a ser gerados")
    finally:
        if proc.poll() is None:
            proc.kill()


@teste
def modo_invalido_no_config_nao_derruba_o_worker():
    import worker as w
    cfg_path = _cfg_arquivo("inval", **{"worker.mode": "realtime"})
    os.environ["FACIAL_CONFIG"] = cfg_path
    assert w.modo_do_config("realtime") == "realtime"

    cfg = yaml.safe_load(open(cfg_path))
    cfg["worker"]["mode"] = "turbo"
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    assert w.modo_do_config("captura") == "captura", "valor inválido mudou o modo"

    with open(cfg_path, "w") as fh:      # YAML corrompido
        fh.write("worker: [isto: nao: e: valido\n")
    assert w.modo_do_config("captura") == "captura", "YAML quebrado mudou o modo"
    return "valor inválido e YAML corrompido mantêm o modo atual"


@teste
def set_mode_edita_preservando_comentarios():
    import subprocess
    cfg_path = _cfg_arquivo("setm", **{"worker.mode": "realtime"})
    # o set_mode preserva o arquivo; usamos o exemplo comentado como origem
    origem = open(os.path.join(RAIZ, "config.pi.example.yaml"), encoding="utf-8").read()
    open(cfg_path, "w", encoding="utf-8").write(origem)
    comentarios_antes = origem.count("#")

    env = dict(os.environ, FACIAL_CONFIG=cfg_path, PYTHONPATH=RAIZ)
    r = subprocess.run([sys.executable, "scripts/set_mode.py", "captura"],
                       cwd=RAIZ, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    depois = open(cfg_path, encoding="utf-8").read()
    assert yaml.safe_load(depois)["worker"]["mode"] == "captura", depois[:200]
    assert depois.count("#") == comentarios_antes, "perdeu comentários do config"
    assert "facial-batch.timer" in r.stdout, r.stdout

    r2 = subprocess.run([sys.executable, "scripts/set_mode.py"],
                        cwd=RAIZ, env=env, capture_output=True, text=True)
    assert "worker.mode = captura" in r2.stdout, r2.stdout
    return f"trocou para captura preservando os {comentarios_antes} comentários"


@teste
def cadastro_usa_o_frame_do_worker_quando_ele_esta_no_ar():
    """Com webcam USB o dispositivo é exclusivo: a API não pode abri-lo.

    Aqui a "câmera" é um caminho inexistente — se a API tentasse abri-la,
    falharia. Ela só funciona se estiver consumindo o frame publicado.
    """
    import importlib
    import core.face_engine as fe

    escrever_config(**{"camera.rtsp_url": "/dev/video-inexistente"})
    fe.FaceEngine = lambda cfg: EngineFalsa(cfg)
    api = importlib.import_module("api")
    importlib.reload(api)

    # worker parado: sem frame publicado -> cai no caminho da câmera (que falha)
    if api.FRAME_PATH.exists():
        api.FRAME_PATH.unlink()
    assert api._frame_do_worker() is None
    h = api.health()
    assert h["enroll_source"] == "camera", h

    # worker publicando: a API usa o frame dele
    quadro = np.full((480, 640, 3), 90, np.uint8)
    cv2.imwrite(str(api.FRAME_PATH), quadro)
    assert api._frame_do_worker() is not None
    assert api.health()["enroll_source"] == "worker"

    frame, origem = api.frame_para_cadastro()
    assert origem == "worker" and frame.shape == (480, 640, 3), (origem, frame)

    resp = api.enroll_start(api.StartReq(name="Teste USB"))
    sid = resp["session_id"]
    cap = api.enroll_capture(api.SessionReq(session_id=sid))
    assert cap["ok"] and cap["count"] == 1, cap
    fim = api.enroll_finish(api.SessionReq(session_id=sid))
    assert fim["captured"] == 1 and fim["name"] == "Teste USB", fim

    # frame velho é ignorado: melhor não cadastrar do que cadastrar imagem antiga
    os.utime(api.FRAME_PATH, (0, 0))
    assert api._frame_do_worker() is None, "frame velho deveria ser descartado"
    return ("preview e captura funcionaram sem abrir a câmera; "
            "frame velho é descartado")


def _api_cliente(nome="api"):
    """Sobe a API com engine falsa e frame vindo de 'worker'. Devolve (client, cfg)."""
    import importlib
    import core.config as cc
    import core.face_engine as fe

    pasta = os.path.join(TMP, nome)
    os.makedirs(pasta, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml")))
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(pasta, "snaps")
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    caminho = os.path.join(pasta, "cfg.yaml")
    yaml.safe_dump(cfg, open(caminho, "w"))
    os.environ["FACIAL_CONFIG"] = caminho
    cc._cache = None

    class E:
        cosine_threshold = 0.5

        def __init__(self, c=None):
            self.k = 0

        def detect(self, img):
            return [face_em(200, 150, 120)]

        def best_face(self, fs):
            return fs[0] if fs else None

        def embed(self, img, f):
            self.k += 1
            v = np.zeros(128, np.float32)
            v[0] = 1.0
            v[self.k % 40 + 1] = 0.05 * (self.k % 3)
            return v / np.linalg.norm(v)

        def match(self, v, m, ids):
            if m is None or not len(ids):
                return None, 0.0
            s = m @ v
            i = int(np.argmax(s))
            return ids[i], float(s[i])

    fe.FaceEngine = E
    # o frame limpo do worker fica ao lado do live.jpg
    cv2.imwrite(os.path.join(pasta, "facial-frame.jpg"),
                np.random.RandomState(3).randint(0, 255, (480, 640, 3), dtype=np.uint8))

    api = importlib.import_module("api")
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app), pasta


@teste
def api_gestao_de_amostras_por_pessoa():
    """Ver, adicionar e excluir amostras, renomear, e cascata na exclusão."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("amostras")

    # cadastro completo -> cada embedding com sua foto
    sid = c.post("/enroll/start", json={"name": "Maria"}).json()["session_id"]
    for _ in range(4):
        assert c.post("/enroll/capture", json={"session_id": sid}).json()["ok"]
    fin = c.post("/enroll/finish", json={"session_id": sid}).json()
    pid = fin["id"]
    assert fin["captured"] == 4
    assert not os.path.isdir(os.path.join(pasta, "snaps", "enroll", sid)), \
        "pasta temporária da sessão não foi limpa"

    det = c.get(f"/people/{pid}/samples").json()
    assert len(det["samples"]) == 4 and det["sem_foto"] == 0
    for a in det["samples"]:
        assert a["snapshot_url"] and "amostras/" in a["snapshot_url"], a
        assert a["quality"] is not None, "nitidez não calculada"
        assert a["similaridade_maxima"] is not None, "redundância não calculada"
        assert c.get(a["snapshot_url"]).status_code == 200, a["snapshot_url"]

    # reforçar cadastro depois
    add = c.post(f"/people/{pid}/samples").json()
    assert add["ok"] and add["total"] == 5, add

    # excluir uma amostra some com o registro E com a foto
    alvo = det["samples"][0]
    arq = os.path.join(pasta, "snaps", alvo["snapshot_url"].split("/snapshots/")[1])
    assert os.path.exists(arq)
    r = c.delete(f"/embeddings/{alvo['id']}")
    assert r.status_code == 200 and r.json()["restantes"] == 4, r.text
    assert not os.path.exists(arq), "foto da amostra excluída ficou no disco"

    # renomear, com validações
    assert c.patch(f"/people/{pid}", json={"name": "Maria Silva"}).status_code == 200
    assert c.patch(f"/people/{pid}", json={"name": "  "}).status_code == 400
    assert c.patch("/people/99999", json={"name": "X"}).status_code == 404

    # excluir a pessoa remove a pasta de amostras dela
    dir_pessoa = os.path.join(pasta, "snaps", "amostras", str(pid))
    assert os.path.isdir(dir_pessoa)
    r = c.delete(f"/people/{pid}")
    assert r.status_code == 200 and r.json()["fotos_removidas"] >= 1, r.text
    assert not os.path.isdir(dir_pessoa), "pasta de amostras ficou no disco"
    return ("cadastro com foto por amostra, nitidez, redundância, reforço, "
            "exclusão individual e cascata")


@teste
def api_trava_a_exclusao_da_ultima_amostra():
    """Pessoa sem embedding nunca seria reconhecida, mas seguiria na lista."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, _ = _api_cliente("ultima")
    sid = c.post("/enroll/start", json={"name": "Solo"}).json()["session_id"]
    assert c.post("/enroll/capture", json={"session_id": sid}).json()["ok"]
    pid = c.post("/enroll/finish", json={"session_id": sid}).json()["id"]

    unica = c.get(f"/people/{pid}/samples").json()["samples"][0]["id"]
    r = c.delete(f"/embeddings/{unica}")
    assert r.status_code == 409, f"deveria recusar com 409, veio {r.status_code}"
    assert "última" in r.json()["detail"].lower(), r.json()
    assert c.get(f"/people/{pid}/samples").json()["samples"], "apagou mesmo assim"
    return "409 ao tentar remover a única amostra; cadastro preservado"


@teste
def api_migra_banco_antigo_sem_perder_cadastro():
    """Banco criado antes das colunas novas precisa continuar funcionando."""
    import sqlite3
    pasta = os.path.join(TMP, "migracao")
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, "velho.db")

    con = sqlite3.connect(caminho)
    con.executescript("""
        CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER,
            name TEXT NOT NULL, score REAL NOT NULL, ts REAL NOT NULL,
            snapshot_path TEXT, is_known INTEGER NOT NULL);
    """)
    v = np.ones(128, np.float32)
    v /= np.linalg.norm(v)
    con.execute("INSERT INTO people(name,created_at) VALUES('Antiga',?)", (time.time(),))
    con.execute("INSERT INTO embeddings(person_id,vec) VALUES(1,?)", (v.tobytes(),))
    con.commit()
    con.close()

    from core.database import Database
    db = Database(caminho)                      # dispara a migração
    pessoas = db.list_people()
    assert len(pessoas) == 1 and pessoas[0]["embeddings"] == 1, pessoas
    amostras = db.list_embeddings(pessoas[0]["id"])
    assert len(amostras) == 1 and amostras[0]["snapshot_path"] is None
    assert np.allclose(amostras[0]["vec"], v), "vetor antigo corrompido"
    # e o banco migrado aceita os campos novos
    eid = db.add_embedding(pessoas[0]["id"], v, "amostras/1/x.jpg", 123.4)
    nova = [a for a in db.list_embeddings(pessoas[0]["id"]) if a["id"] == eid][0]
    assert nova["snapshot_path"] == "amostras/1/x.jpg" and nova["quality"] == 123.4
    return "cadastro e vetor preservados; colunas novas adicionadas sem perda"


@teste
def monitor_identifica_o_worker_sem_falso_positivo():
    """Casar a substring 'worker.py' na linha de comando pega o shell errado."""
    from scripts.monitor import _e_o_worker, explicar_throttled

    # o próprio processo de teste NÃO é o worker
    assert not _e_o_worker(os.getpid()), "confundiu o processo de teste com o worker"
    assert not _e_o_worker(1), "PID 1 não pode passar"
    assert not _e_o_worker(999999), "PID inexistente deve dar False"

    # bits do vcgencmd traduzidos
    assert explicar_throttled(0) == []
    agora = explicar_throttled(0x1)
    assert agora and "AGORA" in agora[0], agora
    hist = explicar_throttled(0x50000)
    assert len(hist) == 2 and all("desde o boot" in h or "houve" in h for h in hist), hist
    return "PID do teste, init e inexistente rejeitados; flags traduzidas"


# --------------------------------------------------------------------------- #
# find_camera / calibração — lógica pura
# --------------------------------------------------------------------------- #
@teste
def find_camera_classifica_sem_afirmar_demais():
    from scripts.find_camera import classify, MAC_RE
    assert "TEM RTSP" in classify({"ports": [554, 80], "ip": "x", "mac": ""})
    assert "não respondeu" in classify({"ports": [], "ip": "x", "mac": ""})
    v = classify({"ports": [80], "ip": "x", "mac": ""})
    assert "roteador" in v and "Mibo" not in v, v
    assert MAC_RE.match("98-2a-0a-91-f5-ff") and MAC_RE.match("aa:bb:cc:dd:ee:ff")
    assert not MAC_RE.match("192.168.15.19")
    return "roteador não é chamado de câmera; MAC validado por formato"


@teste
def calibracao_nao_inventa_limiar_quando_ha_sobreposicao():
    import io
    from contextlib import redirect_stdout
    from scripts.calibrate_threshold import sugerir
    ev = [{"id": 1, "score": 0.42}, {"id": 2, "score": 0.85}, {"id": 3, "score": 0.88}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        sugerir(ev, {"1": "errado", "2": "certo", "3": "certo"}, 0.363)
    saida = buf.getvalue()
    assert "SEPARADAS" in saida and "0.635" in saida, saida

    ev2 = [{"id": 1, "score": 0.50}, {"id": 2, "score": 0.80}, {"id": 3, "score": 0.60}]
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        sugerir(ev2, {"1": "certo", "2": "errado", "3": "certo"}, 0.363)
    s2 = buf2.getvalue()
    assert "SOBREP" in s2 and "Limiar sugerido" not in s2, s2
    return "separadas -> sugere 0.635; sobrepostas -> explica sem inventar número"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main() -> int:
    global TMP
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    selecionados = [t for t in TESTES if filtro in t.__name__]
    if not selecionados:
        print(f"Nenhum teste corresponde a {filtro!r}.")
        return 1

    TMP = tempfile.mkdtemp(prefix="facial-teste-")
    print(f"opencv {cv2.__version__} | numpy {np.__version__} | tmp {TMP}\n")
    falhas = []
    try:
        for fn in selecionados:
            nome = fn.__name__.replace("_", " ")
            try:
                detalhe = fn() or ""
                print(f"  [ok] {nome}")
                if detalhe:
                    print(f"       {detalhe}")
            except AssertionError as exc:
                falhas.append((fn.__name__, f"asserção: {exc}"))
                print(f"  [XX] {nome}\n       asserção: {exc}")
            except Exception as exc:                    # noqa: BLE001
                falhas.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
                print(f"  [XX] {nome}\n       {type(exc).__name__}: {exc}")
    finally:
        os.environ.pop("FACIAL_CONFIG", None)
        shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{len(selecionados) - len(falhas)}/{len(selecionados)} passaram")
    for nome, motivo in falhas:
        print(f"  FALHOU {nome}: {motivo}")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
