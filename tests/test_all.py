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
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["storage"]["db_path"] = os.path.join(TMP, "t.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(TMP, "snaps")
    cfg["storage"]["live_path"] = os.path.join(TMP, "live.jpg")
    cfg["tracking"]["crops_dir"] = os.path.join(TMP, "tracks")
    for k, v in over.items():
        secao, _, chave = k.partition(".")
        cfg[secao][chave] = v
    caminho = os.path.join(TMP, "config.yaml")
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)
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
    from core.camera import _backend_local, _parse_source, Camera
    # O backend de webcam por ÍNDICE depende da plataforma (V4L2 no Linux,
    # DirectShow no Windows), então vem de _backend_local() em vez de estar
    # fixo aqui. Fixar V4L2 fazia este teste falhar no Windows com um
    # "(0, 700)" enigmático — 700 é o CAP_DSHOW.
    local = _backend_local()
    casos = [("rtsp://a@1.2.3.4/x", cv2.CAP_FFMPEG, False),
             ("/tmp/v.avi", cv2.CAP_FFMPEG, False),
             (0, local, True), ("0", local, True),
             ("/dev/video0", cv2.CAP_V4L2, True)]
    for fonte, backend, e_local in casos:
        _, be = _parse_source(fonte)
        assert be == backend, (fonte, be, backend)
        assert Camera(fonte).is_local_device == e_local, fonte
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

    if sys.platform.startswith("win"):
        # No Windows, send_signal(SIGTERM) vira TerminateProcess: mata o
        # processo na hora, sem rodar handler nenhum, e o código de saída é 1.
        # O encerramento gracioso é logicamente impossível ali — e é
        # dispensável, porque quem para o worker com SIGTERM é o systemd, que
        # só existe no Pi. Pular é a resposta correta; "corrigir" seria
        # afrouxar a asserção e o teste deixaria de provar o que importa.
        return "PULADO no Windows: SIGTERM não é entregável (é TerminateProcess)"

    db_path = os.path.join(TMP, "sigterm.db")
    v = os.path.join(TMP, "sig.avi")
    video_sintetico(v, frames=80)
    cfg_path = os.path.join(TMP, "cfg_sigterm.yaml")
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["camera"]["rtsp_url"] = v
    cfg["camera"]["reconnect_delay_seconds"] = 0.2
    cfg["storage"]["db_path"] = db_path
    cfg["storage"]["live_path"] = os.path.join(TMP, "sig-live.jpg")
    cfg["tracking"]["crops_dir"] = os.path.join(TMP, "sig_tracks")
    cfg["tracking"]["max_missing_frames"] = 10 ** 6      # nada encerra sozinho
    cfg["tracking"]["min_track_frames"] = 1
    cfg["worker"]["mode"] = "captura"
    cfg["worker"]["draw_annotations"] = False
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                   allow_unicode=True)

    # engine falsa injetada por sitecustomize, já que roda em outro processo
    shim = os.path.join(TMP, "shim")
    os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, "sitecustomize.py"), "w",
              encoding="utf-8") as fh:
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
    with open(os.path.join(destino, "sitecustomize.py"), "w",
              encoding="utf-8") as fh:
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
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
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
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)
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
        status = json.load(open(status_file, encoding="utf-8"))
        assert status["mode"] == "realtime", status
        assert status["fixo"] is True, status

        # config diz "captura", mas o CLI mandou: não pode trocar
        time.sleep(6)
        status = json.load(open(status_file, encoding="utf-8"))
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
        st = json.load(open(status_file, encoding="utf-8"))
        assert st["mode"] == "captura" and st["fixo"] is False, st

        db = Database(os.path.join(TMP, "hot", "dados.db"))
        trilhas_antes = len(db.pending_tracks())
        assert trilhas_antes > 0, "captura não gravou nada antes da troca"

        # cadastra alguém para o realtime ter o que reconhecer
        pid = db.add_person("Maria")
        db.add_embedding(pid, EngineFalsa().embed(None, None))

        # troca pelo mesmo caminho que a pessoa usaria
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        cfg["worker"]["mode"] = "realtime"
        cfg["worker"]["min_interval_seconds"] = 0.05
        cfg["worker"]["process_every_n_frames"] = 1
        cfg["worker"]["event_cooldown_seconds"] = 0.3
        yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                   allow_unicode=True)

        time.sleep(7)
        st = json.load(open(status_file, encoding="utf-8"))
        assert st["mode"] == "realtime", f"não trocou: {st}"
        assert db.list_events(limit=5), "realtime não gerou eventos após a troca"
        assert len(db.pending_tracks()) >= trilhas_antes, \
            "trilhas pendentes sumiram na transição"

        # A troca a quente já foi provada acima, pelo estado publicado e pelo
        # banco. O que vem agora é só a confirmação no log — e ela depende de
        # encerramento gracioso, que no Windows não existe: send_signal(SIGTERM)
        # vira TerminateProcess, mata sem rodar handler e a saída pendente se
        # perde. Asserção sobre log post-mortem, ali, testaria o sistema
        # operacional, não o worker.
        proc.send_signal(sig.SIGTERM)
        saida = proc.communicate(timeout=25)[0]
        if sys.platform.startswith("win"):
            return ("captura -> realtime sem reiniciar; trilhas preservadas "
                    "(log de encerramento não conferido: SIGTERM no Windows "
                    "não é gracioso)")
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

    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["worker"]["mode"] = "turbo"
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                   allow_unicode=True)
    assert w.modo_do_config("captura") == "captura", "valor inválido mudou o modo"

    with open(cfg_path, "w", encoding="utf-8") as fh:   # YAML corrompido
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


def _api_cliente(nome="api", host="127.0.0.1", **over):
    """Sobe a API com engine falsa e frame vindo de 'worker'. Devolve (client, cfg).

    `host` é o IP que a API verá como origem da requisição. O padrão é
    127.0.0.1 porque a autenticação por exposição libera a máquina local, e
    estes testes exercitam funcionalidade, não credencial. Os testes de
    autenticação passam um IP remoto de propósito.

    `over` aceita chaves "secao.chave" para mexer no config (ex.: api.token).
    """
    import importlib
    import core.config as cc
    import core.face_engine as fe

    pasta = os.path.join(TMP, nome)
    os.makedirs(pasta, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(pasta, "snaps")
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    for k, v in over.items():
        sec, _, chave = k.partition(".")
        cfg.setdefault(sec, {})[chave] = v
    caminho = os.path.join(pasta, "cfg.yaml")
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)
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
    return TestClient(api.app, client=(host, 45678)), pasta


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
def api_presenca_une_as_duas_origens_e_avisa_quando_incompleta():
    """O endpoint de chamada, com os pontos que um integrador precisa confiar."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("presenca")
    from core.database import Database
    db = Database(os.path.join(pasta, "dados.db"))

    ana = db.add_person("Ana")
    bruno = db.add_person("Bruno")
    db.add_person("Carla")                       # cadastrada, não aparece

    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    crop = [{"path": "x.jpg", "quality": 1.0, "face": "[]"}]

    # Ana vem do modo CAPTURA (trilha processada), 2 passagens
    for h in (7.5, 7.7):
        tid = db.add_track(meia + h * 3600, meia + h * 3600 + 1, 9, crop)
        db.resolve_track(tid, ana, "Ana", 0.81, "[]")
    # Bruno vem do modo REALTIME (evento) — a versão antiga do attendance()
    # ignorava esta origem e devolveria chamada sem ele
    db.add_event(bruno, "Bruno", 0.74, "b.jpg", 1)

    r = c.get("/attendance")
    assert r.status_code == 200, r.text
    d = r.json()
    nomes = {p["nome"] for p in d["presentes"]}
    assert nomes == {"Ana", "Bruno"}, f"origens não unidas: {nomes}"
    ana_linha = next(p for p in d["presentes"] if p["nome"] == "Ana")
    assert ana_linha["passagens"] == 2 and "captura" in ana_linha["fontes"]
    bruno_linha = next(p for p in d["presentes"] if p["nome"] == "Bruno")
    assert "realtime" in bruno_linha["fontes"], bruno_linha
    assert [n["nome"] for n in d["ausentes"]] == ["Carla"]
    # quem não foi detectado aparece com origem explícita, não como "ausente" seco
    assert d["ausentes"][0]["origem"] == "nao_identificado"
    assert d["ausentes"][0]["detectado_pelo_sistema"] is False
    assert d["completo"] is True and d["trilhas_pendentes"] == 0
    assert d["total_cadastrados"] == 3 and d["total_presentes"] == 2
    assert d["conferida"] is False, "sem fechamento não pode constar como conferida"
    assert d["aviso"], "chamada não conferida deve avisar"

    # trilha pendente => chamada INCOMPLETA (o campo mais importante)
    db.add_track(meia + 8 * 3600, meia + 8 * 3600 + 1, 5, crop)
    d2 = c.get("/attendance").json()
    assert d2["completo"] is False and d2["trilhas_pendentes"] == 1, d2

    # janela de horário: Ana entrou 7:30, filtro a partir das 7:45 a exclui
    d3 = c.get("/attendance", params={"inicio": "07:45"}).json()
    assert "Ana" not in {p["nome"] for p in d3["presentes"]}, d3["presentes"]

    # validações de entrada
    assert c.get("/attendance", params={"dia": "20/08/2026"}).status_code == 400
    assert c.get("/attendance", params={"inicio": "25:00"}).status_code == 400
    assert c.get("/attendance", params={"inicio": "10:00", "fim": "09:00"}).status_code == 400

    # dia sem movimento devolve estrutura válida e vazia, não erro
    d4 = c.get("/attendance", params={"dia": "2020-01-01"}).json()
    assert d4["total_presentes"] == 0 and len(d4["ausentes"]) == 3
    assert d4["periodo"]["dia"] == "2020-01-01"
    return ("une captura+realtime, sinaliza chamada incompleta, filtra por "
            "horário e recusa entrada inválida")


@teste
def api_chamada_correcao_manual_preserva_o_automatico():
    """A correção não pode sobrescrever o que o reconhecimento detectou.

    É dessa diferença que sai a medida de acerto do sistema: presente marcado
    à mão = falso negativo; ausente marcado à mão = falso positivo.
    """
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("chamada")
    from core.database import Database
    db = Database(os.path.join(pasta, "dados.db"))

    ana = db.add_person("Ana")
    bruno = db.add_person("Bruno")
    carla = db.add_person("Carla")

    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    dia = time.strftime("%Y-%m-%d", time.localtime(meia))
    crop = [{"path": "x.jpg", "quality": 1.0, "face": "[]"}]

    # Ana e Bruno detectados; Carla não
    for pid, nome in ((ana, "Ana"), (bruno, "Bruno")):
        tid = db.add_track(meia + 7.5 * 3600, meia + 7.5 * 3600 + 1, 9, crop)
        db.resolve_track(tid, pid, nome, 0.8, "[]")

    d = c.get("/attendance", params={"dia": dia}).json()
    assert d["total_presentes"] == 2 and len(d["ausentes"]) == 1
    assert d["conferida"] is False and d["aviso"], d
    assert {p["origem"] for p in d["presentes"]} == {"automatico"}

    # Carla veio mas o sistema não pegou -> falso negativo
    r = c.post("/attendance/override", json={
        "dia": dia, "person_id": carla, "presente": True,
        "motivo": "chegou antes da câmera", "autor": "Operador"})
    assert r.status_code == 200, r.text
    # Bruno foi identificado por engano -> falso positivo
    assert c.post("/attendance/override", json={
        "dia": dia, "person_id": bruno, "presente": False,
        "autor": "Operador"}).status_code == 200

    d = c.get("/attendance", params={"dia": dia}).json()
    nomes_presentes = {p["nome"] for p in d["presentes"]}
    assert nomes_presentes == {"Ana", "Carla"}, nomes_presentes
    assert {p["nome"] for p in d["ausentes"]} == {"Bruno"}
    assert d["correcoes"] == {"marcados_presentes": 1, "marcados_ausentes": 1,
                              "total": 2}, d["correcoes"]

    carla_linha = next(p for p in d["presentes"] if p["nome"] == "Carla")
    assert carla_linha["origem"] == "manual_presente"
    assert carla_linha["detectado_pelo_sistema"] is False
    assert carla_linha["correcao"]["motivo"] == "chegou antes da câmera"
    assert carla_linha["correcao"]["autor"] == "Operador"

    bruno_linha = d["ausentes"][0]
    assert bruno_linha["origem"] == "manual_ausente"
    # o detectado original SOBREVIVE — é a evidência que mede o erro
    assert bruno_linha["detectado_pelo_sistema"] is True
    assert bruno_linha["melhor_score"] == 0.8, "score automático foi perdido"

    # desfazer volta ao automático
    assert c.delete("/attendance/override",
                    params={"dia": dia, "person_id": bruno}).json()["removidas"] == 1
    d = c.get("/attendance", params={"dia": dia}).json()
    assert "Bruno" in {p["nome"] for p in d["presentes"]}
    assert next(p for p in d["presentes"]
                if p["nome"] == "Bruno")["origem"] == "automatico"
    return ("correção prevalece na chamada, detecção original preservada, "
            "e os dois tipos de erro contados separadamente")


@teste
def api_chamada_fechamento_trava_edicao_e_exige_lote_vazio():
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("fechar")
    from core.database import Database
    db = Database(os.path.join(pasta, "dados.db"))
    ana = db.add_person("Ana")
    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    dia = time.strftime("%Y-%m-%d", time.localtime(meia))
    crop = [{"path": "x.jpg", "quality": 1.0, "face": "[]"}]

    # trilha PENDENTE deve impedir o fechamento: fecharia chamada incompleta
    db.add_track(meia + 7 * 3600, meia + 7 * 3600 + 1, 5, crop)
    r = c.post("/attendance/close", json={"dia": dia, "autor": "Operador"})
    assert r.status_code == 409, f"deixou fechar com pendência: {r.status_code}"
    assert "recognize_batch" in r.json()["detail"], r.json()

    # resolvida a pendência, fecha
    pend = db.pending_tracks()[0]
    db.resolve_track(pend["id"], ana, "Ana", 0.9, "[]")
    r = c.post("/attendance/close", json={"dia": dia, "autor": "Operador"})
    assert r.status_code == 200, r.text

    d = c.get("/attendance", params={"dia": dia}).json()
    assert d["conferida"] is True and d["fechamento"]["autor"] == "Operador"
    assert d["aviso"] is None, "chamada conferida não deveria avisar"

    # fechada não aceita mais correção
    assert c.post("/attendance/override", json={
        "dia": dia, "person_id": ana, "presente": False}).status_code == 409
    assert c.delete("/attendance/override",
                    params={"dia": dia, "person_id": ana}).status_code == 409

    # reabrir libera
    assert c.delete("/attendance/close", params={"dia": dia}).json()["reaberta"] == 1
    assert c.post("/attendance/override", json={
        "dia": dia, "person_id": ana, "presente": False}).status_code == 200

    # validações de entrada
    assert c.post("/attendance/override", json={
        "dia": "20/08/2026", "person_id": ana, "presente": True}).status_code == 400
    assert c.post("/attendance/override", json={
        "dia": dia, "person_id": 99999, "presente": True}).status_code == 404
    return ("pendência bloqueia o fechamento; fechada recusa edição; "
            "reabrir libera; entrada inválida recusada")


@teste
def inep_normalizacao_e_unicidade():
    """O ID INEP é a chave de casamento com o sistema da escola."""
    from core.database import (Database, InepDuplicado, inep_suspeito,
                               normalizar_inep)

    # normalização: mesma pessoa não pode entrar duas vezes por causa de pontuação
    assert normalizar_inep(" 123.456.789-012 ") == "123456789012"
    assert normalizar_inep("") is None and normalizar_inep(None) is None
    assert normalizar_inep("abc") is None
    # zero à esquerda tem que sobreviver — daí TEXT e não INTEGER
    assert normalizar_inep("000123456789") == "000123456789"

    # formato improvável avisa, mas não bloqueia
    assert inep_suspeito("123456789012") is None
    assert "12" in (inep_suspeito("123") or "")

    pasta = os.path.join(TMP, "inep")
    os.makedirs(pasta, exist_ok=True)
    db = Database(os.path.join(pasta, "dados.db"))

    ana = db.add_person("Ana", "000123456789")
    assert db.get_person(ana)["inep_id"] == "000123456789"
    achado = db.person_by_inep("000.123.456-789")     # busca normaliza também
    assert achado and achado["id"] == ana

    # duplicidade recusada, tanto no cadastro novo...
    try:
        db.add_person("Outra", "000123456789")
        raise AssertionError("aceitou ID INEP duplicado no cadastro")
    except InepDuplicado as exc:
        assert exc.person_id == ana and "Ana" in str(exc)
    # ...quanto na edição
    bruno = db.add_person("Bruno")
    assert db.get_person(bruno)["inep_id"] is None
    try:
        db.set_person_inep(bruno, "000123456789")
        raise AssertionError("aceitou duplicado na edição")
    except InepDuplicado:
        pass

    # atribuir o próprio valor de novo é permitido (não é conflito consigo)
    assert db.set_person_inep(ana, "000123456789") == "000123456789"
    # limpar é permitido, e libera o valor
    assert db.set_person_inep(ana, "") is None
    assert db.set_person_inep(bruno, "000123456789") == "000123456789"
    # vários sem ID coexistem — o índice único é PARCIAL, senão o segundo NULL
    # bateria com o primeiro. Ana ficou sem ID no passo acima, mais Carla e Diana.
    db.add_person("Carla")
    db.add_person("Diana")
    sem_id = [p["name"] for p in db.list_people() if not p["inep_id"]]
    assert sorted(sem_id) == ["Ana", "Carla", "Diana"], sem_id
    return ("pontuação normalizada, zero à esquerda preservado, duplicidade "
            "recusada, vários sem ID coexistem")


@teste
def api_inep_na_chamada_e_busca():
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("inepapi")
    from core.database import Database
    db = Database(os.path.join(pasta, "dados.db"))
    ana = db.add_person("Ana", "000123456789")
    db.add_person("Bruno")            # sem ID

    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    dia = time.strftime("%Y-%m-%d", time.localtime(meia))
    tid = db.add_track(meia + 7.5 * 3600, meia + 7.5 * 3600 + 1, 9,
                       [{"path": "x.jpg", "quality": 1.0, "face": "[]"}])
    db.resolve_track(tid, ana, "Ana", 0.8, "[]")

    d = c.get("/attendance", params={"dia": dia}).json()
    assert d["sem_inep"] == 1, d["sem_inep"]
    presente = d["presentes"][0]
    assert presente["inep_id"] == "000123456789", presente
    assert d["ausentes"][0]["inep_id"] is None

    # busca pelo ID — porta de entrada do outro sistema
    r = c.get("/people/by-inep/000.123.456-789")
    assert r.status_code == 200 and r.json()["name"] == "Ana", r.text
    assert c.get("/people/by-inep/999999999999").status_code == 404

    # edição pela API, com aviso de formato e recusa de duplicidade
    bruno = next(p["id"] for p in db.list_people() if p["name"] == "Bruno")
    r = c.patch(f"/people/{bruno}", json={"inep_id": "123"})
    assert r.status_code == 200 and "aviso" in r.json(), r.json()
    assert c.patch(f"/people/{bruno}",
                   json={"inep_id": "000123456789"}).status_code == 409
    # nome sozinho não mexe no ID
    r = c.patch(f"/people/{bruno}", json={"name": "Bruno Lima"})
    assert r.status_code == 200 and "inep_id" not in r.json()
    assert db.get_person(bruno)["inep_id"] == "123", "nome mexeu no ID INEP"
    return "ID INEP na chamada, contagem de faltantes, busca e validações"


@teste
def api_token_opcional_protege_sem_trancar_o_health():
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"
    import importlib
    import core.config as cc

    pasta = os.path.join(TMP, "token")
    os.makedirs(pasta, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = os.path.join(pasta, "snaps")
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    cfg["api"]["token"] = "segredo-de-teste"
    caminho = os.path.join(pasta, "cfg.yaml")
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)
    os.environ["FACIAL_CONFIG"] = caminho
    cc._cache = None

    import core.face_engine as fe
    fe.FaceEngine = lambda c=None: type("E", (), {"cosine_threshold": 0.5})()
    api = importlib.import_module("api")
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)

    assert c.get("/health").status_code == 200, "/health não pode exigir token"
    assert c.get("/people").status_code == 401, "rota protegida aceitou sem token"
    assert c.get("/attendance").status_code == 401
    assert c.get("/people", headers={"X-API-Token": "errado"}).status_code == 401
    assert c.get("/people", headers={"X-API-Token": "segredo-de-teste"}).status_code == 200
    assert c.get("/people",
                 headers={"Authorization": "Bearer segredo-de-teste"}).status_code == 200
    return "health aberto; token exigido via X-API-Token e Bearer; token errado = 401"


def _cenario_completo(nome):
    """Pessoa com amostra, evento e trilha — cada um com sua imagem em disco."""
    from core.database import Database
    pasta = os.path.join(TMP, nome)
    snaps = os.path.join(pasta, "snaps")
    tracks = os.path.join(pasta, "tracks")
    os.makedirs(snaps, exist_ok=True)
    os.makedirs(tracks, exist_ok=True)
    db = Database(os.path.join(pasta, "dados.db"))

    def img(base, rel):
        caminho = os.path.join(base, rel)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        cv2.imwrite(caminho, np.full((40, 40, 3), 128, np.uint8))
        return rel

    pid = db.add_person("Alvo")
    v = np.ones(128, np.float32)
    v /= np.linalg.norm(v)
    db.add_embedding(pid, v, img(snaps, "amostras/1/a.jpg"), 100.0)
    db.add_event(pid, "Alvo", 0.9, img(snaps, "20260820/ev.jpg"), 1)
    tid = db.add_track(time.time(), time.time() + 1, 8,
                       [{"path": img(tracks, "20260820/t.jpg"),
                         "quality": 1.0, "face": "[]"}])
    db.resolve_track(tid, pid, "Alvo", 0.88, "[]")
    return db, pasta, snaps, tracks, pid


@teste
def exclusao_apaga_historico_e_imagens_das_duas_bases():
    """Pedido de exclusão precisa alcançar eventos, trilhas e as duas pastas."""
    db, pasta, snaps, tracks, pid = _cenario_completo("excl")

    assert os.path.exists(os.path.join(snaps, "20260820/ev.jpg"))
    assert os.path.exists(os.path.join(tracks, "20260820/t.jpg"))

    arquivos = db.delete_person(pid)
    assert len(arquivos["snapshots"]) == 2, arquivos      # amostra + evento
    assert len(arquivos["tracks"]) == 1, arquivos

    # o banco não conhece o disco: quem chama apaga. Aqui simulamos a API.
    for rel in arquivos["snapshots"]:
        os.remove(os.path.join(snaps, rel))
    for rel in arquivos["tracks"]:
        os.remove(os.path.join(tracks, rel))

    assert not db.list_people()
    assert db.count_events() == 0, "evento sobreviveu à exclusão"
    assert db.count_tracks_by_status() == {}, "trilha sobreviveu à exclusão"
    assert not os.path.exists(os.path.join(snaps, "20260820/ev.jpg"))
    assert not os.path.exists(os.path.join(tracks, "20260820/t.jpg"))
    return "eventos, trilhas e imagens das duas bases removidos"


@teste
def exclusao_anonimizada_preserva_contagem_sem_identificar():
    from core.database import ANONIMO
    db, pasta, snaps, tracks, pid = _cenario_completo("anon")

    arquivos = db.delete_person(pid, anonimizar=True)
    assert len(arquivos["snapshots"]) == 2 and len(arquivos["tracks"]) == 1

    assert not db.list_people(), "a pessoa deveria sair da lista"
    # as passagens ficam, mas sem identificar quem
    eventos = db.list_events(limit=10)
    assert len(eventos) == 1, eventos
    assert eventos[0]["person_id"] is None and eventos[0]["name"] == ANONIMO
    assert eventos[0]["snapshot_path"] is None, "referência de foto deveria sumir"
    assert eventos[0]["is_known"] == 0
    # a trilha permanece para estatística, sem dono e sem recortes
    assert db.count_tracks_by_status().get("processado") == 1
    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    assert db.attendance(meia, meia + 86400) == [], \
        "anonimizada não pode mais aparecer na chamada"
    return "pessoa sai da chamada; contagem de passagens sobrevive sem identificação"


@teste
def retencao_alcanca_a_pasta_de_trilhas():
    """A limpeza varria só data/snapshots — os recortes ficavam para sempre."""
    import subprocess
    from core.database import Database

    pasta = os.path.join(TMP, "retencao")
    snaps = os.path.join(pasta, "snaps")
    tracks = os.path.join(pasta, "tracks")
    for base in (snaps, tracks):
        for dia in ("20200101", "20991231"):
            d = os.path.join(base, dia)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "x.jpg"), np.full((40, 40, 3), 90, np.uint8))

    db = Database(os.path.join(pasta, "dados.db"))
    antigo = time.time() - 400 * 86400
    tid = db.add_track(antigo, antigo + 1, 5,
                       [{"path": "20200101/x.jpg", "quality": 1.0, "face": "[]"}])
    db.resolve_track(tid, None, "Desconhecido", 0.2, "[]")

    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = snaps
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    cfg["tracking"]["crops_dir"] = tracks
    caminho = os.path.join(pasta, "cfg.yaml")
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)

    env = dict(os.environ, FACIAL_CONFIG=caminho, PYTHONPATH=RAIZ)
    r = subprocess.run([sys.executable, "scripts/cleanup_snapshots.py",
                        "--days", "30", "--dias-trilhas", "7"],
                       cwd=RAIZ, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    assert not os.path.isdir(os.path.join(tracks, "20200101")), \
        f"pasta de trilhas antiga NÃO foi removida:\n{r.stdout}"
    assert not os.path.isdir(os.path.join(snaps, "20200101"))
    assert os.path.isdir(os.path.join(tracks, "20991231")), "removeu dia futuro"
    assert os.path.isdir(os.path.join(snaps, "20991231"))
    assert not db.track_crops(tid), "referência de recorte ficou no banco"
    return "recortes antigos removidos do disco e do banco; dias recentes intactos"


@teste
def retencao_nao_apaga_recortes_de_trilha_pendente():
    """Se o lote parou, apagar recorte pendente jogaria fora dado não usado."""
    import subprocess
    from core.database import Database

    pasta = os.path.join(TMP, "pendente")
    snaps = os.path.join(pasta, "snaps")
    tracks = os.path.join(pasta, "tracks")
    d = os.path.join(tracks, "20200101")
    os.makedirs(d, exist_ok=True)
    os.makedirs(snaps, exist_ok=True)
    cv2.imwrite(os.path.join(d, "x.jpg"), np.full((40, 40, 3), 90, np.uint8))

    db = Database(os.path.join(pasta, "dados.db"))
    antigo = time.time() - 400 * 86400
    db.add_track(antigo, antigo + 1, 5,                   # fica PENDENTE
                 [{"path": "20200101/x.jpg", "quality": 1.0, "face": "[]"}])
    assert db.pending_tracks_before(time.time() - 7 * 86400) == 1

    cfg = yaml.safe_load(open(os.path.join(RAIZ, "config.pi.example.yaml"),
                                    encoding="utf-8"))
    cfg["storage"]["db_path"] = os.path.join(pasta, "dados.db")
    cfg["storage"]["snapshots_dir"] = snaps
    cfg["storage"]["live_path"] = os.path.join(pasta, "live.jpg")
    cfg["tracking"]["crops_dir"] = tracks
    caminho = os.path.join(pasta, "cfg.yaml")
    yaml.safe_dump(cfg, open(caminho, "w", encoding="utf-8"),
                   allow_unicode=True)

    env = dict(os.environ, FACIAL_CONFIG=caminho, PYTHONPATH=RAIZ)
    r = subprocess.run([sys.executable, "scripts/cleanup_snapshots.py",
                        "--dias-trilhas", "7"],
                       cwd=RAIZ, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.isdir(d), "apagou recorte de trilha PENDENTE"
    assert "PENDENTES" in r.stdout and "facial-batch" in r.stdout, r.stdout
    return "recorte pendente preservado e o motivo explicado na saída"


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


def _cenario_calibracao(nome):
    """Detecções nas duas origens, no dia de hoje."""
    from core.database import Database
    pasta = os.path.join(TMP, nome)
    os.makedirs(pasta, exist_ok=True)
    db = Database(os.path.join(pasta, "dados.db"))
    ana = db.add_person("Ana")
    bruno = db.add_person("Bruno")

    hoje = time.localtime()
    meia = time.mktime((hoje.tm_year, hoje.tm_mon, hoje.tm_mday, 0, 0, 0, 0, 0, -1))
    dia = time.strftime("%Y-%m-%d", time.localtime(meia))
    crop = [{"path": "20260826/c.jpg", "quality": 1.0, "face": "[]"}]

    # Ana pelo modo realtime (events), com dois scores
    for sc in (0.88, 0.71):
        db.add_event(ana, "Ana", sc, "20260826/e.jpg", 1)
    # Bruno pelo modo captura (tracks) — invisível para a versão anterior
    tid = db.add_track(meia + 7.5 * 3600, meia + 7.5 * 3600 + 1, 9, crop)
    db.resolve_track(tid, bruno, "Bruno", 0.41, "[]")
    return db, pasta, dia, ana, bruno


@teste
def calibracao_le_as_duas_origens():
    """A versão anterior lia só `events` e ficava cega no modo captura."""
    from core.database import Database  # noqa: F401
    db, pasta, dia, ana, bruno = _cenario_calibracao("calfonte")

    det = db.list_detections(limit=100)
    fontes = {d["fonte"] for d in det}
    assert fontes == {"evento", "trilha"}, f"faltou uma origem: {fontes}"
    assert len(det) == 3, det
    trilha = next(d for d in det if d["fonte"] == "trilha")
    assert trilha["name"] == "Bruno" and trilha["is_known"] == 1
    assert trilha["snapshot_path"] == "20260826/c.jpg", "recorte não veio"

    # a chave precisa distinguir origens: id repete entre as tabelas
    from scripts.calibrate_threshold import chave
    assert len({chave(d) for d in det}) == 3, "chaves colidiram entre origens"
    return "events + tracks unidos, com recorte e chave sem colisão"


@teste
def calibracao_so_confia_em_chamada_fechada():
    """Chamada aberta sem correção significa 'ninguém olhou', não 'está certo'."""
    from scripts.calibrate_threshold import chave, rotulos_da_chamada
    db, pasta, dia, ana, bruno = _cenario_calibracao("calfech")
    det = db.list_detections(limit=100)

    # chamada AINDA NÃO conferida -> nenhum rótulo derivado
    assert rotulos_da_chamada(db, det) == {}, \
        "tratou chamada aberta como confirmação"

    # marca Bruno como ausente (identificação equivocada) e fecha
    db.set_attendance_override(dia, bruno, False, "não era ele", "Operador")
    db.close_attendance(dia, "Operador", 1, 1)

    labels = rotulos_da_chamada(db, det)
    por_fonte = {chave(d): d for d in det}
    # Ana: conferida e sem correção -> confirmada
    for d in det:
        if d["person_id"] == ana:
            assert labels[chave(d)] == "certo", labels
    # Bruno: marcado ausente -> identificação errada
    trilha = next(d for d in det if d["person_id"] == bruno)
    assert labels[chave(trilha)] == "errado", labels

    # sugestão sai daí, sem ninguém rotular à mão
    import io
    from contextlib import redirect_stdout
    from scripts.calibrate_threshold import juntar_rotulos, sugerir
    combinado, origem = juntar_rotulos(db, det)
    assert origem["chamada"] == 3 and origem["manual"] == 0, origem
    buf = io.StringIO()
    with redirect_stdout(buf):
        sugerir(det, combinado, origem, 0.363)
    saida = buf.getvalue()
    # erro em 0.41, pior acerto em 0.71 -> ponto médio 0.56
    assert "SEPARADAS" in saida and "0.56" in saida, saida
    assert "3 de chamadas conferidas" in saida, saida

    # correção "presente" (falso negativo) não rotula detecção — não houve
    db.reopen_attendance(dia)
    db.set_attendance_override(dia, ana, True, "", "Operador")
    db.close_attendance(dia, "Operador", 2, 2)
    labels2 = rotulos_da_chamada(db, det)
    assert all(labels2[chave(d)] == "certo" or d["person_id"] == bruno
               for d in det if chave(d) in labels2)
    return ("chamada aberta não confirma; fechada rende rótulos automáticos; "
            "ponto médio calculado sem revisão manual")


@teste
def calibracao_nao_inventa_limiar_quando_ha_sobreposicao():
    import io
    from contextlib import redirect_stdout
    from scripts.calibrate_threshold import sugerir

    def det(*scores):
        return [{"fonte": "evento", "id": i + 1, "score": s}
                for i, s in enumerate(scores)]

    def rodar(deteccoes, rotulos):
        origem = {"chamada": len(rotulos), "manual": 0, "total": len(rotulos)}
        buf = io.StringIO()
        with redirect_stdout(buf):
            sugerir(deteccoes, rotulos, origem, 0.363)
        return buf.getvalue()

    # separadas: pior erro 0.42, pior acerto 0.85 -> ponto médio 0.635
    saida = rodar(det(0.42, 0.85, 0.88),
                  {"evento:1": "errado", "evento:2": "certo", "evento:3": "certo"})
    assert "SEPARADAS" in saida and "0.635" in saida, saida

    # sobrepostas: erro 0.80 acima de acertos 0.50 e 0.60 -> não sugere número
    s2 = rodar(det(0.50, 0.80, 0.60),
               {"evento:1": "certo", "evento:2": "errado", "evento:3": "certo"})
    assert "SOBREP" in s2 and "Limiar sugerido" not in s2, s2

    # sem rótulo nenhum: aponta os dois caminhos, não chuta
    s3 = rodar(det(0.5, 0.6), {})
    assert "Sem rótulo nenhum" in s3 and "FECHE chamadas" in s3, s3
    return ("separadas -> 0.635; sobrepostas -> explica sem inventar; "
            "sem rótulo -> orienta a conferir chamadas")


def _banco_recall(nome):
    """Banco com pessoas e um helper para plantar trilhas."""
    from core.database import Database
    pasta = os.path.join(TMP, nome)
    os.makedirs(pasta, exist_ok=True)
    db = Database(os.path.join(pasta, "r.db"))
    ids = {n: db.add_person(n) for n in ("Alfa", "Beta", "Gama", "Delta")}
    crop = [{"path": "c.jpg", "quality": 1.0, "face": "[]"}]

    def trilha(ts, nome_visto, score, status="processado"):
        t = db.add_track(ts, ts + 1, 8, crop)
        pid = ids.get(nome_visto)
        db.resolve_track(t, pid, nome_visto or "Desconhecido", score, "[]",
                         status)
        return t

    return db, ids, trilha


@teste
def recall_classifica_cada_degrau_do_funil():
    """Protocolo espaçado: o horário atribui sozinho, sem ambiguidade."""
    from collections import Counter
    from scripts.measure_recall import atribuir

    db, ids, trilha = _banco_recall("recall_funil")
    base = 1_700_000_000.0
    pessoas = {p["name"].lower(): p["id"] for p in db.list_people()}

    # Uma passagem a cada 60s, cada uma com um desfecho diferente.
    plano = [("Alfa", "acerto"), ("Beta", "desconhecido"), ("Gama", "errada"),
             ("Delta", "sem_recorte"), ("Alfa", "nao_detectada")]
    passagens = []
    for i, (nome, desfecho) in enumerate(plano):
        ts = base + i * 60
        passagens.append({"nome": nome, "ts": ts, "rodada": "sozinho"})
        if desfecho == "acerto":
            trilha(ts + 2, nome, 0.70)
        elif desfecho == "desconhecido":
            trilha(ts + 2, None, 0.30)
        elif desfecho == "errada":
            trilha(ts + 2, "Delta", 0.45)          # Gama identificada como Delta
        elif desfecho == "sem_recorte":
            trilha(ts + 2, None, 0.0, "descartado")
        # nao_detectada: não planta nada

    dets = db.detections_between(base - 30, base + 400)
    sobraram = atribuir(passagens, dets, pessoas, 10.0)
    obtido = Counter(p["resultado"] for p in passagens)

    esperado = {"acerto": 1, "desconhecido": 1, "errada": 1,
                "sem_recorte": 1, "nao_detectada": 1}
    assert dict(obtido) == esperado, f"{dict(obtido)} != {esperado}"
    assert sobraram == 0, f"sobraram {sobraram} detecções"
    assert not any(p["ambiguo"] for p in passagens), "não devia haver ambíguo"

    errada = next(p for p in passagens if p["resultado"] == "errada")
    assert errada["nome"] == "Gama" and errada["visto"] == "Delta", errada
    return "5 degraus reconhecidos; nada sobrando; nenhuma ambiguidade"


@teste
def recall_nao_inventa_erro_com_passagens_vizinhas():
    """O bug que o teste pegou: vizinha próxima virava 'pessoa errada'.

    Alfa e Beta atravessam com 2s de diferença e AMBOS são identificados
    certo. Classificando isolado, a detecção da Alfa cai na janela do Beta e
    é contada como 'Beta identificado como Alfa' — erro inventado.
    """
    from collections import Counter
    from scripts.measure_recall import atribuir

    db, ids, trilha = _banco_recall("recall_vizinha")
    base = 1_700_100_000.0
    pessoas = {p["name"].lower(): p["id"] for p in db.list_people()}

    passagens = [{"nome": "Alfa", "ts": base, "rodada": "grupo"},
                 {"nome": "Beta", "ts": base + 2, "rodada": "grupo"}]
    trilha(base + 1, "Alfa", 0.71)
    trilha(base + 3, "Beta", 0.69)

    dets = db.detections_between(base - 30, base + 60)
    atribuir(passagens, dets, pessoas, 10.0)
    obtido = Counter(p["resultado"] for p in passagens)

    assert obtido["acerto"] == 2, f"esperava 2 acertos, veio {dict(obtido)}"
    assert obtido["errada"] == 0, "inventou erro com vizinha na janela"
    return "duas passagens a 2s, ambas certas, zero erro inventado"


@teste
def recall_marca_ambiguidade_em_travessia_simultanea():
    """Horário idêntico não decide de quem é o rosto — e o script diz isso."""
    import io
    from contextlib import redirect_stdout
    from scripts.measure_recall import atribuir, _relatorio_funil

    db, ids, trilha = _banco_recall("recall_ambiguo")
    base = 1_700_200_000.0
    pessoas = {p["name"].lower(): p["id"] for p in db.list_people()}

    # Três pessoas atravessam JUNTAS; sai uma detecção nomeando a Alfa.
    passagens = [{"nome": n, "ts": base, "rodada": "grupo"}
                 for n in ("Alfa", "Beta", "Gama")]
    trilha(base + 1, "Alfa", 0.70)

    dets = db.detections_between(base - 30, base + 60)
    atribuir(passagens, dets, pessoas, 10.0)

    acerto = next(p for p in passagens if p["resultado"] == "acerto")
    assert acerto["ambiguo"], "devia marcar ambíguo: horários empatados"

    buf = io.StringIO()
    with redirect_stdout(buf):
        _relatorio_funil(passagens, 0.363)
    saida = buf.getvalue()
    assert "não é" in saida and "mensurável" in saida, saida
    assert "UM POR VEZ" in saida, "devia indicar a rodada espaçada"
    return "empate de horário marca ambíguo e o relatório recusa o número"


@teste
def recall_da_chamada_conta_falso_negativo():
    """Presente marcado à mão sem detecção nenhuma = o sistema perdeu."""
    import io
    from contextlib import redirect_stdout
    from scripts.measure_recall import medir_chamada

    db, ids, trilha = _banco_recall("recall_chamada")
    hoje = time.strftime("%Y-%m-%d")
    base = time.mktime(time.strptime(f"{hoje} 07:30:00", "%Y-%m-%d %H:%M:%S"))

    for i, n in enumerate(("Alfa", "Beta")):          # sistema pegou 2
        trilha(base + i * 20, n, 0.70)
    # conferência: Delta estava lá e o sistema não pegou
    db.set_attendance_override(hoje, ids["Delta"], True, "vi entrar", "Alfa")
    db.close_attendance(hoje, "Alfa", 3, 1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = medir_chamada(db)
    saida = buf.getvalue()

    assert rc == 0, saida
    assert "2 de 3" in saida, saida            # 2 identificados de 3 presentes
    assert "Delta" in saida, "devia nomear quem foi perdido"
    assert "66.7%" in saida, saida

    # dia aberto não conta: reabre e a medição não deve mais achar nada
    db.reopen_attendance(hoje)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = medir_chamada(db)
    assert rc2 == 1 and "Nenhuma chamada fechada" in buf2.getvalue()
    return "recall 2/3 com Delta nomeado; chamada reaberta deixa de contar"


@teste
def auth_libera_local_e_recusa_remoto_sem_token():
    """A regra por exposição: local passa, rede sem token é RECUSADA."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    # mesma máquina, nenhum token no config -> tudo liberado
    c, _ = _api_cliente("auth_local", host="127.0.0.1")
    assert c.get("/health").status_code == 200
    assert c.get("/people").status_code == 200, "local não devia exigir token"
    assert c.get("/docs").status_code == 200, "/docs local devia abrir"

    import importlib
    api = importlib.import_module("api")
    # As formas de endereço local que uma lista fixa deixaria de fora.
    # ::ffff:127.0.0.1 é conexão IPv4 chegando por socket IPv6 (dual-stack);
    # 127.0.0.2 porque o loopback é o /8 inteiro, não só o .1.
    for local in ("127.0.0.1", "::1", "127.0.0.2", "::ffff:127.0.0.1",
                  "127.255.255.254"):
        assert api.endereco_local(local), f"{local} devia contar como local"
    for remoto in ("10.0.0.9", "192.168.1.7", "0.0.0.0", "", "testclient",
                   "::ffff:10.0.0.9"):
        assert not api.endereco_local(remoto), f"{remoto} NÃO devia ser local"

    # pela rede, nenhum token configurado -> 503 (antes isto era 200 aberto)
    c2, _ = _api_cliente("auth_remoto_sem", host="10.0.0.9")
    assert c2.get("/health").status_code == 200, "/health fica aberta"
    r = c2.get("/people")
    assert r.status_code == 503, f"esperava 503, veio {r.status_code}"
    assert "token" in r.json()["detail"].lower(), r.json()
    assert "secrets.token_urlsafe" in r.json()["detail"], "devia ensinar a gerar"

    # a forma IPv6-mapeada precisa passar como local, senão o teste local
    # numa máquina em dual-stack tomaria 503
    c3, _ = _api_cliente("auth_v6", host="::ffff:127.0.0.1")
    assert c3.get("/people").status_code == 200, "IPv4 via socket IPv6 travou"
    return ("local sem token: 200 (inclui ::ffff:127.0.0.1) | "
            "rede sem token: 503 com instrução")


@teste
def auth_valida_token_e_protege_o_docs():
    """Token certo passa, errado não, e /docs deixa de vazar o mapa da API."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, _ = _api_cliente("auth_token", host="10.0.0.9", **{"api.token": "s3gr3d0"})

    assert c.get("/people").status_code == 401, "sem cabeçalho devia dar 401"
    assert c.get("/people", headers={"x-api-token": "errado"}).status_code == 401
    assert c.get("/people", headers={"x-api-token": "s3gr3d0"}).status_code == 200
    # Bearer é a forma que o sistema da escola provavelmente vai usar
    assert c.get("/people",
                 headers={"authorization": "Bearer s3gr3d0"}).status_code == 200

    # /docs, /openapi.json e /redoc: registradas no nível do Starlette, não
    # executam dependências do router. Com `dependencies=[Depends(...)]` no app
    # elas respondiam 200 para cliente remoto sem token — o middleware corrige.
    for rota in ("/docs", "/openapi.json", "/redoc"):
        assert c.get(rota).status_code == 401, f"{rota} vazou sem token"
        assert c.get(rota, headers={"x-api-token": "s3gr3d0"}).status_code == 200

    return "401 sem/com token errado; 200 via header e Bearer; /docs protegido"


@teste
def auth_tokens_por_consumidor_e_compatibilidade():
    """Mapa nome->token funciona, identifica quem chamou, e revoga em separado."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    import importlib
    api = importlib.import_module("api")

    # forma nova
    assert api._carregar_tokens({"tokens": {"painel": "a", "escola": "b"}}) == \
        {"painel": "a", "escola": "b"}
    # forma antiga continua valendo
    assert api._carregar_tokens({"token": "x"}) == {"padrao": "x"}
    # as duas juntas
    assert api._carregar_tokens({"token": "x", "tokens": {"escola": "b"}}) == \
        {"padrao": "x", "escola": "b"}
    # vazio e lixo são ignorados, não viram token válido
    assert api._carregar_tokens({"token": "  ", "tokens": {"a": "", "b": None}}) == {}
    assert api._carregar_tokens(None) == {}

    c, _ = _api_cliente("auth_multi", host="10.0.0.9",
                        **{"api.token": "", "api.tokens": {"painel": "p1",
                                                           "escola": "e1"}})
    assert c.get("/people", headers={"x-api-token": "p1"}).status_code == 200
    assert c.get("/people", headers={"x-api-token": "e1"}).status_code == 200
    assert c.get("/people", headers={"x-api-token": "p2"}).status_code == 401
    return "mapa e string convivem; vazio não vira token; dois consumidores ok"


@teste
def auth_snapshot_nao_escapa_da_pasta():
    """Path traversal e o irmão de nome parecido que o startswith deixava passar."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    c, pasta = _api_cliente("auth_path", host="127.0.0.1")
    import importlib
    api = importlib.import_module("api")

    # arquivo legítimo dentro da base
    base = api.SNAP_BASE
    base.mkdir(parents=True, exist_ok=True)
    (base / "ok.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    assert c.get("/snapshots/ok.jpg").status_code == 200

    # o ataque óbvio, já barrado pelo .resolve()
    assert c.get("/snapshots/../../../etc/passwd").status_code in (404, 200)
    assert c.get("/snapshots/..%2f..%2fetc%2fpasswd").status_code == 404

    # o caso que o startswith deixava passar: diretório IRMÃO de nome parecido
    irmao = base.parent / (base.name + "-privado")
    irmao.mkdir(parents=True, exist_ok=True)
    (irmao / "sigilo.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    alvo = f"/snapshots/../{base.name}-privado/sigilo.jpg"
    r = c.get(alvo)
    assert r.status_code == 404, \
        f"vazou arquivo de diretório irmão ({r.status_code}) — is_relative_to falhou"
    return "arquivo válido serve; traversal e diretório irmão barrados"


def _funcao_do_painel(nomes, globais):
    """Extrai funções e constantes do panel/app.py com globais controlados.

    O painel é um script Streamlit: importá-lo executaria a página inteira.
    Pegando só os trechos necessários dá para testar o CÓDIGO REAL — não uma
    réplica que poderia divergir do arquivo sem ninguém notar.

    Aceita lista porque uma função do painel pode chamar outra (a `imagem`
    chama `_cache_imagem`). Extrair só a primeira dava NameError — falha do
    andaime de teste, não do código sob teste.
    """
    import ast
    if isinstance(nomes, str):
        nomes = [nomes]
    fonte = open(os.path.join(RAIZ, "panel", "app.py"), encoding="utf-8").read()
    arvore = ast.parse(fonte)

    corpo = []
    for no in arvore.body:
        if isinstance(no, ast.FunctionDef) and no.name in nomes:
            corpo.append(no)
        # Constantes LITERAIS de módulo que as funções usam (ex.: _CACHE_MAX).
        # A restrição a `ast.Constant` é essencial: sem ela entrariam também
        # `API = cfg.api.base_url...` e `S = requests.Session()`, que são
        # maiúsculos, dependem de estado do módulo (estouraria no exec) e
        # sobrescreveriam justamente os dublês que o teste injeta.
        elif (isinstance(no, ast.Assign)
                and isinstance(no.value, ast.Constant)
                and len(no.targets) == 1
                and isinstance(no.targets[0], ast.Name)
                and no.targets[0].id.isupper()):
            corpo.append(no)

    faltando = [n for n in nomes
                if not any(isinstance(c, ast.FunctionDef) and c.name == n
                           for c in corpo)]
    assert not faltando, f"não achei no panel/app.py: {faltando}"

    mod = ast.Module(body=corpo, type_ignores=[])
    exec(compile(mod, "panel/app.py", "exec"), globais)      # noqa: S102
    return globais[nomes[-1]]


@teste
def painel_baixa_imagem_com_token():
    """A correção que desbloqueia o token: imagem vai por requests, não por URL.

    Passar URL ao st.image faz o Streamlit devolvê-la intacta e o NAVEGADOR
    buscar o arquivo, sem o cabeçalho do token. Ligar o token quebrava todas
    as imagens do painel. Este teste roda a API de verdade com token e
    confirma que a função do painel traz os bytes.
    """
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    import importlib
    import requests

    c, _ = _api_cliente("painel_img", host="10.0.0.9",
                        **{"api.token": "tok3n"})
    api = importlib.import_module("api")
    api.SNAP_BASE.mkdir(parents=True, exist_ok=True)
    (api.SNAP_BASE / "foto.jpg").write_bytes(b"\xff\xd8\xff\xd9CONTEUDO")

    # O TestClient devolve resposta httpx, que não tem `.ok` — atributo do
    # requests, que é o que o painel usa de verdade. Este adaptador só repõe
    # a diferença de biblioteca.
    class Resp:
        def __init__(self, r):
            self.status_code = r.status_code
            self.content = r.content
            self.ok = 200 <= r.status_code < 300
            self._r = r

        def json(self):
            # O painel passou a ler o `detail` da resposta em vez de deduzir a
            # causa pelo código HTTP, então o dublê precisa de json() e text.
            # Levanta ValueError quando o corpo não é JSON, igual ao requests —
            # é desse erro que o painel se defende.
            try:
                return self._r.json()
            except Exception as exc:                     # noqa: BLE001
                raise ValueError(str(exc)) from exc

        @property
        def text(self):
            return self._r.text

    # Session que fala com a API por dentro do TestClient, com o token.
    # Conta as idas à rede, para o teste poder provar que o cache evita
    # rebaixar a mesma foto — que é o ponto da otimização.
    class SessaoFalsa:
        def __init__(self, token):
            self.headers = {"X-API-Token": token} if token else {}
            self.chamadas = 0

        def get(self, url, timeout=None):
            self.chamadas += 1
            caminho = url.replace("http://api-de-teste", "")
            return Resp(c.get(caminho, headers=self.headers))

    capturado = {}

    class StFalso:
        # O `_cache_imagem` real guarda o cache aqui, então o dublê precisa
        # ter session_state — é assim que se testa o código de verdade em vez
        # de uma réplica.
        session_state = {}

        @staticmethod
        def image(dados, **kw):
            capturado["bytes"] = dados

        @staticmethod
        def caption(texto, **kw):
            capturado["aviso"] = texto

    # Cache do painel: dicionário simples, injetado nos globais da função.
    cache = {}
    sessoes = []

    def montar(token, limpar_cache=True):
        """Simula uma sessão nova do painel.

        Limpa o cache por padrão: sem isso, o caso 1 deixaria a foto guardada
        e o caso 2 (sem token) acertaria pelo cache em vez de exercitar o 401.
        O caso que PROVA o cache passa limpar_cache=False de propósito.
        """
        capturado.clear()
        if limpar_cache:
            StFalso.session_state.clear()
        sessao = SessaoFalsa(token)
        sessoes.append(sessao)
        g = {"S": sessao, "API": "http://api-de-teste", "st": StFalso,
             "requests": requests}
        return _funcao_do_painel(["_cache_imagem", "imagem"], g)

    # 1) com o token certo: os bytes chegam
    montar("tok3n")("/snapshots/foto.jpg")
    assert capturado.get("bytes", b"").endswith(b"CONTEUDO"), capturado
    assert "aviso" not in capturado, capturado

    # 2) sem token: não renderiza imagem, e explica que é token
    montar("")("/snapshots/foto.jpg")
    assert "bytes" not in capturado, "não devia renderizar imagem"
    assert "token" in capturado["aviso"].lower(), capturado

    # 3) foto que não existe: a mensagem vem da API e é DIFERENTE da de token.
    # Antes o painel deduzia a causa pelo código HTTP e escrevia "retenção" —
    # suposição que só às vezes era verdade. Agora repassa o `detail`, então o
    # que se testa é que a explicação é a da API e não se confunde com token.
    montar("tok3n")("/snapshots/nao-existe.jpg")
    assert "bytes" not in capturado
    aviso_404 = capturado["aviso"]
    assert "não encontrado" in aviso_404.lower(), aviso_404
    assert "token" not in aviso_404.lower(), \
        "404 não pode ser anunciado como problema de token"

    # 4) aceita caminho relativo e URL absoluta (snapshot_url vem dos dois jeitos)
    montar("tok3n")("http://api-de-teste/snapshots/foto.jpg")
    assert capturado.get("bytes", b"").endswith(b"CONTEUDO"), capturado

    # 5) o cache evita rebaixar a MESMA foto. É o ponto da otimização: o
    # Streamlit reexecuta o script a cada clique, e sem isto a aba de
    # reconhecimentos redescarregava dezenas de imagens por interação.
    f = montar("tok3n")
    sessao = sessoes[-1]
    f("/snapshots/foto.jpg")
    f("/snapshots/foto.jpg")
    f("/snapshots/foto.jpg")
    assert sessao.chamadas == 1, \
        f"cache não pegou: {sessao.chamadas} idas à rede para a mesma foto"
    assert capturado.get("bytes", b"").endswith(b"CONTEUDO"), capturado

    # 6) o preview ao vivo NÃO pode ser cacheado: muda a cada instante.
    f = montar("tok3n")
    sessao = sessoes[-1]
    f("/enroll/preview?t=1")
    f("/enroll/preview?t=1")
    assert sessao.chamadas == 2, "preview ao vivo não pode vir do cache"

    return ("bytes com token; 401 e 404 distintos; caminho e URL; "
            "cache poupa rede na foto e é ignorado no preview ao vivo")


@teste
def painel_nao_tem_mais_st_image_com_url():
    """Nenhum ponto de chamada voltou para o padrão antigo.

    Guarda de regressão: os 5 st.image(f"{API}...") eram o bug. Se alguém
    reintroduzir um, ele quebra silenciosamente só quando o token estiver
    ligado — ou seja, direto em produção.
    """
    import re
    fonte = open(os.path.join(RAIZ, "panel", "app.py"), encoding="utf-8").read()
    ruins = re.findall(r'st\.image\(\s*f?"?\{?API\}?[^\)]*', fonte)
    assert not ruins, f"st.image com URL da API: {ruins}"

    # Todo st.image tem que receber BYTES, nunca uma string. Fixar a lista
    # exata (["r.content"]) quebrou quando o cache adicionou uma segunda
    # chamada legítima — o teste passou a proibir manutenção em vez de
    # proibir a regressão. Agora a regra é a propriedade que importa.
    chamadas = [c.strip() for c in re.findall(r"st\.image\(([^,\)]+)", fonte)]
    assert chamadas, "nenhum st.image encontrado — o wrapper sumiu?"
    for arg in chamadas:
        assert not arg.startswith(('"', "'", 'f"', "f'")), \
            f"st.image recebendo string (vira URL para o navegador): {arg}"
        assert "API" not in arg, f"st.image com URL montada: {arg}"

    # e todos os pontos de exibição usam o wrapper
    assert fonte.count("imagem(") >= 6, "esperava a definição + 5 usos"
    return (f"{len(chamadas)} st.image, todos com bytes; "
            f"{fonte.count('imagem(') - 1} usos do wrapper")


@teste
def rotulos_ficam_no_banco_e_nao_mexem_na_presenca():
    """Uma fonte só para 'certo/errado', e rotular não altera a chamada."""
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    import importlib
    c, pasta = _api_cliente("rotulos", host="127.0.0.1")
    api = importlib.import_module("api")
    db = api.db

    hoje = time.strftime("%Y-%m-%d")
    alfa = db.add_person("Alfa")
    e1 = db.add_event(alfa, "Alfa", 0.91, "20260101/a.jpg", 1)
    e2 = db.add_event(alfa, "Alfa", 0.42, "20260101/b.jpg", 1)

    # --- filtro por pessoa e união das duas origens ------------------------- #
    beta = db.add_person("Beta")
    t = db.add_track(time.time(), time.time() + 1, 8,
                     [{"path": "20260101/c.jpg", "quality": 9.0, "face": "[]"}])
    db.resolve_track(t, beta, "Beta", 0.80, "[]")

    todas = c.get("/detections").json()["deteccoes"]
    assert {d["fonte"] for d in todas} == {"evento", "trilha"}, \
        "a lista tem que unir os dois modos, senão fica vazia em captura"

    so_alfa = c.get("/detections", params={"person_id": alfa}).json()["deteccoes"]
    assert {d["nome"] for d in so_alfa} == {"Alfa"}, so_alfa
    assert len(so_alfa) == 2, so_alfa

    # URLs na base certa de cada origem
    ev = next(d for d in todas if d["fonte"] == "evento")
    tr = next(d for d in todas if d["fonte"] == "trilha")
    assert ev["foto_url"].startswith("/snapshots/"), ev
    assert tr["foto_url"].startswith("/tracks/"), tr

    # --- rotular NÃO pode mexer na presença --------------------------------- #
    antes = c.get("/attendance", params={"dia": hoje}).json()
    presentes_antes = {p["nome"] for p in antes["presentes"]}
    assert "Alfa" in presentes_antes

    r = c.post("/detections/label",
               json={"fonte": "evento", "detection_id": e2,
                     "rotulo": "errado", "autor": "Operador"})
    assert r.status_code == 200, r.text

    depois = c.get("/attendance", params={"dia": hoje}).json()
    assert {p["nome"] for p in depois["presentes"]} == presentes_antes, \
        "rotular detecção não pode alterar a chamada"
    assert depois["correcoes"]["total"] == antes["correcoes"]["total"], \
        "rótulo não é correção de chamada"

    # o rótulo aparece na listagem
    lista = c.get("/detections", params={"person_id": alfa}).json()["deteccoes"]
    marcado = {d["id"]: d["rotulo"] for d in lista if d["fonte"] == "evento"}
    assert marcado[e2] == "errado" and marcado[e1] is None, marcado

    # desfazer
    c.request("DELETE", "/detections/label",
              params={"fonte": "evento", "detection_id": e2})
    lista = c.get("/detections", params={"person_id": alfa}).json()["deteccoes"]
    assert all(d["rotulo"] is None for d in lista), lista

    # valor inválido é recusado, não gravado torto
    assert c.post("/detections/label",
                  json={"fonte": "evento", "detection_id": e1,
                        "rotulo": "talvez"}).status_code == 400

    # --- a calibração lê do banco, e o manual vence o derivado -------------- #
    from scripts.calibrate_threshold import juntar_rotulos
    db.set_detection_label("evento", e1, "errado", "painel")
    db.set_attendance_override(hoje, alfa, True, "", "Operador")
    db.close_attendance(hoje, "Operador", 2, 1)

    deteccoes = db.list_detections(limit=100)
    combinado, origem = juntar_rotulos(db, deteccoes)
    assert combinado[f"evento:{e1}"] == "errado", \
        "o rótulo do painel tem que prevalecer sobre o derivado da chamada"
    assert origem["manual"] >= 1, origem
    return ("duas origens numa lista só; filtro por pessoa; rótulo no banco "
            "sem tocar na presença; manual vence o derivado")


@teste
def rotulos_migram_do_json_antigo():
    """O JSON que o calibrate usava vira linha no banco, uma vez só."""
    import json as _json
    from core.database import Database
    from core.config import project_path
    import scripts.calibrate_threshold as cal

    pasta = os.path.join(TMP, "migra")
    os.makedirs(pasta, exist_ok=True)
    db = Database(os.path.join(pasta, "m.db"))

    antigo = project_path(cal.LABELS_FILE)
    antigo.parent.mkdir(parents=True, exist_ok=True)
    salvo = antigo.read_text(encoding="utf-8") if antigo.exists() else None
    migrado = antigo.with_suffix(antigo.suffix + ".migrado")
    salvo_mig = migrado.read_text(encoding="utf-8") if migrado.exists() else None
    try:
        # chave antiga (só número) significava evento; a nova traz a fonte
        antigo.write_text(_json.dumps({"7": "errado", "trilha:3": "certo"}),
                          encoding="utf-8")

        db.set_detection_label("evento", 7, "certo", "painel")   # já existe
        n = cal._migrar_json(db)

        rotulos = db.detection_labels()
        assert rotulos["trilha:3"] == "certo", rotulos
        # O que já estava no banco NÃO pode ser sobrescrito pelo arquivo:
        # quem rotulou pelo painel decidiu depois.
        assert rotulos["evento:7"] == "certo", \
            "migração não pode desfazer rótulo mais recente do painel"
        assert n == 1, f"devia importar só o que faltava, importou {n}"

        assert not antigo.exists(), "o JSON devia ter sido renomeado"
        assert migrado.exists(), "o original precisa ser preservado"

        # rodar de novo não duplica nem quebra
        assert cal._migrar_json(db) == 0
    finally:
        for p, conteudo in ((antigo, salvo), (migrado, salvo_mig)):
            if conteudo is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(conteudo, encoding="utf-8")
    return "chave antiga normalizada; banco vence o arquivo; JSON preservado"


@teste
def chamada_traz_a_foto_da_melhor_deteccao():
    """Sem foto, conferir a chamada é confirmar um nome, não um rosto.

    E são essas confirmações que viram rótulo da calibração e denominador da
    medição de recall — conferência às cegas contamina as duas.
    """
    try:
        import fastapi.testclient  # noqa: F401
    except ImportError:
        return "PULADO: fastapi.testclient indisponível"

    import importlib
    c, pasta = _api_cliente("chamada_foto", host="127.0.0.1")
    api = importlib.import_module("api")
    db = api.db

    hoje = time.strftime("%Y-%m-%d")
    base = time.mktime(time.strptime(f"{hoje} 07:30:00", "%Y-%m-%d %H:%M:%S"))

    alfa = db.add_person("Alfa")
    beta = db.add_person("Beta")

    # Alfa vem do modo REALTIME: duas passagens, e a melhor NÃO é a primeira.
    db.add_event(alfa, "Alfa", 0.55, "20260101/fraca.jpg", 1)
    db.add_event(alfa, "Alfa", 0.91, "20260101/forte.jpg", 1)

    # Beta vem do modo CAPTURA: a foto sai de track_crops, em OUTRA base.
    t = db.add_track(base + 60, base + 61, 9,
                     [{"path": "20260101/ruim.jpg", "quality": 10.0, "face": "[]"},
                      {"path": "20260101/bom.jpg", "quality": 99.0, "face": "[]"}])
    db.resolve_track(t, beta, "Beta", 0.77, "[]")

    d = c.get("/attendance", params={"dia": hoje}).json()
    por_nome = {p["nome"]: p for p in d["presentes"]}
    assert set(por_nome) == {"Alfa", "Beta"}, d

    # A foto tem que ser a do MAIOR score, não a primeira nem a última.
    assert por_nome["Alfa"]["foto_url"] == "/snapshots/20260101/forte.jpg", \
        por_nome["Alfa"]
    assert por_nome["Alfa"]["foto_score"] == 0.91

    # Recorte de trilha usa a rota /tracks/, que fica em base diferente.
    # Montar /snapshots/ para ele dava 404 em tudo vindo do modo captura.
    assert por_nome["Beta"]["foto_url"].startswith("/tracks/"), por_nome["Beta"]
    assert "bom.jpg" in por_nome["Beta"]["foto_url"], "devia pegar o recorte de maior qualidade"

    # As duas rotas servem de verdade, cada uma da sua base.
    api.SNAP_BASE.joinpath("20260101").mkdir(parents=True, exist_ok=True)
    api.SNAP_BASE.joinpath("20260101/forte.jpg").write_bytes(b"\xff\xd8SNAP")
    api.TRACKS_BASE.joinpath("20260101").mkdir(parents=True, exist_ok=True)
    api.TRACKS_BASE.joinpath("20260101/bom.jpg").write_bytes(b"\xff\xd8CROP")

    r = c.get(por_nome["Alfa"]["foto_url"])
    assert r.status_code == 200 and r.content.endswith(b"SNAP"), r.status_code
    r = c.get(por_nome["Beta"]["foto_url"])
    assert r.status_code == 200 and r.content.endswith(b"CROP"), r.status_code

    # A rota nova tem a mesma proteção de caminho da antiga.
    assert c.get("/tracks/../../etc/passwd").status_code == 404

    # Quem não foi detectado não tem foto — e isso é informação, não erro:
    # marcar presente aí é o falso negativo que a medição procura.
    gama = db.add_person("Gama")
    d2 = c.get("/attendance", params={"dia": hoje}).json()
    ausente = next(p for p in d2["ausentes"] if p["nome"] == "Gama")
    assert "foto_url" not in ausente, ausente
    return ("foto é a de maior score; realtime em /snapshots e captura em "
            "/tracks; ambas servidas; ausente sem foto")


@teste
def recall_aborta_com_trilha_pendente_e_avisa_do_cooldown():
    """Dois jeitos de o relatório mentir por causa da configuração.

    Trilha `pendente` não casa com degrau nenhum do funil, então a passagem
    cairia em "nunca detectada" — culpando a câmera por lote não executado.
    E o cooldown do worker suprime evento repetido da mesma pessoa, o que
    produz o mesmo falso negativo.
    """
    import io
    from contextlib import redirect_stdout
    from pathlib import Path
    from scripts.measure_recall import _passagens_proximas, medir_passagens

    # --- a contagem de passagens dentro do cooldown ------------------------ #
    base = 1_700_000_000.0
    p = [{"nome": "Alfa", "ts": base}, {"nome": "Beta", "ts": base + 3},
         {"nome": "Alfa", "ts": base + 8},          # 8s < 15s -> suprimida
         {"nome": "Alfa", "ts": base + 40}]         # 32s depois -> ok
    assert _passagens_proximas(p, 15.0) == 1, _passagens_proximas(p, 15.0)
    assert _passagens_proximas(p, 0.0) == 0
    assert _passagens_proximas(p, 60.0) == 2, "8s e 40s ambos < 60s"

    # --- aborta quando há trilha pendente ---------------------------------- #
    db, ids, trilha = _banco_recall("recall_pendente")
    pasta = os.path.join(TMP, "recall_pendente")
    crop = [{"path": "c.jpg", "quality": 1.0, "face": "[]"}]
    db.add_track(base + 1, base + 2, 8, crop)        # fica 'pendente'

    csv_path = os.path.join(pasta, "reg.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("nome;hora;rodada\n")
        fh.write(f"Alfa;{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base))};solo\n")

    class CfgFalso(dict):
        class _R:
            cosine_threshold = 0.363
        class _A:
            host, port = "127.0.0.1", 8000
        recognition, api = _R(), _A()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = medir_passagens(db, CfgFalso(), Path(csv_path), 10.0,
                             time.strftime("%Y-%m-%d", time.localtime(base)))
    saida = buf.getvalue()
    assert rc == 1, f"devia abortar, retornou {rc}\n{saida}"
    assert "NÃO foram" in saida and "recognize_batch" in saida, saida
    assert "nunca detectada" not in saida.split("Rode primeiro")[0].split("⚠")[0], \
        "não devia imprimir o funil com dado incompleto"
    return "pendente aborta apontando o lote; cooldown conta passagens afetadas"


@teste
def camera_backend_por_plataforma():
    """Webcam por índice: V4L2 no Linux, DirectShow no Windows.

    V4L2 é API do Linux e não existe no Windows — pedi-la lá faz o
    VideoCapture não abrir, com o sintoma inútil "can't open camera by index".
    O projeto roda no Pi (Linux) e no PC da equipe (Windows), então a escolha
    tem que ser por plataforma.
    """
    import importlib
    real = sys.platform
    try:
        esperado = {"win32": cv2.CAP_DSHOW,
                    "linux": cv2.CAP_V4L2,
                    "darwin": cv2.CAP_ANY}
        for plat, backend in esperado.items():
            sys.platform = plat
            cam = importlib.reload(importlib.import_module("core.camera"))
            for fonte in (0, "0", "1"):
                _, b = cam._parse_source(fonte)
                assert b == backend, f"{plat} com {fonte!r}: {b} != {backend}"
            # RTSP e arquivo continuam no FFmpeg em toda plataforma
            for fonte in ("rtsp://host/stream", "video.avi"):
                _, b = cam._parse_source(fonte)
                assert b == cv2.CAP_FFMPEG, f"{plat} {fonte}: {b}"
            # /dev/video0 só faz sentido no Linux, e segue explícito
            assert cam._parse_source("/dev/video0")[1] == cv2.CAP_V4L2
    finally:
        sys.platform = real
        importlib.reload(importlib.import_module("core.camera"))

    # is_local_device precisa valer para webcam em QUALQUER plataforma: é o que
    # habilita pedir MJPG. Comparar com CAP_V4L2 deixava o Windows entregando
    # YUYV, que satura o barramento USB.
    import core.camera as cam
    for fonte in (0, "0"):
        c = cam.Camera(fonte)
        assert c.is_local_device, f"{fonte!r} devia ser dispositivo local"
    assert not cam.Camera("rtsp://host/stream").is_local_device
    return "DSHOW no Windows, V4L2 no Linux; RTSP no FFmpeg; MJPG habilitado"


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
