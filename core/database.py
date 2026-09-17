"""Persistência em SQLite (modo WAL).

Tabelas:
  people      (id, name, created_at)
  embeddings  (id, person_id, vec BLOB)   -- bytes de numpy float32 (128,)
  events      (id, person_id, name, score, ts, snapshot_path, is_known)

WAL permite 1 escritor + N leitores em acesso local (worker grava, API lê),
que é exatamente o cenário deste projeto.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import project_path


# Nome usado nas linhas cujo titular foi removido com anonimização.
ANONIMO = "(removido)"

class MatriculaDuplicada(Exception):
    """Outro aluno já tem essa matrícula."""

    def __init__(self, person_id: int, nome: str):
        self.person_id, self.nome = person_id, nome
        super().__init__(f"Matrícula já usada por '{nome}' (id {person_id}).")


def normalizar_matricula(valor) -> str | None:
    """Normaliza a matrícula: tira espaços das pontas e colapsa internos.

    **Nenhuma validação de formato, de propósito.** A versão anterior deste
    campo era o ID INEP, e eu tinha posto uma checagem de "12 dígitos" baseada
    no que eu *supunha* do formato do Censo Escolar — suposição que se revelou
    errada duas vezes: o identificador usado pela escola não é o INEP, é a
    matrícula, e o formato dela não foi confirmado.

    Então aqui só se remove ruído de digitação. Não filtra por dígito: se a
    matrícula tiver letra, prefixo de turma ou barra, ela sobrevive. Validar
    formato que não se conhece transforma dado válido em dado rejeitado, e o
    erro aparece no pior momento — na hora de cadastrar um aluno de verdade.

    Guardado como TEXT, nunca INTEGER: é identificador, não quantidade, e
    zero à esquerda tem que sobreviver.
    """
    if valor is None:
        return None
    limpo = " ".join(str(valor).split())
    return limpo or None


@dataclass
class Gallery:
    matrix: np.ndarray | None = None          # (N, 128) normalizado
    ids: list = field(default_factory=list)   # person_id por linha
    names: dict = field(default_factory=dict)  # person_id -> nome


class Database:
    def __init__(self, db_path: str):
        self.path = str(project_path(db_path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init(self):
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    vec       BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     INTEGER,
                    name          TEXT NOT NULL,
                    score         REAL NOT NULL,
                    ts            REAL NOT NULL,
                    snapshot_path TEXT,
                    is_known      INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_emb_person ON embeddings(person_id);

                -- Pipeline de duas fases (modo captura).
                -- Uma "trilha" é uma pessoa acompanhada ao longo dos frames.
                -- A captura grava as trilhas; o reconhecimento em lote as
                -- processa depois e preenche person_id/name/score.
                CREATE TABLE IF NOT EXISTS tracks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at  REAL NOT NULL,
                    ended_at    REAL NOT NULL,
                    frames      INTEGER NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pendente',
                    person_id   INTEGER,
                    name        TEXT,
                    score       REAL,
                    votes       TEXT
                );
                CREATE TABLE IF NOT EXISTS track_crops (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id  INTEGER NOT NULL,
                    path      TEXT NOT NULL,
                    quality   REAL NOT NULL,
                    face      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
                CREATE INDEX IF NOT EXISTS idx_tracks_started ON tracks(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_crops_track ON track_crops(track_id);

                -- Correção manual da chamada.
                --
                -- Fica em tabela SEPARADA de propósito: sobrescrever eventos ou
                -- trilhas destruiria o que o sistema detectou, e é justamente a
                -- diferença entre o automático e o corrigido que mede o acerto
                -- do reconhecimento. Uma correção só existe quando a pessoa
                -- DISCORDA da máquina.
                --   presente=1 -> o sistema não detectou, mas a criança veio
                --                 (falso negativo do reconhecimento)
                --   presente=0 -> o sistema detectou, mas era engano
                --                 (falso positivo do reconhecimento)
                CREATE TABLE IF NOT EXISTS attendance_overrides (
                    dia        TEXT    NOT NULL,
                    person_id  INTEGER NOT NULL,
                    presente   INTEGER NOT NULL,
                    motivo     TEXT,
                    autor      TEXT,
                    created_at REAL    NOT NULL,
                    PRIMARY KEY (dia, person_id)
                );

                -- Fechamento da chamada: separa "dados parciais, ainda em
                -- conferência" de "conferida por uma pessoa". Sem isso, quem
                -- consome não distingue rascunho de registro.
                CREATE TABLE IF NOT EXISTS attendance_closures (
                    dia         TEXT PRIMARY KEY,
                    closed_at   REAL NOT NULL,
                    autor       TEXT,
                    presentes   INTEGER,
                    correcoes   INTEGER
                );

                -- Rótulo de uma detecção ESPECÍFICA: "esta aqui não é ela".
                --
                -- Mais fino que attendance_overrides, que fala de pessoa/dia.
                -- Uma pessoa pode ter cinco detecções no dia, quatro certas e
                -- uma errada — e é essa granularidade que a calibração precisa
                -- para achar o limiar.
                --
                -- A chave é o PAR (fonte, detection_id): `events` e `tracks`
                -- têm autoincremento próprio, então o id sozinho colide entre
                -- as duas.
                --
                -- Deliberadamente NÃO altera a presença. Rotular é medir o
                -- acerto do reconhecimento; decidir quem esteve na escola é a
                -- aba Chamada. Misturar faria uma revisão de fotos mexer na
                -- frequência do aluno sem que ninguém pedisse.
                CREATE TABLE IF NOT EXISTS detection_labels (
                    fonte        TEXT    NOT NULL,   -- 'evento' | 'trilha'
                    detection_id INTEGER NOT NULL,
                    rotulo       TEXT    NOT NULL,   -- 'certo' | 'errado'
                    autor        TEXT,
                    created_at   REAL    NOT NULL,
                    PRIMARY KEY (fonte, detection_id)
                );
                """
            )
        self._migrar()

    def _migrar(self):
        """Acrescenta colunas novas a bancos já existentes.

        `CREATE TABLE IF NOT EXISTS` não altera tabela que já existe, então um
        banco criado por versão anterior ficaria sem as colunas novas. Aqui a
        adição é feita só quando falta, e sem apagar nada — o cadastro de quem
        já foi registrado continua valendo (fica apenas sem foto associada,
        porque essa informação não existia na época).
        """
        with self._connect() as con:
            existentes = {r["name"] for r in con.execute("PRAGMA table_info(embeddings)")}
            for coluna, tipo in (("snapshot_path", "TEXT"),
                                 ("created_at", "REAL"),
                                 ("quality", "REAL")):
                if coluna not in existentes:
                    con.execute(f"ALTER TABLE embeddings ADD COLUMN {coluna} {tipo}")

            # Matrícula do aluno: o identificador que o sistema de gestão da
            # escola usa. Índice único PARCIAL — dois alunos não podem
            # compartilhar a mesma matrícula, mas vários podem estar sem
            # preencher (sem a cláusula WHERE, o segundo NULL colidiria).
            #
            # A coluna `inep_id` de antes NÃO é removida: o dado já digitado
            # continua lá, intacto, para o caso de o Censo Escolar voltar a
            # ser necessário. Remover coluna em SQLite é operação destrutiva
            # e o ganho seria só estético.
            pessoas = {r["name"] for r in con.execute("PRAGMA table_info(people)")}
            if "matricula" not in pessoas:
                con.execute("ALTER TABLE people ADD COLUMN matricula TEXT")
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_people_matricula "
                "ON people(matricula) WHERE matricula IS NOT NULL")

    # ---- pessoas / embeddings -------------------------------------------------
    def person_by_matricula(self, matricula: str) -> dict | None:
        alvo = normalizar_matricula(matricula)
        if not alvo:
            return None
        with self._connect() as con:
            r = con.execute(
                "SELECT id, name, matricula, created_at FROM people "
                "WHERE matricula=?", (alvo,)).fetchone()
        return dict(r) if r else None

    def set_person_matricula(self, person_id: int, matricula) -> str | None:
        """Define (ou limpa) a matrícula. Levanta MatriculaDuplicada se for de outro."""
        alvo = normalizar_matricula(matricula)
        if alvo:
            dono = self.person_by_matricula(alvo)
            if dono and dono["id"] != person_id:
                raise MatriculaDuplicada(dono["id"], dono["name"])
        with self._connect() as con:
            con.execute("UPDATE people SET matricula=? WHERE id=?",
                        (alvo, person_id))
        return alvo

    def add_person(self, name: str, matricula=None) -> int:
        alvo = normalizar_matricula(matricula)
        if alvo:
            dono = self.person_by_matricula(alvo)
            if dono:
                raise MatriculaDuplicada(dono["id"], dono["name"])
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO people(name, created_at, matricula) VALUES(?, ?, ?)",
                (name, time.time(), alvo),
            )
            return int(cur.lastrowid)

    def add_embedding(self, person_id: int, vec, snapshot_path: str = None,
                      quality: float = None) -> int:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO embeddings(person_id, vec, snapshot_path, created_at, quality)
                VALUES(?, ?, ?, ?, ?)
                """,
                (person_id, blob, snapshot_path, time.time(), quality),
            )
            return int(cur.lastrowid)

    def list_embeddings(self, person_id: int) -> list[dict]:
        """Amostras da pessoa, com vetor — o vetor é usado para medir redundância."""
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, snapshot_path, created_at, quality, vec
                FROM embeddings WHERE person_id=? ORDER BY id
                """,
                (person_id,),
            ).fetchall()
        saida = []
        for r in rows:
            d = dict(r)
            d["vec"] = np.frombuffer(r["vec"], dtype=np.float32)
            saida.append(d)
        return saida

    def get_embedding(self, embedding_id: int) -> dict | None:
        with self._connect() as con:
            r = con.execute(
                "SELECT id, person_id, snapshot_path FROM embeddings WHERE id=?",
                (embedding_id,),
            ).fetchone()
        return dict(r) if r else None

    def count_embeddings(self, person_id: int) -> int:
        with self._connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id=?",
                (person_id,)).fetchone()[0])

    def delete_embedding(self, embedding_id: int) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM embeddings WHERE id=?", (embedding_id,))

    def get_person(self, person_id: int) -> dict | None:
        with self._connect() as con:
            r = con.execute(
                "SELECT id, name, matricula, created_at FROM people WHERE id=?",
                (person_id,)).fetchone()
        return dict(r) if r else None

    def rename_person(self, person_id: int, name: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))

    def delete_person(self, person_id: int, anonimizar: bool = False) -> dict:
        """Apaga a pessoa e TODO o rastro dela. Devolve os arquivos a remover.

        Quem chama é responsável por remover os arquivos — o banco não conhece o
        sistema de arquivos. Devolver a lista evita deixar imagens de rosto no
        disco depois de um pedido de exclusão.

        `anonimizar=True` preserva as linhas de eventos e trilhas com
        `person_id` nulo e nome neutro — a estatística de "alguém passou às
        7:42" continua, sem identificar quem. As IMAGENS são apagadas nos dois
        modos, porque a imagem do rosto é o dado identificante.

        Devolve os caminhos por base, já que as imagens moram em dois lugares:
            {"snapshots": [...],   # relativos a storage.snapshots_dir
             "tracks":    [...]}   # relativos a tracking.crops_dir
        """
        with self._connect() as con:
            snapshots = [r["p"] for r in con.execute(
                """
                SELECT snapshot_path AS p FROM embeddings
                 WHERE person_id=? AND snapshot_path IS NOT NULL
                UNION ALL
                SELECT snapshot_path AS p FROM events
                 WHERE person_id=? AND snapshot_path IS NOT NULL
                """, (person_id, person_id))]
            tracks = [r["path"] for r in con.execute(
                """
                SELECT tc.path FROM track_crops tc
                JOIN tracks t ON t.id = tc.track_id
                WHERE t.person_id = ?
                """, (person_id,))]

            # embeddings e a pessoa saem sempre — sem eles não há reconhecimento
            con.execute("DELETE FROM embeddings WHERE person_id=?", (person_id,))

            if anonimizar:
                con.execute(
                    "UPDATE events SET person_id=NULL, name=?, snapshot_path=NULL, "
                    "is_known=0 WHERE person_id=?", (ANONIMO, person_id))
                con.execute(
                    "UPDATE tracks SET person_id=NULL, name=? WHERE person_id=?",
                    (ANONIMO, person_id))
                # os recortes das trilhas viram órfãos: removemos as linhas
                con.execute(
                    """
                    DELETE FROM track_crops WHERE track_id IN
                        (SELECT id FROM tracks WHERE name = ? AND person_id IS NULL)
                    """, (ANONIMO,))
            else:
                con.execute(
                    """
                    DELETE FROM track_crops WHERE track_id IN
                        (SELECT id FROM tracks WHERE person_id = ?)
                    """, (person_id,))
                con.execute("DELETE FROM tracks WHERE person_id=?", (person_id,))
                con.execute("DELETE FROM events WHERE person_id=?", (person_id,))

            con.execute("DELETE FROM people WHERE id=?", (person_id,))
        return {"snapshots": snapshots, "tracks": tracks}

    def list_people(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT p.id, p.name, p.matricula, p.created_at,
                       COUNT(e.id) AS embeddings
                FROM people p
                LEFT JOIN embeddings e ON e.person_id = p.id
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def load_gallery(self) -> Gallery:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT e.person_id AS pid, e.vec AS vec, p.name AS name
                FROM embeddings e
                JOIN people p ON p.id = e.person_id
                """
            ).fetchall()
        if not rows:
            return Gallery()
        vecs, ids, names = [], [], {}
        for r in rows:
            vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
            ids.append(int(r["pid"]))
            names[int(r["pid"])] = r["name"]
        matrix = np.vstack(vecs).astype(np.float32)
        return Gallery(matrix=matrix, ids=ids, names=names)

    # ---- eventos --------------------------------------------------------------
    def add_event(self, person_id, name, score, snapshot_path, is_known) -> int:
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO events(person_id, name, score, ts, snapshot_path, is_known)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (person_id, name, float(score), time.time(), snapshot_path, 1 if is_known else 0),
            )
            return int(cur.lastrowid)

    def list_events(self, limit: int = 50) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- trilhas (pipeline de duas fases) -------------------------------------
    def add_track(self, started_at: float, ended_at: float, frames: int,
                  crops: list[dict]) -> int:
        """Grava uma trilha e seus recortes. `crops`: [{path, quality, face}].

        `face` é a linha de 15 valores do YuNet já em coordenadas do recorte,
        serializada em JSON — é o que permite ao lote refazer exatamente o mesmo
        alinhamento que o modo em tempo real faria.
        """
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO tracks(started_at, ended_at, frames, status)
                VALUES(?, ?, ?, 'pendente')
                """,
                (started_at, ended_at, frames),
            )
            track_id = int(cur.lastrowid)
            con.executemany(
                "INSERT INTO track_crops(track_id, path, quality, face) VALUES(?,?,?,?)",
                [(track_id, c["path"], float(c["quality"]), c["face"]) for c in crops],
            )
        return track_id

    def pending_tracks(self, limit: int = 500) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM tracks WHERE status = 'pendente'
                ORDER BY started_at LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def track_crops(self, track_id: int) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM track_crops WHERE track_id=? ORDER BY quality DESC",
                (track_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_track(self, track_id: int, person_id, name: str, score: float,
                      votes: str, status: str = "processado") -> None:
        with self._connect() as con:
            con.execute(
                """
                UPDATE tracks SET status=?, person_id=?, name=?, score=?, votes=?
                WHERE id=?
                """,
                (status, person_id, name, float(score), votes, track_id),
            )

    def reset_processed_tracks(self) -> int:
        """Devolve as trilhas processadas para 'pendente' (reprocessamento)."""
        with self._connect() as con:
            return con.execute(
                "UPDATE tracks SET status='pendente' WHERE status='processado'"
            ).rowcount or 0

    def count_tracks_by_status(self) -> dict:
        with self._connect() as con:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM tracks GROUP BY status"
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def attendance(self, day_start: float, day_end: float) -> list[dict]:
        """Presença do período: uma linha por pessoa identificada.

        Para chamada o que importa é "foi vista pelo menos uma vez", não quantas
        vezes passou — por isso agrupa por pessoa e devolve a primeira aparição.

        Une as DUAS origens possíveis, porque cada modo do worker grava em lugar
        diferente e ignorar uma delas devolveria chamada vazia:
          - modo captura  -> tabela `tracks` (reconhecimento em lote);
          - modo realtime -> tabela `events` (reconhecimento na hora).
        A coluna `fontes` diz de onde veio cada pessoa, o que ajuda a auditar
        quando os dois modos foram usados no mesmo dia.
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT person_id, name,
                       MIN(ts)                        AS primeira,
                       MAX(fim)                       AS ultima,
                       COUNT(*)                       AS passagens,
                       MAX(score)                     AS melhor_score,
                       GROUP_CONCAT(DISTINCT fonte)   AS fontes
                FROM (
                    SELECT person_id, name, started_at AS ts, ended_at AS fim,
                           score, 'captura' AS fonte
                    FROM tracks
                    WHERE status = 'processado' AND person_id IS NOT NULL
                      AND started_at >= ? AND started_at < ?
                    UNION ALL
                    SELECT person_id, name, ts, ts AS fim,
                           score, 'realtime' AS fonte
                    FROM events
                    WHERE is_known = 1 AND person_id IS NOT NULL
                      AND ts >= ? AND ts < ?
                )
                GROUP BY person_id
                ORDER BY primeira
                """,
                (day_start, day_end, day_start, day_end),
            ).fetchall()
        return [dict(r) for r in rows]

    def deteccoes_de(self, person_id=None, inicio: float = None,
                     fim: float = None, limit: int = 200) -> list[dict]:
        """Detecções das duas origens, com filtro opcional de pessoa e período.

        A aba de reconhecimentos lia só `events`, e por isso ficava vazia no
        modo captura — o modo que roda na escola. Aqui as duas origens vêm
        juntas, com `fonte` distinguindo, e cada uma já traz a foto e a base
        correta (elas ficam em diretórios diferentes).

        `person_id=None` traz todo mundo, inclusive os desconhecidos, que são
        justamente os que interessam quando se suspeita de falso negativo.
        """
        cond_t = ["t.status = 'processado'"]
        cond_e = ["1=1"]
        args_t, args_e = [], []
        if person_id is not None:
            cond_t.append("t.person_id = ?")
            cond_e.append("person_id = ?")
            args_t.append(int(person_id))
            args_e.append(int(person_id))
        if inicio is not None:
            cond_t.append("t.started_at >= ?")
            cond_e.append("ts >= ?")
            args_t.append(inicio)
            args_e.append(inicio)
        if fim is not None:
            cond_t.append("t.started_at < ?")
            cond_e.append("ts < ?")
            args_t.append(fim)
            args_e.append(fim)

        sql = f"""
            SELECT 'trilha' AS fonte, t.id, t.person_id, t.name, t.score,
                   t.started_at AS ts,
                   (SELECT path FROM track_crops c WHERE c.track_id = t.id
                     ORDER BY quality DESC LIMIT 1) AS foto,
                   CASE WHEN t.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_known
            FROM tracks t WHERE {' AND '.join(cond_t)}
            UNION ALL
            SELECT 'evento' AS fonte, id, person_id, name, score, ts,
                   snapshot_path AS foto, is_known
            FROM events WHERE {' AND '.join(cond_e)}
            ORDER BY ts DESC LIMIT ?
        """
        with self._connect() as con:
            rows = con.execute(sql, (*args_t, *args_e, int(limit))).fetchall()
        return [dict(r) for r in rows]

    # ---- rótulos por detecção -------------------------------------------------
    def set_detection_label(self, fonte: str, detection_id: int, rotulo: str,
                            autor: str = "") -> None:
        """Marca uma detecção como 'certo' ou 'errado'."""
        if rotulo not in ("certo", "errado"):
            raise ValueError("rotulo deve ser 'certo' ou 'errado'")
        if fonte not in ("evento", "trilha"):
            raise ValueError("fonte deve ser 'evento' ou 'trilha'")
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO detection_labels
                       (fonte, detection_id, rotulo, autor, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fonte, detection_id) DO UPDATE SET
                    rotulo=excluded.rotulo, autor=excluded.autor,
                    created_at=excluded.created_at
                """,
                (fonte, int(detection_id), rotulo, autor, time.time()),
            )

    def remove_detection_label(self, fonte: str, detection_id: int) -> int:
        with self._connect() as con:
            return con.execute(
                "DELETE FROM detection_labels WHERE fonte=? AND detection_id=?",
                (fonte, int(detection_id))).rowcount or 0

    def detection_labels(self) -> dict:
        """{"fonte:id": rotulo} — o formato que a calibração já usa como chave."""
        with self._connect() as con:
            return {f"{r['fonte']}:{r['detection_id']}": r["rotulo"]
                    for r in con.execute(
                        "SELECT fonte, detection_id, rotulo FROM detection_labels")}

    def importar_rotulos(self, rotulos: dict) -> int:
        """Importa rótulos no formato {"fonte:id": "certo"}, sem sobrescrever.

        Serve para migrar o arquivo JSON que o calibrate_threshold usava antes
        de os rótulos irem para o banco. `INSERT OR IGNORE` porque o que já
        está no banco é mais recente que o arquivo — se alguém já rotulou pelo
        painel, a migração não pode desfazer.
        """
        linhas = []
        for chave, rotulo in (rotulos or {}).items():
            fonte, _, ident = str(chave).partition(":")
            if not ident.isdigit() or rotulo not in ("certo", "errado"):
                continue
            if fonte not in ("evento", "trilha"):
                continue
            linhas.append((fonte, int(ident), rotulo, "migrado", time.time()))
        if not linhas:
            return 0
        with self._connect() as con:
            cur = con.executemany(
                """
                INSERT OR IGNORE INTO detection_labels
                       (fonte, detection_id, rotulo, autor, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, linhas)
            return cur.rowcount or 0

    def melhores_fotos(self, day_start: float, day_end: float) -> dict:
        """person_id -> a foto da detecção de MAIOR score no período.

        Existe para a conferência humana da chamada: sem ver o rosto, marcar
        "presente" é confirmar no escuro — e são essas confirmações que
        alimentam a calibração e a medição de recall. Conferência às cegas
        contamina as duas.

        Consulta separada de `attendance()` de propósito. Aquela usa `MIN(ts)`
        e `MAX(score)` juntos, e o SQLite só garante que as colunas avulsas
        venham da linha certa quando há UM agregado desses. Com dois, qual
        linha alimenta a foto seria indefinido — e o erro apareceria como uma
        foto de outra passagem, difícil de perceber e pior que não ter foto.

        Devolve `fonte` porque as duas origens guardam imagem em bases
        diferentes: `realtime` em snapshots/, `captura` em tracks/.
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT person_id, foto, fonte, score FROM (
                    SELECT person_id, score, 'captura' AS fonte,
                           (SELECT path FROM track_crops c
                             WHERE c.track_id = t.id
                             ORDER BY quality DESC LIMIT 1) AS foto
                    FROM tracks t
                    WHERE status = 'processado' AND person_id IS NOT NULL
                      AND started_at >= ? AND started_at < ?
                    UNION ALL
                    SELECT person_id, score, 'realtime' AS fonte,
                           snapshot_path AS foto
                    FROM events
                    WHERE is_known = 1 AND person_id IS NOT NULL
                      AND ts >= ? AND ts < ?
                )
                WHERE foto IS NOT NULL
                ORDER BY person_id, score DESC
                """,
                (day_start, day_end, day_start, day_end),
            ).fetchall()

        # Primeira linha de cada pessoa é a de maior score (ORDER BY acima).
        melhores: dict[int, dict] = {}
        for r in rows:
            pid = int(r["person_id"])
            if pid not in melhores:
                melhores[pid] = {"foto": r["foto"], "fonte": r["fonte"],
                                 "score": float(r["score"])}
        return melhores

    # ---- retenção das trilhas -------------------------------------------------
    def pending_tracks_before(self, ts: float) -> int:
        """Trilhas pendentes mais antigas que `ts`.

        Serve de trava de segurança na limpeza: trilha pendente antiga significa
        que o reconhecimento em lote parou de rodar. Apagar os recortes dela
        destruiria dado que ainda não foi aproveitado.
        """
        with self._connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM tracks WHERE status='pendente' AND started_at < ?",
                (ts,)).fetchone()[0])

    def drop_track_crops_before(self, ts: float) -> int:
        """Remove as LINHAS de recorte de trilhas já resolvidas antes de `ts`.

        Só as imagens: a linha em `tracks` fica, porque ela é o registro de
        presença. Recorte é evidência transitória; trilha é o dado.
        """
        with self._connect() as con:
            return con.execute(
                """
                DELETE FROM track_crops WHERE track_id IN (
                    SELECT id FROM tracks
                     WHERE status IN ('processado','descartado') AND started_at < ?
                )
                """, (ts,)).rowcount or 0

    def purge_tracks_before(self, ts: float) -> int:
        """Apaga as trilhas (o registro em si) anteriores a `ts`.

        Prazo mais longo que o dos recortes: aqui mora o histórico de presença.
        Pendentes são preservadas de propósito, para não perder o que o lote
        ainda não processou.
        """
        with self._connect() as con:
            con.execute(
                """
                DELETE FROM track_crops WHERE track_id IN (
                    SELECT id FROM tracks WHERE status != 'pendente' AND started_at < ?
                )
                """, (ts,))
            return con.execute(
                "DELETE FROM tracks WHERE status != 'pendente' AND started_at < ?",
                (ts,)).rowcount or 0

    # ---- correção manual da chamada -------------------------------------------
    def set_attendance_override(self, dia: str, person_id: int, presente: bool,
                                motivo: str = "", autor: str = "") -> None:
        """Registra que uma pessoa DISCORDOU do resultado automático naquele dia."""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO attendance_overrides
                       (dia, person_id, presente, motivo, autor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dia, person_id) DO UPDATE SET
                    presente=excluded.presente, motivo=excluded.motivo,
                    autor=excluded.autor, created_at=excluded.created_at
                """,
                (dia, person_id, 1 if presente else 0, motivo, autor, time.time()),
            )

    def remove_attendance_override(self, dia: str, person_id: int) -> int:
        """Desfaz a correção — volta a valer o que o reconhecimento disse."""
        with self._connect() as con:
            return con.execute(
                "DELETE FROM attendance_overrides WHERE dia=? AND person_id=?",
                (dia, person_id)).rowcount or 0

    def attendance_overrides(self, dia: str) -> dict:
        """person_id -> {presente, motivo, autor, created_at} do dia."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM attendance_overrides WHERE dia=?", (dia,)).fetchall()
        return {int(r["person_id"]): dict(r) for r in rows}

    def close_attendance(self, dia: str, autor: str, presentes: int,
                         correcoes: int) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO attendance_closures
                       (dia, closed_at, autor, presentes, correcoes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dia) DO UPDATE SET
                    closed_at=excluded.closed_at, autor=excluded.autor,
                    presentes=excluded.presentes, correcoes=excluded.correcoes
                """,
                (dia, time.time(), autor, presentes, correcoes),
            )

    def reopen_attendance(self, dia: str) -> int:
        with self._connect() as con:
            return con.execute(
                "DELETE FROM attendance_closures WHERE dia=?", (dia,)).rowcount or 0

    def attendance_closure(self, dia: str) -> dict | None:
        with self._connect() as con:
            r = con.execute("SELECT * FROM attendance_closures WHERE dia=?",
                            (dia,)).fetchone()
        return dict(r) if r else None

    # ---- leitura unificada para calibração ------------------------------------
    def list_detections(self, limit: int = 500) -> list[dict]:
        """Reconhecimentos das DUAS origens, num formato só.

        O modo realtime grava em `events`, o captura em `tracks`. Ler apenas uma
        delas fazia a calibração ficar cega justamente no modo usado em
        produção. A coluna `fonte` diz de onde veio cada linha, e `id` não é
        único entre as origens — a chave é o par (fonte, id).
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT 'evento' AS fonte, id, person_id, name, score, ts,
                       snapshot_path, is_known
                FROM events
                UNION ALL
                SELECT 'trilha' AS fonte, t.id, t.person_id, t.name, t.score,
                       t.started_at AS ts,
                       (SELECT path FROM track_crops c WHERE c.track_id = t.id
                         ORDER BY quality DESC LIMIT 1) AS snapshot_path,
                       CASE WHEN t.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_known
                FROM tracks t
                WHERE t.status = 'processado'
                ORDER BY ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def detections_between(self, inicio: float, fim: float) -> list[dict]:
        """Detecções de um intervalo, das duas origens e de TODOS os status.

        Difere de `list_detections()` em dois pontos, ambos exigidos pela medição
        de recall: filtra por tempo (a medição cruza janelas de passagem) e não
        descarta trilha com `status != 'processado'`. Trilha `descartado` é um
        degrau próprio do funil — o rosto foi detectado e rastreado, mas nenhum
        recorte ficou legível. Escondê-la faria essa falha ser contada como
        "nunca detectado", que tem causa e solução completamente diferentes.
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT 'evento' AS fonte, id, person_id, name, score, ts,
                       'processado' AS status, snapshot_path AS crop
                FROM events WHERE ts BETWEEN ? AND ?
                UNION ALL
                SELECT 'trilha' AS fonte, t.id, t.person_id, t.name, t.score,
                       t.started_at AS ts, t.status,
                       (SELECT path FROM track_crops c WHERE c.track_id = t.id
                         ORDER BY quality DESC LIMIT 1) AS crop
                FROM tracks t WHERE t.started_at BETWEEN ? AND ?
                ORDER BY ts
                """,
                (inicio, fim, inicio, fim),
            ).fetchall()
        return [dict(r) for r in rows]

    def closed_days(self) -> set:
        with self._connect() as con:
            return {r["dia"] for r in con.execute(
                "SELECT dia FROM attendance_closures")}

    def all_overrides(self) -> dict:
        """(dia, person_id) -> presente(0/1), de todas as chamadas."""
        with self._connect() as con:
            return {(r["dia"], int(r["person_id"])): int(r["presente"])
                    for r in con.execute(
                        "SELECT dia, person_id, presente FROM attendance_overrides")}

    def count_events(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def purge_events_before(self, ts: float, vacuum: bool = True) -> int:
        """Apaga eventos anteriores a `ts` e devolve quantos saíram.

        O VACUUM roda em conexão separada e em autocommit: o SQLite recusa
        VACUUM dentro de uma transação, e o `with con` abre uma implicitamente.
        """
        con = self._connect()
        try:
            cur = con.execute("DELETE FROM events WHERE ts < ?", (ts,))
            removed = cur.rowcount or 0
            con.commit()
        finally:
            con.close()

        if removed and vacuum:
            con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            try:
                con.execute("VACUUM")
            finally:
                con.close()
        return removed
