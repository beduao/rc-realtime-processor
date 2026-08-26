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

# Quantidade de dígitos esperada na identificação única do Censo Escolar.
# É só para AVISAR — nunca para rejeitar. Se a realidade do dado for outra,
# o cadastro precisa funcionar mesmo assim.
INEP_DIGITOS = 12


class InepDuplicado(Exception):
    """Outro aluno já tem esse ID INEP."""

    def __init__(self, person_id: int, nome: str):
        self.person_id, self.nome = person_id, nome
        super().__init__(f"ID INEP já usado por '{nome}' (id {person_id}).")


def normalizar_inep(valor) -> str | None:
    """Limpa o ID INEP: tira espaços, pontos e traços. Vazio vira None.

    Normalizar na entrada evita o problema clássico de o mesmo aluno entrar
    duas vezes porque alguém digitou com ponto e outra pessoa sem.
    """
    if valor is None:
        return None
    limpo = "".join(c for c in str(valor) if c.isdigit())
    return limpo or None


def inep_suspeito(valor: str) -> str | None:
    """Devolve um aviso se o valor não parecer um ID INEP — ou None se parecer.

    Aviso, não erro: eu não tenho certeza absoluta do formato oficial, então
    bloquear seria pior que sinalizar.
    """
    if not valor:
        return None
    if len(valor) != INEP_DIGITOS:
        return (f"tem {len(valor)} dígitos; o esperado no Censo Escolar é "
                f"{INEP_DIGITOS}. Confira se copiou o campo certo.")
    return None


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

            # ID INEP do aluno: identificador do Censo Escolar, usado pelo
            # sistema de gestão da escola. Guardado como TEXT, não INTEGER —
            # é identificador, não quantidade, e zero à esquerda tem que
            # sobreviver. Índice único PARCIAL: dois alunos não podem
            # compartilhar o mesmo ID, mas vários podem estar sem preencher.
            pessoas = {r["name"] for r in con.execute("PRAGMA table_info(people)")}
            if "inep_id" not in pessoas:
                con.execute("ALTER TABLE people ADD COLUMN inep_id TEXT")
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_people_inep "
                "ON people(inep_id) WHERE inep_id IS NOT NULL")

    # ---- pessoas / embeddings -------------------------------------------------
    def person_by_inep(self, inep_id: str) -> dict | None:
        alvo = normalizar_inep(inep_id)
        if not alvo:
            return None
        with self._connect() as con:
            r = con.execute(
                "SELECT id, name, inep_id, created_at FROM people WHERE inep_id=?",
                (alvo,)).fetchone()
        return dict(r) if r else None

    def set_person_inep(self, person_id: int, inep_id) -> str | None:
        """Define (ou limpa) o ID INEP. Levanta InepDuplicado se já for de outro."""
        alvo = normalizar_inep(inep_id)
        if alvo:
            dono = self.person_by_inep(alvo)
            if dono and dono["id"] != person_id:
                raise InepDuplicado(dono["id"], dono["name"])
        with self._connect() as con:
            con.execute("UPDATE people SET inep_id=? WHERE id=?", (alvo, person_id))
        return alvo

    def add_person(self, name: str, inep_id=None) -> int:
        alvo = normalizar_inep(inep_id)
        if alvo:
            dono = self.person_by_inep(alvo)
            if dono:
                raise InepDuplicado(dono["id"], dono["name"])
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO people(name, created_at, inep_id) VALUES(?, ?, ?)",
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
                "SELECT id, name, inep_id, created_at FROM people WHERE id=?",
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
                SELECT p.id, p.name, p.inep_id, p.created_at, COUNT(e.id) AS embeddings
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
