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
                """
            )

    # ---- pessoas / embeddings -------------------------------------------------
    def add_person(self, name: str) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO people(name, created_at) VALUES(?, ?)",
                (name, time.time()),
            )
            return int(cur.lastrowid)

    def add_embedding(self, person_id: int, vec) -> None:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self._connect() as con:
            con.execute(
                "INSERT INTO embeddings(person_id, vec) VALUES(?, ?)",
                (person_id, blob),
            )

    def delete_person(self, person_id: int) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM embeddings WHERE person_id=?", (person_id,))
            con.execute("DELETE FROM people WHERE id=?", (person_id,))

    def list_people(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT p.id, p.name, p.created_at, COUNT(e.id) AS embeddings
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
