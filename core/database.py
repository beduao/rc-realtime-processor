"""Persistência em SQLite (modo WAL).

Tabelas:
  people      (id, name, created_at)
  embeddings  (id, person_id, vec BLOB)   -- bytes de numpy float32 (128,)
  events      (id, person_id, name, score, ts, snapshot_path, is_known)

WAL permite 1 escritor + N leitores em acesso local (worker grava, API lê),
que é exatamente o cenário deste projeto.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chamada import retirar_pessoa, tirar_fotos
from .config import project_path


# Nome usado nas linhas cujo titular foi removido com anonimização.
ANONIMO = "(removido)"

# Modos de `delete_person`, do que preserva mais para o que apaga tudo.
MODOS_EXCLUSAO = ("manter_historico", "anonimizar", "apagar")

class AlunoIdDuplicado(Exception):
    """Outro aluno já tem esse ID."""

    def __init__(self, person_id: int, nome: str):
        self.person_id, self.nome = person_id, nome
        super().__init__(f"ID de aluno já usado por '{nome}' (id {person_id}).")


def normalizar_aluno_id(valor) -> str | None:
    """Normaliza o ID do aluno: tira espaços das pontas e colapsa internos.

    **Nenhuma validação de formato, de propósito**: o formato do ID que o
    sistema de gestão emite não foi confirmado. Só se remove ruído de
    digitação — letra, hífen, barra e UUID sobrevivem. Validar formato que não
    se conhece transforma dado válido em dado rejeitado, e o erro aparece no
    pior momento: na hora de cadastrar um aluno de verdade. Já aconteceu uma
    vez neste projeto (ver MUDANCAS.md, 7.7).

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

                -- A chamada como foi CONFERIDA, congelada no fechamento.
                --
                -- Sem isto, um dia fechado continuava sendo recalculado a cada
                -- consulta, e mudava sem ninguém pedir: reprocessar o lote
                -- trocava quem aparecia presente, e a retenção de 30 dias
                -- apagava as trilhas — o dia voltava com todo mundo ausente e
                -- ainda marcado como conferido.
                --
                -- Cada fechamento é uma versão nova; reabrir e fechar de novo
                -- NÃO apaga a anterior. Quem consome vê pelo número que a
                -- chamada mudou depois de já ter sido entregue.
                --
                -- `payload` é o JSON de /attendance no instante do fechamento.
                -- Contém nomes e IDs de aluno: a retenção destas linhas ainda
                -- precisa ser decidida por quem responde por proteção de dados.
                CREATE TABLE IF NOT EXISTS attendance_versions (
                    dia        TEXT    NOT NULL,
                    versao     INTEGER NOT NULL,
                    closed_at  REAL    NOT NULL,
                    autor      TEXT,
                    payload    TEXT    NOT NULL,
                    PRIMARY KEY (dia, versao)
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
                -- `pessoa_correta` guarda QUEM de fato é, quando se sabe. A
                -- semântica, com rotulo='errado':
                --   pessoa_correta = N     -> é a pessoa N
                --   pessoa_correta = -1    -> não é ninguém cadastrado
                --   pessoa_correta = NULL  -> não foi informado de quem é
                --                             (todo rótulo anterior à coluna)
                -- "Ninguém cadastrado" era NULL no começo, o que o tornava
                -- indistinguível dos rótulos antigos: a tela lia NULL como
                -- "ninguém" e a calibração como "não informado".
                -- Isso transforma o rótulo de um binário em par
                -- (detecção -> identidade real), que é o que permite medir de
                -- verdade: sem o alvo, "errado" só diz o que NÃO é.
                CREATE TABLE IF NOT EXISTS detection_labels (
                    fonte          TEXT    NOT NULL,  -- 'evento' | 'trilha'
                    detection_id   INTEGER NOT NULL,
                    rotulo         TEXT    NOT NULL,  -- 'certo' | 'errado'
                    pessoa_correta INTEGER,
                    autor          TEXT,
                    created_at     REAL    NOT NULL,
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

            # Recorte LIMPO do rosto de cada evento.
            #
            # O `snapshot_path` do modo realtime guarda o frame ANOTADO — com
            # a caixa e o nome desenhados — porque é o que serve para uma
            # pessoa conferir quem foi reconhecido e em que contexto. Mas isso
            # o torna inútil como amostra de cadastro: o embedding sairia de
            # uma imagem com um retângulo sobre o rosto.
            #
            # O modo captura já guarda recorte limpo em `track_crops`. Esta
            # coluna dá ao realtime a mesma capacidade.
            eventos = {r["name"] for r in con.execute("PRAGMA table_info(events)")}
            if "crop_path" not in eventos:
                con.execute("ALTER TABLE events ADD COLUMN crop_path TEXT")

            # Rótulo ganhou o alvo da correção depois de existir.
            rot = {r["name"] for r in con.execute(
                "PRAGMA table_info(detection_labels)")}
            if rot and "pessoa_correta" not in rot:
                con.execute("ALTER TABLE detection_labels "
                            "ADD COLUMN pessoa_correta INTEGER")

            # Qual versão congelada o fechamento atual representa. Fechamento
            # anterior a esta coluna fica com NULL: não há o que servir
            # congelado, e a chamada daquele dia segue recalculada como antes.
            fech = {r["name"] for r in con.execute(
                "PRAGMA table_info(attendance_closures)")}
            if "versao" not in fech:
                con.execute("ALTER TABLE attendance_closures ADD COLUMN versao INTEGER")

            pessoas = {r["name"] for r in con.execute("PRAGMA table_info(people)")}
            if "matricula" not in pessoas:
                con.execute("ALTER TABLE people ADD COLUMN matricula TEXT")

            # Aluno REMOVIDO do cadastro, mas que continua no histórico de
            # chamadas: a linha em `people` fica (nome e `aluno_id`), sem
            # biometria nem foto. NULL = ativo. É o que o modo de exclusão
            # padrão faz — ver `delete_person`.
            if "removido_em" not in pessoas:
                con.execute("ALTER TABLE people ADD COLUMN removido_em REAL")

            # ID do aluno emitido pelo sistema de gestão: a coluna viva de um
            # campo que antes foi `inep_id` e depois `matricula`. A matrícula
            # saiu porque carrega o ano ("2026/0142") e muda na virada — uma
            # chave de integração que muda em janeiro quebra todo o cadastro
            # biométrico feito até dezembro.
            #
            # As colunas anteriores (`inep_id`, `matricula`) NÃO são removidas
            # aqui: o dado já digitado continua intacto, e remover coluna é
            # destrutivo. Ficam dormentes — nada lê nem escreve nelas.
            if "aluno_id" not in pessoas:
                con.execute("ALTER TABLE people ADD COLUMN aluno_id TEXT")
                # Copia o que já estava digitado. No piloto esses valores
                # foram preenchidos à mão como identificador, e perdê-los em
                # silêncio seria pior do que trazê-los: quando os IDs reais
                # chegarem, é só sobrescrever pela tela de Pessoas.
                con.execute("UPDATE people SET aluno_id = matricula "
                            "WHERE matricula IS NOT NULL AND aluno_id IS NULL")

            # Índice único PARCIAL: dois alunos não podem compartilhar o ID,
            # mas vários podem estar sem preencher (sem o `IS NOT NULL`, o
            # segundo NULL colidiria). E único só entre os ATIVOS: com o aluno
            # removido mantendo o ID no histórico, um índice entre todos
            # impediria recadastrar a mesma criança. Trocar ou derrubar índice
            # não apaga dado nenhum — índice é só estrutura de busca.
            con.execute("DROP INDEX IF EXISTS idx_people_matricula")
            con.execute("DROP INDEX IF EXISTS idx_people_matricula_ativa")
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_people_aluno_id_ativo "
                "ON people(aluno_id) "
                "WHERE aluno_id IS NOT NULL AND removido_em IS NULL")

    # ---- pessoas / embeddings -------------------------------------------------
    def person_by_aluno_id(self, aluno_id: str) -> dict | None:
        alvo = normalizar_aluno_id(aluno_id)
        if not alvo:
            return None
        # Só entre os ativos: aluno removido guarda o ID no histórico, mas ele
        # fica livre para um recadastro.
        with self._connect() as con:
            r = con.execute(
                "SELECT id, name, aluno_id, created_at FROM people "
                "WHERE aluno_id=? AND removido_em IS NULL", (alvo,)).fetchone()
        return dict(r) if r else None

    def set_person_aluno_id(self, person_id: int, aluno_id) -> str | None:
        """Define (ou limpa) o ID do aluno. Levanta AlunoIdDuplicado se for de outro."""
        alvo = normalizar_aluno_id(aluno_id)
        if alvo:
            dono = self.person_by_aluno_id(alvo)
            if dono and dono["id"] != person_id:
                raise AlunoIdDuplicado(dono["id"], dono["name"])
        with self._connect() as con:
            con.execute("UPDATE people SET aluno_id=? WHERE id=?",
                        (alvo, person_id))
        return alvo

    def add_person(self, name: str, aluno_id=None) -> int:
        alvo = normalizar_aluno_id(aluno_id)
        if alvo:
            dono = self.person_by_aluno_id(alvo)
            if dono:
                raise AlunoIdDuplicado(dono["id"], dono["name"])
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO people(name, created_at, aluno_id) VALUES(?, ?, ?)",
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

    def get_person(self, person_id: int,
                   incluir_removidos: bool = False) -> dict | None:
        """A pessoa, ou None. Removido do cadastro conta como inexistente,
        salvo com `incluir_removidos` — que é o que a chamada de um dia antigo
        precisa para corrigir a presença de quem estava matriculado então."""
        sql = ("SELECT id, name, aluno_id, created_at, removido_em FROM people "
               "WHERE id=?")
        if not incluir_removidos:
            sql += " AND removido_em IS NULL"
        with self._connect() as con:
            r = con.execute(sql, (person_id,)).fetchone()
        return dict(r) if r else None

    def rename_person(self, person_id: int, name: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))

    def delete_person(self, person_id: int, anonimizar: bool = False,
                      modo: str = None) -> dict:
        """Remove a pessoa. Devolve os arquivos de imagem a apagar do disco.

        Três modos (`MODOS_EXCLUSAO`). Em TODOS saem as imagens do rosto e os
        embeddings — a biometria. O que muda é o que acontece com o histórico:

          "manter_historico"  aluno sai do cadastro e deixa de ser reconhecido,
                              mas nome e `aluno_id` continuam nas chamadas,
                              como presente ou ausente, nos dias em que ele
                              estava matriculado. A chamada é registro escolar;
                              a foto e o vetor do rosto é que são o dado
                              sensível. É o modo padrão da API.
          "anonimizar"        as passagens ficam, sem identificar quem.
          "apagar"            some tudo, inclusive o histórico — para um pedido
                              de exclusão completa.

        Sem `modo`, vale o parâmetro antigo `anonimizar` (False = "apagar"),
        para quem já chamava assim continuar com o mesmo efeito.

        Quem chama é responsável por remover os arquivos — o banco não conhece o
        sistema de arquivos. Devolve os caminhos por base, já que as imagens
        moram em dois lugares:
            {"snapshots": [...],   # relativos a storage.snapshots_dir
             "tracks":    [...]}   # relativos a tracking.crops_dir
        """
        modo = modo or ("anonimizar" if anonimizar else "apagar")
        if modo not in MODOS_EXCLUSAO:
            raise ValueError(f"modo de exclusão inválido: {modo!r}")
        anonimizar = modo == "anonimizar"
        with self._connect() as con:
            # `crop_path` entra junto: é o recorte LIMPO do rosto que o modo
            # realtime grava ao lado do snapshot. Ficou de fora quando a coluna
            # foi criada, e a exclusão passou a deixar exatamente o dado mais
            # identificante — o rosto sem moldura — para trás no disco.
            snapshots = [r["p"] for r in con.execute(
                """
                SELECT snapshot_path AS p FROM embeddings
                 WHERE person_id=? AND snapshot_path IS NOT NULL
                UNION ALL
                SELECT snapshot_path AS p FROM events
                 WHERE person_id=? AND snapshot_path IS NOT NULL
                UNION ALL
                SELECT crop_path AS p FROM events
                 WHERE person_id=? AND crop_path IS NOT NULL
                """, (person_id, person_id, person_id))]
            tracks = [r["path"] for r in con.execute(
                """
                SELECT tc.path FROM track_crops tc
                JOIN tracks t ON t.id = tc.track_id
                WHERE t.person_id = ?
                """, (person_id,))]

            # embeddings saem sempre — sem eles não há reconhecimento
            con.execute("DELETE FROM embeddings WHERE person_id=?", (person_id,))

            if modo == "manter_historico":
                # Só a imagem sai. Eventos e trilhas ficam com nome e id, e
                # é deles que a chamada de um dia ainda aberto se recalcula;
                # correções de presença e rótulos também ficam, porque são
                # parte desse mesmo histórico.
                con.execute(
                    "UPDATE events SET snapshot_path=NULL, crop_path=NULL "
                    "WHERE person_id=?", (person_id,))
                con.execute(
                    "DELETE FROM track_crops WHERE track_id IN "
                    "(SELECT id FROM tracks WHERE person_id=?)", (person_id,))
                for r in con.execute(
                        "SELECT dia, versao, payload FROM attendance_versions"
                        ).fetchall():
                    payload = json.loads(r["payload"])
                    if tirar_fotos(payload, person_id):
                        con.execute(
                            "UPDATE attendance_versions SET payload=? "
                            "WHERE dia=? AND versao=?",
                            (json.dumps(payload, ensure_ascii=False),
                             r["dia"], r["versao"]))
                con.execute("UPDATE people SET removido_em=? WHERE id=?",
                            (time.time(), person_id))
                return {"snapshots": snapshots, "tracks": tracks}

            # Correções de presença dela: guardam motivo e autor ligados à
            # pessoa, e sem a pessoa não alimentam mais nada.
            con.execute("DELETE FROM attendance_overrides WHERE person_id=?",
                        (person_id,))

            # Rótulos das detecções DELA saem nos dois modos: um rótulo é uma
            # afirmação sobre a identidade de alguém. Precisa rodar antes de
            # mexer em events/tracks, enquanto o person_id ainda as localiza.
            con.execute(
                """
                DELETE FROM detection_labels
                 WHERE (fonte = 'evento' AND detection_id IN
                          (SELECT id FROM events WHERE person_id = ?))
                    OR (fonte = 'trilha' AND detection_id IN
                          (SELECT id FROM tracks WHERE person_id = ?))
                """, (person_id, person_id))
            # Detecções de OUTROS que foram corrigidas para "era esta pessoa"
            # passam a "não é ninguém cadastrado" — o que, removida a pessoa,
            # é literalmente verdade. Continuam contando como erro do sistema.
            con.execute(
                "UPDATE detection_labels SET pessoa_correta=? "
                "WHERE pessoa_correta=?", (self.DESCONHECIDO, person_id))

            if anonimizar:
                con.execute(
                    "UPDATE events SET person_id=NULL, name=?, snapshot_path=NULL, "
                    "crop_path=NULL, is_known=0 WHERE person_id=?",
                    (ANONIMO, person_id))
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

            # Chamadas congeladas: a exclusão vale para elas também. É a única
            # mudança que uma versão fechada aceita sem ser reaberta.
            for r in con.execute(
                    "SELECT dia, versao, payload FROM attendance_versions").fetchall():
                payload = json.loads(r["payload"])
                if retirar_pessoa(payload, person_id, anonimizar, ANONIMO):
                    con.execute(
                        "UPDATE attendance_versions SET payload=? "
                        "WHERE dia=? AND versao=?",
                        (json.dumps(payload, ensure_ascii=False),
                         r["dia"], r["versao"]))

            con.execute("DELETE FROM people WHERE id=?", (person_id,))
        return {"snapshots": snapshots, "tracks": tracks}

    def list_people(self, incluir_removidos: bool = False) -> list[dict]:
        """Cadastrados. Com `incluir_removidos`, também quem saiu do cadastro
        mas continua no histórico — a chamada de dias antigos precisa deles."""
        onde = "" if incluir_removidos else "WHERE p.removido_em IS NULL"
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT p.id, p.name, p.aluno_id, p.created_at, p.removido_em,
                       COUNT(e.id) AS embeddings
                FROM people p
                LEFT JOIN embeddings e ON e.person_id = p.id
                {onde}
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def load_gallery(self) -> Gallery:
        # O filtro de removido é redundante (os embeddings dele já foram
        # apagados) e fica de propósito: removido NUNCA pode ser reconhecido.
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT e.person_id AS pid, e.vec AS vec, p.name AS name
                FROM embeddings e
                JOIN people p ON p.id = e.person_id
                WHERE p.removido_em IS NULL
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
    def add_event(self, person_id, name, score, snapshot_path, is_known,
                  ts: float = None, crop_path: str = None) -> int:
        """Registra um reconhecimento. `ts` é o instante da OBSERVAÇÃO.

        Quando omitido usa a hora do insert, o que embute a latência do
        processamento — embedding, anotação e gravação do JPEG — no horário
        registrado. São centenas de milissegundos que:

          - deslocam o horário da passagem, que é o dado que a chamada expõe;
          - fazem o intervalo entre eventos gravados ficar MENOR que o
            `event_cooldown_seconds`, porque o cooldown é medido no início do
            frame e o `ts` no fim. Era a causa da instabilidade do teste do
            worker: intervalos de 0.28 s com cooldown de 0.3 s.

        O worker passa o instante do frame. O padrão continua existindo para
        quem grava fora do laço (testes, scripts).
        """
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO events(person_id, name, score, ts, snapshot_path,
                                   is_known, crop_path)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (person_id, name, float(score),
                 time.time() if ts is None else float(ts),
                 snapshot_path, 1 if is_known else 0, crop_path),
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

    def count_pending_tracks(self, inicio: float = None, fim: float = None) -> int:
        """Trilhas pendentes, opcionalmente só as que começaram em [inicio, fim).

        A chamada de um dia só depende das trilhas DAQUELE dia. Contar as
        pendentes do banco inteiro fazia o fechamento ser recusado durante a
        aula inteira: no modo captura sempre há alguém que acabou de passar e
        ainda espera o lote, que roda a cada 10 minutos.
        """
        cond, args = ["status = 'pendente'"], []
        if inicio is not None:
            cond.append("started_at >= ?")
            args.append(inicio)
        if fim is not None:
            cond.append("started_at < ?")
            args.append(fim)
        with self._connect() as con:
            return int(con.execute(
                f"SELECT COUNT(*) FROM tracks WHERE {' AND '.join(cond)}",
                args).fetchone()[0])

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

    # Valor de `person_id` que seleciona quem NÃO é ninguém cadastrado. É
    # também o que `detection_labels.pessoa_correta` guarda para "não é
    # ninguém cadastrado" — o NULL ficou reservado para "não informado".
    DESCONHECIDO = -1

    def deteccoes_de(self, person_id=None, inicio: float = None,
                     fim: float = None, limit: int = 200) -> list[dict]:
        """Detecções das duas origens, filtráveis pela identidade CORRIGIDA.

        A aba de reconhecimentos lia só `events`, e por isso ficava vazia no
        modo captura — o modo que roda na escola. Aqui as duas origens vêm
        juntas, com `fonte` distinguindo, e cada uma já traz a foto.

        **O filtro usa a identidade efetiva, não a que o sistema afirmou.**
        Quando alguém marca "não é essa pessoa, é a fulana", a detecção sai da
        lista de quem o sistema disse e entra na da fulana. Sem isso, marcar
        erro não limpava nada: o rosto errado continuava aparecendo sob o nome
        errado, e a revisão não convergia nunca.

            efetivo = pessoa_correta, quando o rótulo diz 'errado'
                      person_id do sistema, caso contrário

        `person_id=None` traz todos. `DESCONHECIDO` traz quem ficou sem
        identidade — os não reconhecidos pelo sistema, os marcados como "não é
        ninguém cadastrado" e os marcados errados sem dizer de quem eram. Esse
        balde é o que permite procurar falso negativo: quem está lá deveria
        estar em algum nome. A coluna `pessoa_correta` distingue os casos.
        """
        cond, args = [], []
        if inicio is not None:
            cond.append("ts >= ?")
            args.append(inicio)
        if fim is not None:
            cond.append("ts < ?")
            args.append(fim)
        if person_id is not None:
            if int(person_id) == self.DESCONHECIDO:
                cond.append("efetivo IS NULL")
            else:
                cond.append("efetivo = ?")
                args.append(int(person_id))
        onde = ("WHERE " + " AND ".join(cond)) if cond else ""

        # O LEFT JOIN traz o rótulo de cada detecção; `efetivo` é calculado na
        # subconsulta para poder ser filtrado e ordenado do lado de fora. O
        # LIMIT fica DEPOIS do filtro — aplicá-lo antes devolveria menos linhas
        # que o pedido sempre que houvesse correção no meio.
        sql = f"""
            SELECT * FROM (
                SELECT 'trilha' AS fonte, t.id, t.person_id, t.name, t.score,
                       t.started_at AS ts,
                       (SELECT path FROM track_crops c WHERE c.track_id = t.id
                         ORDER BY quality DESC LIMIT 1) AS foto,
                       CASE WHEN t.person_id IS NOT NULL THEN 1 ELSE 0 END
                           AS is_known,
                       l.rotulo, l.pessoa_correta,
                       CASE WHEN l.rotulo = 'errado'
                            THEN NULLIF(l.pessoa_correta, {self.DESCONHECIDO})
                            ELSE t.person_id END AS efetivo
                FROM tracks t
                LEFT JOIN detection_labels l
                       ON l.fonte = 'trilha' AND l.detection_id = t.id
                WHERE t.status = 'processado'
                UNION ALL
                SELECT 'evento' AS fonte, e.id, e.person_id, e.name, e.score,
                       e.ts, e.snapshot_path AS foto, e.is_known,
                       l.rotulo, l.pessoa_correta,
                       CASE WHEN l.rotulo = 'errado'
                            THEN NULLIF(l.pessoa_correta, {self.DESCONHECIDO})
                            ELSE e.person_id END AS efetivo
                FROM events e
                LEFT JOIN detection_labels l
                       ON l.fonte = 'evento' AND l.detection_id = e.id
            )
            {onde}
            ORDER BY ts DESC LIMIT ?
        """
        with self._connect() as con:
            rows = con.execute(sql, (*args, int(limit))).fetchall()
        return [dict(r) for r in rows]

    # ---- rótulos por detecção -------------------------------------------------
    def set_detection_label(self, fonte: str, detection_id: int, rotulo: str,
                            autor: str = "", pessoa_correta=None) -> None:
        """Marca uma detecção como 'certo' ou 'errado'.

        `pessoa_correta` só faz sentido com 'errado':
            N     -> é a pessoa N
            -1    -> não é ninguém cadastrado (`DESCONHECIDO`)
            None  -> errado, mas sem dizer de quem é
        Com 'certo' é forçado a NULL: guardar alvo num acerto criaria um estado
        contraditório esperando para confundir alguém.
        """
        if rotulo not in ("certo", "errado"):
            raise ValueError("rotulo deve ser 'certo' ou 'errado'")
        if fonte not in ("evento", "trilha"):
            raise ValueError("fonte deve ser 'evento' ou 'trilha'")
        alvo = None
        if rotulo == "errado" and pessoa_correta is not None:
            alvo = int(pessoa_correta)
            if alvo != self.DESCONHECIDO and self.get_person(alvo) is None:
                raise ValueError(f"pessoa {alvo} não existe")
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO detection_labels
                       (fonte, detection_id, rotulo, pessoa_correta, autor,
                        created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fonte, detection_id) DO UPDATE SET
                    rotulo=excluded.rotulo,
                    pessoa_correta=excluded.pessoa_correta,
                    autor=excluded.autor, created_at=excluded.created_at
                """,
                (fonte, int(detection_id), rotulo, alvo, autor, time.time()),
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

    def detections_by_keys(self, chaves) -> list[dict]:
        """Busca detecções específicas por "fonte:id", fora de qualquer janela.

        `list_detections` devolve as N mais recentes, o que serve para o
        histograma mas descarta rótulo antigo. E rótulo é caro: cada um é
        alguém olhando uma foto e decidindo. Perder isso porque o worker
        gerou mais eventos depois é inaceitável — foi o que aconteceu, com
        211 marcações ignoradas em silêncio.
        """
        por_fonte = {"evento": [], "trilha": []}
        for c in chaves:
            fonte, _, ident = str(c).partition(":")
            if fonte in por_fonte and ident.isdigit():
                por_fonte[fonte].append(int(ident))
        if not por_fonte["evento"] and not por_fonte["trilha"]:
            return []

        linhas = []
        with self._connect() as con:
            if por_fonte["evento"]:
                marcas = ",".join("?" * len(por_fonte["evento"]))
                linhas += [dict(r) for r in con.execute(
                    f"""SELECT 'evento' AS fonte, id, person_id, name, score, ts,
                               snapshot_path, is_known
                        FROM events WHERE id IN ({marcas})""",
                    por_fonte["evento"])]
            if por_fonte["trilha"]:
                marcas = ",".join("?" * len(por_fonte["trilha"]))
                linhas += [dict(r) for r in con.execute(
                    f"""SELECT 'trilha' AS fonte, t.id, t.person_id, t.name,
                               t.score, t.started_at AS ts,
                               (SELECT path FROM track_crops c
                                 WHERE c.track_id = t.id
                                 ORDER BY quality DESC LIMIT 1) AS snapshot_path,
                               CASE WHEN t.person_id IS NOT NULL THEN 1 ELSE 0 END
                                   AS is_known
                        FROM tracks t WHERE t.id IN ({marcas})""",
                    por_fonte["trilha"])]
        return linhas

    def detection_labels_detalhados(self) -> dict:
        """{"fonte:id": {rotulo, pessoa_correta, nome_correto, autor}}.

        A versão simples basta para calcular o limiar, que só precisa do
        binário. Esta existe para responder "de quem ERA o rosto", que é o que
        diz quais cadastros estão sendo confundidos entre si — informação mais
        acionável que o limiar, porque aponta a pessoa a recadastrar.
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT l.fonte, l.detection_id, l.rotulo, l.pessoa_correta,
                       l.autor, p.name AS nome_correto
                FROM detection_labels l
                LEFT JOIN people p ON p.id = l.pessoa_correta
                """).fetchall()
        return {f"{r['fonte']}:{r['detection_id']}": dict(r) for r in rows}

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

        Detecção marcada como 'errado' é EXCLUÍDA: mostrar como "melhor
        captura" de alguém uma foto que uma pessoa já declarou não ser dela
        seria contradizer a própria conferência.
        """
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT person_id, foto, fonte, score FROM (
                    SELECT t.person_id, t.score, 'captura' AS fonte,
                           (SELECT path FROM track_crops c
                             WHERE c.track_id = t.id
                             ORDER BY quality DESC LIMIT 1) AS foto,
                           l.rotulo
                    FROM tracks t
                    LEFT JOIN detection_labels l
                           ON l.fonte = 'trilha' AND l.detection_id = t.id
                    WHERE t.status = 'processado' AND t.person_id IS NOT NULL
                      AND t.started_at >= ? AND t.started_at < ?
                    UNION ALL
                    SELECT e.person_id, e.score, 'realtime' AS fonte,
                           e.snapshot_path AS foto, l.rotulo
                    FROM events e
                    LEFT JOIN detection_labels l
                           ON l.fonte = 'evento' AND l.detection_id = e.id
                    WHERE e.is_known = 1 AND e.person_id IS NOT NULL
                      AND e.ts >= ? AND e.ts < ?
                )
                WHERE foto IS NOT NULL AND (rotulo IS NULL OR rotulo <> 'errado')
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
                         correcoes: int, payload: dict = None) -> int | None:
        """Fecha o dia. Com `payload`, congela a chamada numa versão nova.

        Devolve o número da versão (None sem payload). Número e gravação saem
        da mesma transação IMMEDIATE: dois fechamentos simultâneos não
        disputam o mesmo número.
        """
        agora = time.time()
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            versao = None
            if payload is not None:
                versao = int(con.execute(
                    "SELECT COALESCE(MAX(versao), 0) + 1 FROM attendance_versions "
                    "WHERE dia=?", (dia,)).fetchone()[0])
                con.execute(
                    "INSERT INTO attendance_versions "
                    "(dia, versao, closed_at, autor, payload) VALUES (?,?,?,?,?)",
                    (dia, versao, agora, autor,
                     json.dumps(payload, ensure_ascii=False)))
            con.execute(
                """
                INSERT INTO attendance_closures
                       (dia, closed_at, autor, presentes, correcoes, versao)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dia) DO UPDATE SET
                    closed_at=excluded.closed_at, autor=excluded.autor,
                    presentes=excluded.presentes, correcoes=excluded.correcoes,
                    versao=excluded.versao
                """,
                (dia, agora, autor, presentes, correcoes, versao),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        return versao

    def reopen_attendance(self, dia: str) -> int:
        """Reabre o dia. As versões congeladas ficam: são o histórico."""
        with self._connect() as con:
            return con.execute(
                "DELETE FROM attendance_closures WHERE dia=?", (dia,)).rowcount or 0

    def attendance_version(self, dia: str, versao: int = None) -> dict | None:
        """Uma versão congelada (a mais recente, sem `versao`), com o payload."""
        sql = "SELECT * FROM attendance_versions WHERE dia=? "
        args = [dia]
        if versao is None:
            sql += "ORDER BY versao DESC LIMIT 1"
        else:
            sql += "AND versao=?"
            args.append(int(versao))
        with self._connect() as con:
            r = con.execute(sql, args).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def ultima_versao(self, dia: str) -> int | None:
        with self._connect() as con:
            v = con.execute("SELECT MAX(versao) FROM attendance_versions WHERE dia=?",
                            (dia,)).fetchone()[0]
        return int(v) if v is not None else None

    def attendance_versions(self, dia: str) -> list[dict]:
        """Todas as versões do dia, da primeira à última, com o payload."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM attendance_versions WHERE dia=? ORDER BY versao",
                (dia,)).fetchall()
        saida = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            saida.append(d)
        return saida

    def tem_registros_brutos(self, inicio: float, fim: float) -> bool:
        """Sobrou alguma trilha ou evento no intervalo?

        Falso para um dia que já existiu é sinal de que a retenção apagou os
        dados brutos — e aí a única fonte do que o sistema viu naquele dia é
        a chamada congelada.
        """
        with self._connect() as con:
            return bool(con.execute(
                """
                SELECT EXISTS(SELECT 1 FROM tracks
                               WHERE started_at >= ? AND started_at < ?)
                    OR EXISTS(SELECT 1 FROM events WHERE ts >= ? AND ts < ?)
                """, (inicio, fim, inicio, fim)).fetchone()[0])

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
            self.vacuum()
        return removed

    def vacuum(self) -> None:
        """Compacta o arquivo do banco, devolvendo ao disco o espaço apagado.

        Método próprio porque a limpeza apaga eventos E trilhas e só depois
        quer compactar. Antes ela pedia o VACUUM chamando de novo
        `purge_events_before`, que na segunda vez não apagava nada — e sem
        nada apagado o VACUUM não rodava. Nunca rodou.
        """
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            con.execute("VACUUM")
        finally:
            con.close()
