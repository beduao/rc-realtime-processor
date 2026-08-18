"""Fase 2: reconhece as trilhas capturadas pelo worker em modo `captura`.

Por que em lote é MELHOR, e não apenas mais barato
--------------------------------------------------
No modo em tempo real cada rosto é julgado por um único frame. Se naquele
instante a pessoa estava de perfil ou borrada, o embedding sai ruim e o
resultado é aleatório — foi assim que um rosto errado passou com score 0,37.

Aqui cada pessoa chega com vários recortes, já escolhidos por nitidez. Cada um
é reconhecido separadamente e o resultado sai por **votação**: só vira presença
quem ganhou a maioria dos votos válidos. Um recorte ruim é voto vencido, não
decisão final.

Uso:
    python scripts/recognize_batch.py                 # processa o que está pendente
    python scripts/recognize_batch.py --limite 50      # em blocos
    python scripts/recognize_batch.py --presenca        # relatório do dia
    python scripts/recognize_batch.py --presenca --dia 2026-08-03
    python scripts/recognize_batch.py --reprocessar     # refaz tudo (ex.: após novo cadastro)
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.config import load_config, project_path  # noqa: E402
from core.database import Database  # noqa: E402
from core.face_engine import FaceEngine  # noqa: E402


def _crops_dir(cfg):
    tcfg = cfg.get("tracking") or {}
    return project_path(tcfg.get("crops_dir", "data/tracks"))


def processar_trilha(engine, gallery, base, trilha, crops, limiar, min_votos):
    """Reconhece uma trilha por votação entre seus recortes."""
    votos = []          # (person_id ou None, score)
    for c in crops:
        caminho = base / c["path"]
        img = cv2.imread(str(caminho))
        if img is None:
            continue
        try:
            face = np.asarray(json.loads(c["face"]), dtype=np.float32)
            vec = engine.embed(img, face)
            pid, score = engine.match(vec, gallery.matrix, gallery.ids)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [erro] recorte {c['path']}: {type(exc).__name__}: {exc}")
            continue
        votos.append((pid if (pid is not None and score >= limiar) else None,
                      float(score)))

    if not votos:
        return None, "Sem recorte legível", 0.0, votos, "descartado"

    validos = [(pid, s) for pid, s in votos if pid is not None]
    if not validos:
        melhor = max(s for _, s in votos)
        return None, "Desconhecido", melhor, votos, "processado"

    contagem = Counter(pid for pid, _ in validos)
    pid_vencedor, n_votos = contagem.most_common(1)[0]
    if n_votos < min_votos:
        melhor = max(s for _, s in votos)
        return None, "Desconhecido", melhor, votos, "processado"

    scores = [s for pid, s in validos if pid == pid_vencedor]
    # score final = média dos votos vencedores (mais estável que o máximo, que
    # premiaria um único acerto sortudo)
    return (pid_vencedor, gallery.names.get(pid_vencedor, "?"),
            sum(scores) / len(scores), votos, "processado")


def reconhecer(cfg, db, limite: int, reprocessar: bool) -> int:
    # Checagens baratas primeiro: carregar o SFace custa ~37 MB e alguns
    # segundos no Pi, não faz sentido pagar isso para descobrir que não há
    # galeria ou nada pendente.
    gallery = db.load_gallery()
    if not gallery.ids:
        print("Nenhuma pessoa cadastrada — não há com o que comparar.")
        print("Cadastre pelo painel e rode de novo (as trilhas ficam pendentes,")
        print("nada é perdido).")
        return 1

    if not reprocessar and not db.pending_tracks(limit=1):
        print("Nada pendente.")
        return 0

    try:
        engine = FaceEngine(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Não consegui carregar os modelos:\n\n{exc}")
        return 1

    base = _crops_dir(cfg)
    limiar = engine.cosine_threshold
    min_votos = int((cfg.get("batch") or {}).get("min_votos", 2))

    if reprocessar:
        n = db.reset_processed_tracks()
        print(f"{n} trilha(s) marcada(s) para reprocessamento.")

    pendentes = db.pending_tracks(limit=limite)
    if not pendentes:
        print("Nada pendente.")
        return 0

    print(f"Reconhecendo {len(pendentes)} trilha(s) | limiar={limiar} "
          f"| mínimo de votos concordantes={min_votos} "
          f"| galeria: {len(gallery.ids)} embeddings\n")

    t0 = time.time()
    identificadas = desconhecidas = descartadas = 0
    for tr in pendentes:
        crops = db.track_crops(tr["id"])
        pid, nome, score, votos, status = processar_trilha(
            engine, gallery, base, tr, crops, limiar, min_votos)

        resumo = ", ".join(f"{p if p is not None else '-'}:{s:.3f}" for p, s in votos)
        db.resolve_track(tr["id"], pid, nome, score, json.dumps(votos), status)

        hora = time.strftime("%H:%M:%S", time.localtime(tr["started_at"]))
        if pid is not None:
            identificadas += 1
            print(f"  [{hora}] trilha {tr['id']}: {nome} "
                  f"(score médio {score:.3f}) | votos: {resumo}")
        elif status == "descartado":
            descartadas += 1
            print(f"  [{hora}] trilha {tr['id']}: descartada (recortes ilegíveis)")
        else:
            desconhecidas += 1
            print(f"  [{hora}] trilha {tr['id']}: Desconhecido "
                  f"(melhor {score:.3f} < {limiar}) | votos: {resumo}")

    dt = time.time() - t0
    print(f"\n{identificadas} identificada(s), {desconhecidas} desconhecida(s), "
          f"{descartadas} descartada(s) em {dt:.1f}s "
          f"({dt / max(1, len(pendentes)):.2f}s por trilha)")
    return 0


def presenca(db, dia: str) -> int:
    if dia:
        try:
            base = time.strptime(dia, "%Y-%m-%d")
        except ValueError:
            print("Data inválida. Use o formato AAAA-MM-DD.")
            return 1
        inicio = time.mktime(base)
    else:
        agora = time.localtime()
        inicio = time.mktime((agora.tm_year, agora.tm_mon, agora.tm_mday,
                              0, 0, 0, 0, 0, -1))
    fim = inicio + 86400
    rotulo = time.strftime("%d/%m/%Y", time.localtime(inicio))

    linhas = db.attendance(inicio, fim)
    pessoas = {p["id"]: p["name"] for p in db.list_people()}
    presentes = {l["person_id"] for l in linhas}

    pend = db.count_tracks_by_status().get("pendente", 0)
    print(f"\n=== Presença de {rotulo} ===")
    if pend:
        print(f"⚠  {pend} trilha(s) ainda não reconhecida(s). Rode o "
              f"reconhecimento antes de usar este relatório.")

    print(f"\nPRESENTES ({len(linhas)}):")
    for l in linhas:
        print(f"  {time.strftime('%H:%M', time.localtime(l['primeira']))}  "
              f"{l['name']:<28} {l['passagens']} passagem(ns), "
              f"melhor score {l['melhor_score']:.3f}")

    ausentes = [n for pid, n in pessoas.items() if pid not in presentes]
    print(f"\nNÃO IDENTIFICADOS ({len(ausentes)}):")
    for n in sorted(ausentes, key=str.lower):
        print(f"  {n}")
    if ausentes:
        print("\n  Atenção: 'não identificado' não é o mesmo que 'ausente'. Pode")
        print("  ser falha de captura. Confira as trilhas marcadas como")
        print("  Desconhecido antes de tratar como falta.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconhecimento em lote das trilhas.")
    ap.add_argument("--limite", type=int, default=500,
                    help="máximo de trilhas por execução (padrão 500)")
    ap.add_argument("--reprocessar", action="store_true",
                    help="reprocessa também as já processadas (use após novos cadastros)")
    ap.add_argument("--presenca", action="store_true", help="relatório de presença")
    ap.add_argument("--dia", default="", help="dia do relatório (AAAA-MM-DD)")
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg.storage.db_path)
    if args.presenca:
        return presenca(db, args.dia)
    return reconhecer(cfg, db, args.limite, args.reprocessar)


if __name__ == "__main__":
    raise SystemExit(main())
