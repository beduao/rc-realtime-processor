"""Calibra o `recognition.cosine_threshold` a partir dos eventos reais.

O limiar decide entre "é a Maria" e "é um desconhecido". O valor de fábrica
(0.363) é o ponto de partida do SFace, não uma verdade para o seu ambiente:
iluminação, ângulo e distância mudam a distribuição dos scores. Calibrar no
chute costuma trocar um problema pelo outro.

Uso:
    python scripts/calibrate_threshold.py              # relatório dos scores
    python scripts/calibrate_threshold.py --review     # marca acerto/erro e sugere limiar
    python scripts/calibrate_threshold.py --simular 0.45   # efeito de um limiar

Como funciona o --review: ele mostra cada reconhecimento (com o link da foto) e
pergunta se acertou. Com isso separa duas distribuições — a dos acertos e a dos
erros — e propõe um limiar entre elas. As respostas ficam salvas, então dá para
revisar aos poucos.
"""

import argparse
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config_or_exit, project_path  # noqa: E402
from core.database import Database  # noqa: E402

LABELS_FILE = "data/calibracao.json"


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "IP_DO_PI"


def _load_labels() -> dict:
    path = project_path(LABELS_FILE)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_labels(labels: dict) -> None:
    path = project_path(LABELS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def _histogram(scores: list[float], largura: int = 40) -> None:
    """Histograma em texto: mostra onde os scores se concentram."""
    if not scores:
        return
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        print(f"      todos em {lo:.3f}")
        return
    faixas = 10
    passo = (hi - lo) / faixas
    for i in range(faixas):
        a, b = lo + i * passo, lo + (i + 1) * passo
        n = sum(1 for s in scores if (a <= s < b or (i == faixas - 1 and s == b)))
        barra = "#" * int(round(n / max(1, len(scores)) * largura))
        print(f"      {a:.3f}–{b:.3f}  {barra} {n}")


def relatorio(db: Database, cfg) -> list[dict]:
    limiar = float(cfg.recognition.cosine_threshold)
    eventos = db.list_events(limit=500)
    conhecidos = [e for e in eventos if e["is_known"]]
    desconhecidos = [e for e in eventos if not e["is_known"]]

    print(f"\nLimiar atual: {limiar}")
    print(f"Eventos analisados: {len(eventos)} "
          f"({len(conhecidos)} reconhecidos, {len(desconhecidos)} desconhecidos)")

    if conhecidos:
        scores = sorted(e["score"] for e in conhecidos)
        print(f"\nScores dos RECONHECIDOS (min {scores[0]:.3f} / "
              f"mediana {scores[len(scores) // 2]:.3f} / max {scores[-1]:.3f}):")
        _histogram(scores)
        print("\n  Um falso positivo aparece aqui como um score BAIXO — é o rosto")
        print("  de outra pessoa que mesmo assim passou do limiar.")

    if desconhecidos:
        scores = sorted((e["score"] for e in desconhecidos), reverse=True)
        print(f"\nScores dos DESCONHECIDOS (maior {scores[0]:.3f}):")
        print("  Se você reconhece alguém cadastrado entre estes, o limiar está")
        print("  alto demais (falso negativo).")

    return conhecidos


def review(db: Database, cfg, ip: str, porta: int) -> None:
    eventos = relatorio(db, cfg)
    if not eventos:
        print("\nSem reconhecimentos para revisar ainda.")
        return

    labels = _load_labels()
    # Do menor para o maior score: os falsos positivos costumam estar no início.
    eventos.sort(key=lambda e: e["score"])
    pendentes = [e for e in eventos if str(e["id"]) not in labels]

    print(f"\n{len(pendentes)} evento(s) a revisar "
          f"({len(labels)} já revisado(s)).")
    print("Responda:  s = acertou   n = ERROU (outra pessoa)   p = pular   q = sair\n")

    for ev in pendentes:
        url = f"http://{ip}:{porta}/snapshots/{ev['snapshot_path']}"
        print(f"  score {ev['score']:.3f}  ->  identificou como '{ev['name']}'")
        print(f"  foto: {url}")
        try:
            resp = input("  acertou? [s/n/p/q] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(interrompido)")
            break
        if resp == "q":
            break
        if resp in ("s", "n"):
            labels[str(ev["id"])] = "certo" if resp == "s" else "errado"
            _save_labels(labels)
        print()

    sugerir(eventos, labels, float(cfg.recognition.cosine_threshold))


def sugerir(eventos: list[dict], labels: dict, limiar_atual: float) -> None:
    certos = [e["score"] for e in eventos if labels.get(str(e["id"])) == "certo"]
    errados = [e["score"] for e in eventos if labels.get(str(e["id"])) == "errado"]

    print("\n--- Sugestão de limiar ---")
    print(f"  acertos marcados: {len(certos)} | erros marcados: {len(errados)}")
    if not certos and not errados:
        print("  Marque alguns eventos com --review para eu poder sugerir.")
        return

    if certos:
        print(f"  menor score de um ACERTO:  {min(certos):.3f}")
    if errados:
        print(f"  maior score de um ERRO:    {max(errados):.3f}")

    if not errados:
        print("\n  Nenhum falso positivo marcado. Se eles acontecem mas você ainda")
        print("  não os marcou, rode --review de novo depois de mais passagens.")
        return
    if not certos:
        print(f"\n  Só há erros marcados. Suba o limiar acima de {max(errados):.3f}")
        print("  e verifique se você continua sendo reconhecida.")
        return

    pior_erro, pior_acerto = max(errados), min(certos)
    if pior_erro < pior_acerto:
        sugerido = round((pior_erro + pior_acerto) / 2, 3)
        print(f"\n  As duas distribuições estão SEPARADAS "
              f"({pior_erro:.3f} < {pior_acerto:.3f}).")
        print(f"  Limiar sugerido: {sugerido}   (atual: {limiar_atual})")
        print("\n  No config.yaml:")
        print("    recognition:")
        print(f"      cosine_threshold: {sugerido}")
        print("\n  Depois:  sudo systemctl restart facial-worker")
    else:
        print(f"\n  ⚠ As distribuições se SOBREPÕEM "
              f"({pior_acerto:.3f} <= {pior_erro:.3f}).")
        print("  Nenhum limiar separa os dois casos — mexer nele só troca")
        print("  falso positivo por falso negativo. O que resolve de verdade:")
        print("    1. cadastrar MAIS amostras suas, com ângulos e luz variados;")
        print("    2. cadastrar também as outras pessoas que passam — assim o")
        print("       rosto delas casa com elas, não com você;")
        print("    3. aumentar recognition.min_face_size (rosto pequeno gera")
        print("       embedding ruim e é fonte clássica de confusão);")
        print("    4. melhorar o enquadramento: rosto de frente, sem contraluz.")


def simular(db: Database, cfg, novo: float) -> None:
    eventos = db.list_events(limit=500)
    labels = _load_labels()
    conhecidos = [e for e in eventos if e["is_known"]]
    rejeitados = [e for e in conhecidos if e["score"] < novo]

    print(f"\nCom cosine_threshold = {novo} (atual: {cfg.recognition.cosine_threshold}):")
    print(f"  {len(rejeitados)} dos {len(conhecidos)} reconhecimentos passariam a "
          f"'Desconhecido'.")

    certos = sum(1 for e in rejeitados if labels.get(str(e["id"])) == "certo")
    errados = sum(1 for e in rejeitados if labels.get(str(e["id"])) == "errado")
    if certos or errados:
        print(f"  Dentre os revisados: eliminaria {errados} erro(s) "
              f"e perderia {certos} acerto(s).")
        if errados and not certos:
            print("  Ou seja: só ganho, sem perda — bom candidato.")
    else:
        print("  (rode --review para saber quais desses eram erros de verdade)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra o limiar de reconhecimento.")
    ap.add_argument("--review", action="store_true",
                    help="revisa os eventos um a um e sugere o limiar")
    ap.add_argument("--simular", type=float, metavar="LIMIAR",
                    help="mostra o efeito de um limiar sem aplicá-lo")
    args = ap.parse_args()

    cfg = load_config_or_exit()
    db = Database(cfg.storage.db_path)
    porta = int(cfg.api.get("port", 8000))

    if args.simular is not None:
        simular(db, cfg, args.simular)
    elif args.review:
        review(db, cfg, _local_ip(), porta)
    else:
        eventos = relatorio(db, cfg)
        sugerir(eventos, _load_labels(), float(cfg.recognition.cosine_threshold))
        print("\nPara marcar quais foram erros:  "
              "python scripts/calibrate_threshold.py --review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
