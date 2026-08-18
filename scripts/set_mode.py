"""Troca o modo de processamento sem editar YAML na mão e sem reiniciar nada.

O worker relê `worker.mode` a cada 10 segundos, então a troca vale sozinha —
não precisa de `systemctl restart`, e no modo captura as trilhas que estavam
em cena são salvas antes da transição.

Uso:
    python scripts/set_mode.py                 # mostra o modo atual
    python scripts/set_mode.py realtime         # reconhece na hora
    python scripts/set_mode.py captura          # captura agora, reconhece depois

Quando usar cada um:
    realtime  — poucas pessoas por vez, ou teste em que você quer ver o
                resultado na hora no painel.
    captura   — grupos, chamada de presença. Aguenta o corredor cheio e decide
                por votação entre vários recortes, o que erra menos.
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config_path, live_image_path, load_config  # noqa: E402

MODOS = ("realtime", "captura")


def _status(cfg) -> dict:
    """Estado publicado pelo worker (o modo REALMENTE em execução)."""
    caminho = live_image_path(cfg).with_name("facial-status.json")
    try:
        with open(caminho, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def mostrar(cfg) -> int:
    configurado = str((cfg.get("worker") or {}).get("mode", "realtime"))
    print(f"config.yaml:  worker.mode = {configurado}")

    st = _status(cfg)
    if not st:
        print("worker:       sem status publicado (o serviço está rodando?)")
        print("              sudo systemctl status facial-worker")
        return 0

    idade = time.time() - st.get("updated_at", 0)
    if idade > 600:
        print(f"worker:       status com {idade / 60:.0f} min — provavelmente parado")
    else:
        extra = " (fixado por --mode)" if st.get("fixo") else ""
        print(f"worker:       rodando em {st.get('mode')}{extra}, "
              f"atualizado há {idade:.0f}s")
        if st.get("pendentes"):
            print(f"              {st['pendentes']} trilha(s) aguardando reconhecimento")
    if st.get("fixo") and st.get("mode") != configurado:
        print("\n⚠  O worker foi iniciado com --mode, então ignora o config.yaml.")
        print("   Para voltar a obedecer o config, reinicie sem esse argumento:")
        print("   sudo systemctl restart facial-worker")
    return 0


def trocar(cfg, novo: str) -> int:
    caminho = config_path()
    if caminho.name.endswith(".example.yaml"):
        print(f"Você ainda não tem config.yaml — está usando {caminho.name}.")
        print("Crie o seu antes:  cp config.pi.example.yaml config.yaml")
        return 1

    texto = caminho.read_text(encoding="utf-8")
    atual = str((cfg.get("worker") or {}).get("mode", "realtime"))
    if atual == novo:
        print(f"Já está em {novo}.")
        return mostrar(cfg)

    # Substituição pontual da linha, preservando comentários e formatação do
    # arquivo — reescrever o YAML inteiro apagaria toda a documentação inline.
    novo_texto, n = re.subn(r"(?m)^(\s*mode:\s*).*$",
                            lambda m: f'{m.group(1)}"{novo}"', texto, count=1)
    if n == 0:
        print("Não encontrei a linha 'mode:' no config.yaml. Adicione dentro de "
              "'worker:':\n    mode: \"" + novo + '"')
        return 1

    caminho.write_text(novo_texto, encoding="utf-8")
    print(f"config.yaml: {atual} -> {novo}")
    print("O worker aplica em até 10 segundos, sem reiniciar.")

    if novo == "captura":
        print("\nLembre de ligar o reconhecimento em lote, se ainda não estiver:")
        print("  sudo systemctl enable --now facial-batch.timer")
        print("  python scripts/recognize_batch.py --presenca")
    else:
        print("\nNo modo realtime o lote fica ocioso, mas pode continuar ligado —")
        print("ele ainda processa trilhas pendentes que sobraram da captura.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Mostra ou troca o modo do worker.")
    ap.add_argument("modo", nargs="?", choices=MODOS,
                    help="sem argumento, apenas mostra o estado atual")
    args = ap.parse_args()

    cfg = load_config()
    return mostrar(cfg) if args.modo is None else trocar(cfg, args.modo)


if __name__ == "__main__":
    raise SystemExit(main())
