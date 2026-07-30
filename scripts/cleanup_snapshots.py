"""Retenção de snapshots e eventos — protege o cartão SD do Raspberry Pi.

Um corredor movimentado pode gerar centenas de MB por dia em data/snapshots/.
No Pi isso enche o cartão e o serviço morre sem aviso claro.

Uso:
    python scripts/cleanup_snapshots.py --days 30           # apaga o que passou de 30 dias
    python scripts/cleanup_snapshots.py --days 30 --dry-run # só mostra o que faria
    python scripts/cleanup_snapshots.py --max-mb 2000       # respeita também um teto de tamanho

Também é uma exigência prática de LGPD: rosto é dado biométrico sensível e
precisa de prazo de retenção definido.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config, project_path  # noqa: E402
from core.database import Database  # noqa: E402


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description="Apaga snapshots e eventos antigos.")
    ap.add_argument("--days", type=int, default=30,
                    help="mantém apenas os últimos N dias (padrão: 30)")
    ap.add_argument("--max-mb", type=int, default=0,
                    help="se >0, remove as pastas mais antigas até caber neste tamanho")
    ap.add_argument("--dry-run", action="store_true", help="não apaga nada, só relata")
    ap.add_argument("--keep-events", action="store_true",
                    help="apaga as imagens mas mantém os registros no banco")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="não compacta o banco (o VACUUM reescreve o arquivo inteiro "
                         "no cartão SD; útil evitar se o banco for grande)")
    args = ap.parse_args()

    cfg = load_config()
    base = project_path(cfg.storage.snapshots_dir)
    if not base.exists():
        print(f"Nada a fazer: {base} não existe.")
        return 0

    cutoff_ts = time.time() - args.days * 86400
    cutoff_day = time.strftime("%Y%m%d", time.localtime(cutoff_ts))

    # As pastas são AAAAMMDD, então comparação de string funciona como data.
    day_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.isdigit())
    old = [d for d in day_dirs if d.name < cutoff_day]

    removed_mb = 0.0
    for d in old:
        size = _dir_size_mb(d)
        print(f"{'[dry-run] ' if args.dry_run else ''}removendo {d.name} ({size:.1f} MB)")
        if not args.dry_run:
            shutil.rmtree(d, ignore_errors=True)
        removed_mb += size

    # teto de tamanho: continua apagando do mais antigo para o mais novo
    if args.max_mb:
        remaining = [d for d in day_dirs if d not in old and d.exists()]
        while remaining and _dir_size_mb(base) > args.max_mb:
            d = remaining.pop(0)
            size = _dir_size_mb(d)
            print(f"{'[dry-run] ' if args.dry_run else ''}teto de {args.max_mb} MB "
                  f"-> removendo {d.name} ({size:.1f} MB)")
            if not args.dry_run:
                shutil.rmtree(d, ignore_errors=True)
            removed_mb += size
            if args.dry_run:
                break   # sem apagar de verdade, o loop nunca convergiria

    # limpa também os registros que apontariam para imagens já apagadas
    if not args.keep_events and not args.dry_run and old:
        db = Database(cfg.storage.db_path)
        removed = db.purge_events_before(cutoff_ts, vacuum=not args.no_vacuum)
        print(f"eventos removidos do banco: {removed}")

    print(f"total liberado: {removed_mb:.1f} MB | restante em snapshots: "
          f"{_dir_size_mb(base):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
