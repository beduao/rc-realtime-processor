"""Retenção de imagens e registros — protege o cartão SD e atende a LGPD.

Há TRÊS acervos de imagem, com prazos diferentes porque têm naturezas
diferentes:

  data/snapshots/AAAAMMDD/     fotos das passagens registradas.
                               Prazo: --days (padrão 30).

  data/tracks/AAAAMMDD/        recortes usados pelo reconhecimento em lote.
                               São evidência TRANSITÓRIA: depois que o lote
                               processou a trilha, servem só para auditoria.
                               Prazo curto: --dias-trilhas (padrão 7).

  data/snapshots/amostras/<id>/  fotos do cadastro de cada pessoa.
                               NÃO têm prazo: valem enquanto a pessoa estiver
                               cadastrada, e são removidas quando ela é
                               excluída pelo painel.

E dois acervos de registro no banco, com o mesmo prazo de --days:
  events   (modo realtime)  e  tracks (modo captura)

Uso:
    python scripts/cleanup_snapshots.py --days 30 --dias-trilhas 7
    python scripts/cleanup_snapshots.py --dry-run          # só relata
    python scripts/cleanup_snapshots.py --max-mb 2000      # teto de tamanho
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config_or_exit, project_path  # noqa: E402
from core.database import Database  # noqa: E402


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def _dias_anteriores(base: Path, corte_dia: str) -> list[Path]:
    """Pastas AAAAMMDD anteriores ao corte.

    O nome da pasta é a data, então comparação de string equivale a comparação
    de data — e a limpeza vira remoção de diretório, sem varrer arquivo por
    arquivo checando mtime.
    """
    if not base.exists():
        return []
    dias = sorted(d for d in base.iterdir() if d.is_dir() and d.name.isdigit())
    return [d for d in dias if d.name < corte_dia]


def _remover(pastas: list[Path], rotulo: str, dry_run: bool) -> float:
    liberado = 0.0
    for d in pastas:
        tam = _dir_size_mb(d)
        print(f"  {'[dry-run] ' if dry_run else ''}{rotulo}: removendo {d.name} "
              f"({tam:.1f} MB)")
        if not dry_run:
            shutil.rmtree(d, ignore_errors=True)
        liberado += tam
    return liberado


def main() -> int:
    ap = argparse.ArgumentParser(description="Apaga imagens e registros antigos.")
    ap.add_argument("--days", type=int, default=30,
                    help="prazo dos snapshots de passagem e dos registros (padrão 30)")
    ap.add_argument("--dias-trilhas", type=int, default=7,
                    help="prazo dos recortes do modo captura (padrão 7)")
    ap.add_argument("--max-mb", type=int, default=0,
                    help="se >0, remove os dias mais antigos até caber neste tamanho")
    ap.add_argument("--dry-run", action="store_true", help="não apaga nada, só relata")
    ap.add_argument("--keep-events", action="store_true",
                    help="apaga as imagens mas mantém os registros no banco")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="não compacta o banco (o VACUUM reescreve o arquivo inteiro "
                         "no cartão SD; útil evitar se o banco for grande)")
    args = ap.parse_args()

    cfg = load_config_or_exit()
    db = Database(cfg.storage.db_path)

    snaps = project_path(cfg.storage.snapshots_dir)
    tracks = project_path((cfg.get("tracking") or {}).get("crops_dir", "data/tracks"))

    agora = time.time()
    corte_reg = agora - args.days * 86400
    corte_snap_dia = time.strftime("%Y%m%d", time.localtime(corte_reg))
    corte_tr = agora - args.dias_trilhas * 86400
    corte_tr_dia = time.strftime("%Y%m%d", time.localtime(corte_tr))

    liberado = 0.0

    # ---- 1. recortes das trilhas (prazo curto) ----------------------------- #
    # Trava: se o lote parou de rodar, há trilhas pendentes antigas e apagar os
    # recortes delas jogaria fora dado que nunca foi aproveitado.
    pendentes_antigas = db.pending_tracks_before(corte_tr)
    if pendentes_antigas:
        print(f"⚠  {pendentes_antigas} trilha(s) PENDENTES mais antigas que "
              f"{args.dias_trilhas} dias.")
        print("   O reconhecimento em lote não está rodando. Os recortes NÃO serão")
        print("   apagados para não perder esses dados. Verifique:")
        print("     sudo systemctl status facial-batch.timer")
        print("     python scripts/recognize_batch.py")
    else:
        antigas = _dias_anteriores(tracks, corte_tr_dia)
        if antigas:
            liberado += _remover(antigas, "trilhas", args.dry_run)
            if not args.dry_run:
                n = db.drop_track_crops_before(corte_tr)
                print(f"  trilhas: {n} referência(s) de recorte removida(s) do banco")
        else:
            print(f"  trilhas: nada além de {args.dias_trilhas} dias")

    # ---- 2. snapshots de passagem (prazo longo) ---------------------------- #
    antigos = _dias_anteriores(snaps, corte_snap_dia)
    if antigos:
        liberado += _remover(antigos, "snapshots", args.dry_run)
    else:
        print(f"  snapshots: nada além de {args.days} dias")

    # ---- 3. teto de tamanho ------------------------------------------------ #
    if args.max_mb and not args.dry_run:
        restantes = [d for d in sorted(snaps.iterdir())
                     if d.is_dir() and d.name.isdigit()] if snaps.exists() else []
        while restantes and _dir_size_mb(snaps) > args.max_mb:
            d = restantes.pop(0)
            tam = _dir_size_mb(d)
            print(f"  teto de {args.max_mb} MB -> removendo {d.name} ({tam:.1f} MB)")
            shutil.rmtree(d, ignore_errors=True)
            liberado += tam

    # ---- 4. registros no banco --------------------------------------------- #
    if not args.keep_events and not args.dry_run:
        ev = db.purge_events_before(corte_reg, vacuum=False)
        tr = db.purge_tracks_before(corte_reg)
        print(f"  banco: {ev} evento(s) e {tr} trilha(s) removidas")
        if (ev or tr) and not args.no_vacuum:
            db.purge_events_before(corte_reg, vacuum=True)   # dispara o VACUUM

    total_snaps = _dir_size_mb(snaps) if snaps.exists() else 0.0
    total_tracks = _dir_size_mb(tracks) if tracks.exists() else 0.0
    print(f"\nliberado: {liberado:.1f} MB | em uso agora: "
          f"snapshots {total_snaps:.1f} MB + trilhas {total_tracks:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
