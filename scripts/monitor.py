"""Monitora o Pi COM O RECONHECIMENTO RODANDO, sem interferir nele.

Diferença para o `check_pi.py`: aquele abre a câmera para medir, o que com
webcam USB é impossível enquanto o worker roda (dispositivo V4L2 é exclusivo) e
com câmera IP abre uma segunda conexão RTSP. Este aqui não toca na câmera —
lê tudo de /proc, do `vcgencmd`, do arquivo de status do worker e do banco.

Uso:
    python scripts/monitor.py                    # até Ctrl+C, amostra a cada 2s
    python scripts/monitor.py --segundos 300      # roda 5 min e resume
    python scripts/monitor.py --intervalo 5
    python scripts/monitor.py --uma-vez           # leitura única

O que observar num Pi 3B:
    temp      acima de 80 °C ele reduz o clock sozinho e tudo fica mais lento
    throttled diferente de 0x0 indica subtensão — fonte fraca, comum com
              webcam USB no mesmo barramento
    mem       menos de ~80 MB livres leva a swap, e swap em cartão SD é fatal
              para latência
    pend      trilhas aguardando reconhecimento; se só cresce, o lote não está
              dando conta do fluxo
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import live_image_path, load_config_or_exit, project_path  # noqa: E402

TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# --------------------------------------------------------------------------- #
# coletores
# --------------------------------------------------------------------------- #
def cpu_total():
    """(usado, total) em ticks — a porcentagem sai da diferença entre amostras."""
    with open("/proc/stat") as fh:
        campos = [float(x) for x in fh.readline().split()[1:]]
    ocioso = campos[3] + (campos[4] if len(campos) > 4 else 0)
    return sum(campos) - ocioso, sum(campos)


def memoria():
    m = {}
    with open("/proc/meminfo") as fh:
        for linha in fh:
            chave, _, valor = linha.partition(":")
            m[chave] = int(valor.split()[0]) // 1024      # MB
    return {
        "disponivel": m.get("MemAvailable", 0),
        "total": m.get("MemTotal", 0),
        "swap_usado": m.get("SwapTotal", 0) - m.get("SwapFree", 0),
    }


def _vcgencmd(*args) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True,
                              text=True, timeout=4).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def temperatura():
    """Temperatura em °C, ou None se não houver como medir nesta máquina."""
    saida = _vcgencmd("measure_temp")            # temp=61.0'C
    if "=" in saida:
        try:
            return float(saida.split("=")[1].split("'")[0])
        except (IndexError, ValueError):
            pass
    try:                                          # fallback sem vcgencmd
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read()) / 1000.0
    except OSError:
        return None


def throttled():
    """Bitmask do vcgencmd, ou None quando não dá para medir.

    None é diferente de 0: 0 significa "medi e está tudo bem", None significa
    "não sei" — e afirmar que não houve subtensão sem ter medido seria pior
    que não dizer nada.
    """
    saida = _vcgencmd("get_throttled")           # throttled=0x0
    if "=" in saida:
        try:
            return int(saida.split("=")[1], 16)
        except ValueError:
            pass
    return None


def explicar_throttled(flags: int) -> list[str]:
    """Traduz os bits do vcgencmd. Os bits 16..19 são histórico desde o boot."""
    if flags <= 0:
        return []
    mapa = [
        (0, "subtensão AGORA"), (1, "clock limitado AGORA"),
        (2, "throttling térmico AGORA"), (3, "temperatura crítica AGORA"),
        (16, "houve subtensão desde o boot"), (17, "houve limite de clock"),
        (18, "houve throttling térmico"), (19, "houve temperatura crítica"),
    ]
    return [texto for bit, texto in mapa if flags & (1 << bit)]


def processo(pid):
    """(cpu_ticks, memoria_MB) do processo, ou None se ele não existe mais."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            campos = fh.read().rsplit(") ", 1)[1].split()
        utime, stime = float(campos[11]), float(campos[12])
        with open(f"/proc/{pid}/statm") as fh:
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // (1024 ** 2)
        return utime + stime, rss
    except (OSError, IndexError, ValueError):
        return None


def _e_o_worker(pid) -> bool:
    """Confere se o PID é mesmo `python .../worker.py`.

    Buscar a substring "worker.py" na linha de comando inteira dá falso
    positivo em qualquer shell, grep ou editor que a mencione. A linha vem
    separada por bytes nulos, então comparamos argumento por argumento.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            args = [a.decode("utf-8", "replace")
                    for a in fh.read().split(b"\0") if a]
    except OSError:
        return False
    if not args:
        return False
    return ("python" in os.path.basename(args[0])
            and any(a.endswith("worker.py") for a in args[1:]))


def achar_worker(status: dict):
    """PID do worker: o status publica, mas confirmamos que ainda é ele."""
    pid = status.get("pid")
    if pid and _e_o_worker(pid):
        return pid
    for entrada in os.listdir("/proc"):     # status velho: procura pelo nome
        if entrada.isdigit() and _e_o_worker(entrada):
            return int(entrada)
    return None


def ler_status(caminho):
    try:
        with open(caminho, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- #
# laço
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Monitora o Pi durante o reconhecimento.")
    ap.add_argument("--intervalo", type=float, default=2.0, help="segundos entre amostras")
    ap.add_argument("--segundos", type=float, default=0, help="duração (0 = até Ctrl+C)")
    ap.add_argument("--uma-vez", action="store_true", help="uma leitura e sai")
    args = ap.parse_args()

    cfg = load_config_or_exit()
    status_path = live_image_path(cfg).with_name("facial-status.json")
    db = None
    try:
        from core.database import Database
        db = Database(cfg.storage.db_path)
    except Exception:                                    # noqa: BLE001
        pass
    raiz = project_path(".")

    print(f"monitorando a cada {args.intervalo:g}s — Ctrl+C encerra e resume\n")
    cab = (f"{'hora':<9}{'cpu':>5}{'temp':>7}{'livre':>8}{'swap':>7}  "
           f"{'worker':<10}{'cpu':>6}{'mem':>7}{'fps':>6}{'pend':>6}{'disco':>7}")
    print(cab)
    print("-" * len(cab))

    ant_cpu = cpu_total()
    ant_proc = ant_t = None
    temp_max = None
    flags_vistos = None            # None = nunca consegui medir
    pend_inicial = pend_atual = None
    amostras = 0
    fim = time.time() + args.segundos if args.segundos else None

    try:
        while True:
            time.sleep(args.intervalo)
            agora = time.time()

            usado, total = cpu_total()
            d_usado, d_total = usado - ant_cpu[0], total - ant_cpu[1]
            cpu_pct = 100 * d_usado / d_total if d_total else 0
            ant_cpu = (usado, total)

            mem = memoria()
            temp = temperatura()
            if temp is not None:
                temp_max = temp if temp_max is None else max(temp_max, temp)
            fl = throttled()
            if fl is not None:
                flags_vistos = fl if flags_vistos is None else (flags_vistos | fl)

            st = ler_status(status_path)
            modo = st.get("mode", "?")
            fresco = st and (agora - st.get("updated_at", 0) < 600)
            pid = achar_worker(st)

            p_cpu, p_mem = "-", "-"
            atual = processo(pid) if pid else None
            if atual and ant_proc and ant_t:
                d = (atual[0] - ant_proc[0]) / TICKS
                p_cpu = f"{100 * d / (agora - ant_t):.0f}%"
                p_mem = f"{atual[1]}MB"
            elif atual:
                p_mem = f"{atual[1]}MB"
            ant_proc, ant_t = atual, agora

            if db is not None:
                try:
                    pend_atual = db.count_tracks_by_status().get("pendente", 0)
                except Exception:                        # noqa: BLE001
                    pend_atual = None
            if pend_inicial is None:
                pend_inicial = pend_atual

            livre_gb = shutil.disk_usage(str(raiz)).free // (1024 ** 3)
            fps = st.get("fps", "-") if fresco else "-"

            temp_txt = f"{temp:>6.1f}C" if temp is not None else f"{'-':>7}"
            print(f"{time.strftime('%H:%M:%S'):<9}{cpu_pct:>4.0f}%{temp_txt}"
                  f"{mem['disponivel']:>6}MB{mem['swap_usado']:>5}MB  "
                  f"{(modo if fresco else 'parado?'):<10}{p_cpu:>6}{p_mem:>7}"
                  f"{str(fps):>6}{str(pend_atual if pend_atual is not None else '-'):>6}"
                  f"{livre_gb:>5}GB")

            avisos = []
            if temp is not None and temp >= 80:
                avisos.append(f"temperatura {temp:.0f}°C — o Pi já está reduzindo o clock")
            elif temp is not None and temp >= 70:
                avisos.append(f"temperatura {temp:.0f}°C — perto do limite de throttling")
            if fl and (fl & 0xF):
                avisos.append("subtensão/throttling ACONTECENDO agora: " +
                              ", ".join(explicar_throttled(fl & 0xF)))
            if mem["disponivel"] < 80:
                avisos.append(f"só {mem['disponivel']} MB livres — risco de swap")
            if mem["swap_usado"] > 50:
                avisos.append(f"{mem['swap_usado']} MB em swap — latência vai sofrer")
            if not fresco and st:
                avisos.append("status do worker desatualizado — serviço parado?")
            for a in avisos:
                print(f"           !! {a}")

            amostras += 1
            if args.uma_vez or (fim and agora >= fim):
                break
    except KeyboardInterrupt:
        print()

    # ---- resumo ----------------------------------------------------------- #
    print("\n=== resumo ===")
    print(f"  amostras: {amostras}")
    if temp_max is not None:
        print(f"  temperatura máxima: {temp_max:.1f}°C")
    else:
        print("  temperatura: não foi possível medir nesta máquina")

    if flags_vistos is None:
        print("  energia/throttling: não medido (vcgencmd indisponível — "
              "este script foi feito para rodar NO Pi)")
    elif flags_vistos == 0:
        print("  sem subtensão nem throttling durante a observação")
    else:
        print("  ⚠ eventos de energia/temperatura:")
        for h in explicar_throttled(flags_vistos):
            print(f"      - {h}")
        print("    Subtensão com webcam USB costuma ser fonte fraca ou cabo ruim:")
        print("    use fonte oficial de 2,5 A, ou hub USB com alimentação própria.")

    if pend_inicial is not None and pend_atual is not None:
        delta = pend_atual - pend_inicial
        if delta > 0:
            print(f"  ⚠ fila de trilhas CRESCEU {delta} ({pend_inicial} -> {pend_atual}):")
            print("    o reconhecimento em lote não está acompanhando a captura.")
            print("    Rode o lote mais vezes, ou processe num computador mais rápido.")
        elif delta < 0:
            print(f"  fila diminuiu {-delta} ({pend_inicial} -> {pend_atual}): o lote "
                  "está dando conta")
        else:
            print(f"  fila estável em {pend_atual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
