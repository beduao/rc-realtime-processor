#!/usr/bin/env python3
"""Confere se esta máquina está pronta para rodar TUDO local (worker+API+painel).

Feito para o cenário "câmera no meu PC, painel no localhost": lista o que
falta antes de abrir os três terminais, em vez de deixar cada peça falhar com
mensagem própria.

    python scripts/check_local.py
    python scripts/check_local.py --camera 1     # testar outro índice

Diferente do `check_pi.py`, que mede desempenho e assume Linux/Raspberry.
"""
from __future__ import annotations

import argparse
import platform
import socket
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

OK, AVISO, ERRO = "[ok]", "[!]", "[XX]"
problemas: list[str] = []


def diz(marca, texto, detalhe=""):
    print(f"  {marca:<5} {texto}")
    if detalhe:
        for linha in str(detalhe).rstrip().split("\n"):
            print(f"        {linha}")
    if marca == ERRO:
        problemas.append(texto)


def secao(titulo):
    print(f"\n{titulo}")


# --------------------------------------------------------------------------- #
def checar_python():
    secao("Python e dependências")
    v = sys.version_info
    if v < (3, 9):
        diz(ERRO, f"Python {v.major}.{v.minor} é antigo",
            "Use 3.11+ de python.org (marque 'Add python.exe to PATH').")
    else:
        diz(OK, f"Python {v.major}.{v.minor}.{v.micro}")

    diz(OK, f"Sistema: {platform.system()} {platform.release()} "
            f"({platform.machine()})")

    # Caminho com acento quebra o importador ONNX do OpenCV no Windows: ele
    # abre o arquivo com std::ifstream, que converte pela página de código
    # ANSI local. O erro que sai é "Can't read ONNX file", que parece
    # corrupção do modelo e manda procurar no lugar errado.
    try:
        str(RAIZ).encode("ascii")
        ascii_ok = True
    except UnicodeEncodeError:
        ascii_ok = False
    if not ascii_ok:
        fora = "".join(sorted({c for c in str(RAIZ) if ord(c) > 127}))
        diz(AVISO, f"o caminho do projeto tem caractere não-ASCII: {fora}",
            f"{RAIZ}\n"
            "O módulo DNN do OpenCV não abre caminho com acento no Windows.\n"
            "O core/face_engine.py contorna isso (nome curto 8.3 e, se\n"
            "preciso, cópia ASCII do modelo). Se ainda assim falhar com\n"
            "\"Can't read ONNX file\", mova o projeto para um caminho sem\n"
            "acento, por exemplo C:\\rc-realtime-processor.")

    faltando = []
    try:
        import cv2
        versao = tuple(int(x) for x in cv2.__version__.split(".")[:2])
        if versao < (4, 8):
            diz(ERRO, f"OpenCV {cv2.__version__} é antigo demais",
                "O detector YuNet 2023mar exige OpenCV >= 4.8. Rode:\n"
                "  python -m pip install -U 'opencv-contrib-python-headless>=4.9,<5'")
        else:
            diz(OK, f"OpenCV {cv2.__version__}")
    except ImportError:
        faltando.append("opencv-contrib-python-headless")

    for mod, pacote in (("numpy", "numpy"), ("yaml", "pyyaml"),
                        ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                        ("streamlit", "streamlit"), ("requests", "requests")):
        try:
            __import__(mod)
        except ImportError:
            faltando.append(pacote)

    if faltando:
        diz(ERRO, f"Pacotes faltando: {', '.join(faltando)}",
            "python -m pip install -r requirements-pi.txt -r requirements-panel.txt")
    else:
        diz(OK, "numpy, pyyaml, fastapi, uvicorn, streamlit, requests")


def checar_config():
    secao("Configuração")
    caminho = RAIZ / "config.yaml"
    if not caminho.exists():
        diz(ERRO, "config.yaml não existe",
            "Windows:  copy config.example.yaml config.yaml\n"
            "Git Bash: cp config.example.yaml config.yaml")
        return None

    try:
        from core.config import load_config
        cfg = load_config()
    except Exception as exc:                             # noqa: BLE001
        diz(ERRO, f"config.yaml não pôde ser lido: {type(exc).__name__}", exc)
        return None
    diz(OK, "config.yaml carregado")

    fonte = cfg.camera.rtsp_url
    if str(fonte).strip().isdigit():
        diz(OK, f"camera.rtsp_url = {fonte} (webcam local)")
    elif "SENHA_CODIFICADA" in str(fonte) or not str(fonte).strip():
        diz(ERRO, "camera.rtsp_url não foi configurada",
            "Para a webcam do PC, use:  rtsp_url: 0")
    else:
        diz(AVISO, f"camera.rtsp_url = {fonte}",
            "É uma URL/arquivo. Para a webcam deste PC, troque por 0.")

    base = cfg.api.base_url
    if "localhost" in base or "127.0.0.1" in base:
        diz(OK, f"api.base_url = {base} (painel fala com a API local)")
    else:
        diz(ERRO, f"api.base_url = {base}",
            "Aponta para outra máquina (o Pi). Rodando tudo aqui, use:\n"
            "  base_url: \"http://localhost:8000\"")

    live = str((cfg.get("storage") or {}).get("live_path") or "")
    if live.startswith("/dev/shm"):
        diz(ERRO, f"storage.live_path = {live}",
            "/dev/shm é tmpfs do Linux e não existe no Windows. Use:\n"
            "  live_path: \"data/live.jpg\"")
    elif live:
        diz(OK, f"storage.live_path = {live}")

    token = str((cfg.get("api") or {}).get("token") or "").strip()
    if token:
        diz(OK, "api.token configurado (não é exigido em acesso local)")
    else:
        diz(OK, "api.token vazio — sem problema: requisição local dispensa token")
    return cfg


def checar_modelos(cfg):
    secao("Modelos ONNX")
    try:
        from core.config import project_path
    except Exception:                                    # noqa: BLE001
        return
    m = (cfg.get("models") or {}) if cfg else {}
    alvos = [m.get("detector", "models/face_detection_yunet_2023mar.onnx"),
             m.get("recognizer", "models/face_recognition_sface_2021dec.onnx")]
    faltou = False
    for rel in alvos:
        p = project_path(rel)
        if not p.exists():
            diz(ERRO, f"não encontrado: {rel}")
            faltou = True
            continue
        kb = p.stat().st_size // 1024
        if kb < 100:
            diz(ERRO, f"{p.name} tem só {kb} KB — download incompleto")
            faltou = True
        else:
            diz(OK, f"{p.name} ({kb} KB)")
    if faltou:
        diz(AVISO, "baixe com:  python models/download_models.py")


def varrer_cameras(pular=None, ate=4):
    """Testa índices 0..ate em cada backend, distinguindo TRÊS desfechos.

    A primeira versão colapsava dois casos muito diferentes em "não achei":

      não abriu         -> dispositivo inexistente, ocupado ou sem permissão
      abriu, sem frame  -> o dispositivo EXISTE e foi aberto, mas a entrega do
                           quadro foi negada. É a assinatura clássica do
                           bloqueio de privacidade do Windows para apps da
                           área de trabalho, e de webcam presa por outro app.
      abriu com frame   -> funciona

    Sem separar o segundo caso, o diagnóstico apontava "nenhuma câmera" numa
    máquina que tem câmera — mandando procurar no lugar errado.

    Devolve (funcionando, abriu_sem_frame).
    """
    import cv2
    candidatos = []
    if sys.platform.startswith("win"):
        # CAP_FFMPEG entra porque, no relato real, foi o único backend que
        # enumerou os dispositivos (a mensagem "index out of range" dele
        # revelou nb_devices=2 quando DSHOW e MSMF diziam não ter nada).
        candidatos = [(cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_MSMF, "CAP_MSMF"),
                      (cv2.CAP_FFMPEG, "CAP_FFMPEG"), (cv2.CAP_ANY, "CAP_ANY")]
    elif sys.platform.startswith("linux"):
        candidatos = [(cv2.CAP_V4L2, "CAP_V4L2"), (cv2.CAP_ANY, "CAP_ANY")]
    else:
        candidatos = [(cv2.CAP_ANY, "CAP_ANY")]

    # Silencia o log do OpenCV durante a varredura: são dezenas de linhas de
    # aviso esperadas (é uma varredura, falha é o caso comum) e elas escondem
    # o resultado. O relatório abaixo diz o que importa.
    nivel_antigo = None
    try:
        nivel_antigo = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:                                    # noqa: BLE001
        pass

    funcionando, sem_frame = [], []
    try:
        for idx in range(ate + 1):
            for backend, nome in candidatos:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:                        # noqa: BLE001
                    continue
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    funcionando.append((idx, nome, f"{w}x{h}"))
                    break
                sem_frame.append((idx, nome))
    finally:
        if nivel_antigo is not None:
            try:
                cv2.utils.logging.setLogLevel(nivel_antigo)
            except Exception:                            # noqa: BLE001
                pass
    return funcionando, sem_frame


def _worker_com_a_camera(cfg):
    """(worker_vivo, idade_do_frame) — o worker está publicando frame agora?

    Webcam e câmera CSI aceitam UM processo por vez. Com o worker rodando, o
    teste de abrir a câmera aqui está condenado a falhar, e reportar isso como
    "nenhuma câmera encontrada" manda procurar defeito onde não há.
    """
    if cfg is None:
        return False, None
    try:
        from core.config import frame_image_path
        caminho = frame_image_path(cfg)
        idade = time.time() - caminho.stat().st_mtime
        return idade < 15, round(idade, 1)
    except Exception:                                    # noqa: BLE001
        return False, None


def checar_camera(indice, cfg=None):
    secao(f"Câmera (índice {indice})")
    try:
        import cv2
        from core.camera import _backend_local, _parse_source
    except ImportError:
        diz(AVISO, "OpenCV ausente — pulei o teste de câmera")
        return

    vivo, idade = _worker_com_a_camera(cfg)
    if vivo:
        diz(OK, f"o worker está rodando e publicando frame (há {idade}s)",
            "A câmera é dele: webcam e CSI aceitam só um processo. NÃO vou\n"
            "tentar abrir o dispositivo — a tentativa falharia e pareceria\n"
            "defeito. É assim que o cadastro funciona com o worker no ar: a\n"
            "API consome o frame publicado em vez de disputar a câmera.\n"
            "Para testar a câmera diretamente, pare o worker antes.")
        return

    nomes = {cv2.CAP_DSHOW: "CAP_DSHOW (DirectShow)",
             cv2.CAP_V4L2: "CAP_V4L2", cv2.CAP_ANY: "CAP_ANY"}
    backend = _backend_local()
    diz(OK, f"backend para esta plataforma: {nomes.get(backend, backend)}")

    # Silencia o log do OpenCV já na PRIMEIRA tentativa: a falha aqui é
    # esperada em vários cenários e o aviso dele, no meio do relatório,
    # confunde mais do que informa. O que importa este script diz por conta.
    nivel = None
    try:
        nivel = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:                                    # noqa: BLE001
        pass

    fonte, backend = _parse_source(indice)
    cap = cv2.VideoCapture(fonte, backend)
    if nivel is not None:
        try:
            cv2.utils.logging.setLogLevel(nivel)
        except Exception:                                # noqa: BLE001
            pass
    if not cap.isOpened():
        cap.release()
        diz(AVISO, f"não abriu no índice {indice} — varrendo índices e "
                   "backends (silenciando o log do OpenCV)...")
        funcionando, sem_frame = varrer_cameras(indice)

        if funcionando:
            linhas = ["Encontrei câmera FUNCIONANDO em:"]
            for idx, nome_backend, dims in funcionando:
                linhas.append(f"  índice {idx} com {nome_backend} -> {dims}")
            linhas.append("")
            if funcionando[0][0] != indice:
                linhas.append(f"Ajuste no config.yaml:  rtsp_url: {funcionando[0][0]}")
            else:
                linhas.append("O índice está certo, mas só um backend "
                              "alternativo abriu — me avise para eu tratar "
                              "isso no código.")
            diz(ERRO, f"o índice {indice} não abriu, mas há câmera disponível",
                "\n".join(linhas))
            return

        if sem_frame:
            onde = ", ".join(f"índice {i} ({b})" for i, b in sem_frame[:6])
            diz(ERRO, "o dispositivo ABRIU mas não entregou nenhum quadro",
                f"Aconteceu em: {onde}\n"
                "\n"
                "Isso é diferente de 'não existe câmera' — ela existe e foi\n"
                "aberta. Quem negou foi a entrega do quadro. Duas causas, nesta\n"
                "ordem:\n"
                "  1. Privacidade do Windows. Configurações > Privacidade e\n"
                "     segurança > Câmera. São DOIS interruptores, e o segundo\n"
                "     passa batido: 'Acesso à câmera' e, mais abaixo,\n"
                "     'Permitir que aplicativos da área de trabalho acessem\n"
                "     sua câmera'. Python cai no segundo.\n"
                "  2. outro programa segurando a webcam (Teams, Zoom, Meet,\n"
                "     Câmera do Windows, OBS) — feche todos e repita.")
            return

        diz(ERRO, "nenhum índice abriu, em nenhum backend",
            "Nem sequer abriu o dispositivo. Verifique:\n"
            "  1. no PowerShell, se o Windows enxerga a câmera:\n"
            "     Get-PnpDevice -Class Camera,Image | "
            "Format-Table FriendlyName,Status\n"
            "  2. Gerenciador de Dispositivos: a câmera aparece sem aviso\n"
            "     amarelo?\n"
            "  3. se for câmera IP (Intelbras), o índice não se aplica — use\n"
            "     a URL rtsp:// em camera.rtsp_url.")
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        diz(ERRO, "a câmera ABRIU mas não entregou nenhum quadro",
            "A câmera existe e foi aberta — quem foi negado é o quadro.\n"
            "  1. Privacidade do Windows: Configurações > Privacidade e\n"
            "     segurança > Câmera. São DOIS interruptores; o segundo passa\n"
            "     batido: 'Permitir que aplicativos da área de trabalho\n"
            "     acessem sua câmera'. Python cai nessa categoria.\n"
            "  2. outro programa segurando a webcam (Teams, Zoom, Meet,\n"
            "     Câmera do Windows, OBS) — feche todos e repita.")
        return

    h, w = frame.shape[:2]
    diz(OK, f"frame recebido: {w}x{h}")

    # fps real: o valor de CAP_PROP_FPS mente com frequência em webcam
    inicio, n = time.time(), 0
    while time.time() - inicio < 2.0:
        if cap.read()[0]:
            n += 1
    fps = n / (time.time() - inicio)
    cap.release()
    diz(OK
        if fps >= 8 else AVISO, f"~{fps:.1f} fps medidos")

    saida = RAIZ / "data" / "test_frame_local.jpg"
    saida.parent.mkdir(parents=True, exist_ok=True)
    # Confere se gravou de verdade. A versão anterior usava cv2.imwrite e
    # anunciava "frame salvo" sem verificar — com caminho não-ASCII o imwrite
    # devolve False em silêncio, e o arquivo nunca existiu. Foi essa mensagem
    # falsa que atrasou o diagnóstico de um problema de codificação de caminho.
    from core.imagem import escrever as escrever_imagem
    if escrever_imagem(saida, frame) and saida.exists():
        diz(OK, f"frame salvo em {saida.relative_to(RAIZ)} — abra para "
                "conferir enquadramento")
    else:
        diz(ERRO, f"não conseguiu gravar {saida.relative_to(RAIZ)}",
            "Verifique permissão de escrita e espaço em disco.")


def checar_portas():
    secao("Portas")
    for porta, quem in ((8000, "API"), (8501, "painel Streamlit")):
        s = socket.socket()
        s.settimeout(0.4)
        livre = s.connect_ex(("127.0.0.1", porta)) != 0
        s.close()
        if livre:
            diz(OK, f"{porta} livre ({quem})")
        else:
            diz(AVISO, f"{porta} já está em uso ({quem})",
                "Se for uma execução anterior sua, tudo bem — reaproveite. "
                "Senão, feche o processo ou troque a porta.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verifica se dá para rodar worker, API e painel nesta máquina.")
    ap.add_argument("--camera", type=int, default=None,
                    help="índice da webcam a testar (padrão: o do config)")
    args = ap.parse_args(argv)

    print("=" * 68)
    print("  Verificação para rodar TUDO nesta máquina")
    print("=" * 68)

    checar_python()
    cfg = checar_config()
    if cfg:
        checar_modelos(cfg)

    indice = args.camera
    if indice is None and cfg:
        fonte = str(cfg.camera.rtsp_url).strip()
        indice = int(fonte) if fonte.isdigit() else None
    if indice is None:
        indice = 0
    checar_camera(indice, cfg)
    checar_portas()

    print("\n" + "=" * 68)
    if problemas:
        print(f"  {len(problemas)} problema(s) a resolver antes de começar:")
        for p in problemas:
            print(f"    - {p}")
        print("=" * 68)
        return 1

    print("  Tudo pronto. Abra três terminais, com o venv ativado em cada:")
    print()
    print("    1)  python -m uvicorn api:app --host 127.0.0.1 --port 8000")
    print("    2)  python worker.py")
    print("    3)  python -m streamlit run panel/app.py")
    print()
    print("  O painel abre em http://localhost:8501")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
