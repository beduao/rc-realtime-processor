#!/usr/bin/env bash
#
# Instalador para Raspberry Pi (worker + API de reconhecimento facial).
#
# Uso, dentro da pasta do projeto no Pi:
#     chmod +x install_pi.sh
#     ./install_pi.sh
#
# Opções:
#     --rtsp "rtsp://..."   já grava a URL da câmera no config.yaml
#     --int8                usa os modelos quantizados (mais rápido no Pi 3B)
#     --skip-camera-test    não tenta conectar na câmera durante a instalação
#     --reinstall           recria o ambiente virtual do zero
#     --no-services         instala tudo mas não mexe no systemd
#     --port 8000           porta da API (padrão 8000)
#     --retention-days 30   dias de snapshots mantidos pela limpeza diária
#     --max-mb 2000         teto de tamanho da pasta de snapshots
#
set -euo pipefail

# ---------------------------------------------------------------- aparência --
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi
step()  { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$RESET"; }
ok()    { printf '%s  ok%s  %s\n'   "$GREEN" "$RESET" "$*"; }
warn()  { printf '%s  !!%s  %s\n'   "$YELLOW" "$RESET" "$*"; }
die()   { printf '\n%s  xx  %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }

# ------------------------------------------------------------------ opções ---
RTSP_URL="${RTSP_URL:-}"
USE_INT8=0
SKIP_CAM=0
REINSTALL=0
DO_SERVICES=1
API_PORT=8000
RETENTION_DAYS=30
MAX_MB=2000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtsp)            RTSP_URL="${2:-}"; shift 2 ;;
    --int8)            USE_INT8=1; shift ;;
    --skip-camera-test) SKIP_CAM=1; shift ;;
    --reinstall)       REINSTALL=1; shift ;;
    --no-services)     DO_SERVICES=0; shift ;;
    --port)            API_PORT="${2:-8000}"; shift 2 ;;
    --retention-days)  RETENTION_DAYS="${2:-30}"; shift 2 ;;
    --max-mb)          MAX_MB="${2:-2000}"; shift 2 ;;
    -h|--help)         sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 die "opção desconhecida: $1  (use --help)" ;;
  esac
done

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
VENV="$PROJ/.venv"
PY="$VENV/bin/python"

# ------------------------------------------------------- 1. sanidade do host -
step "1/9  Checando o sistema"

[[ "$EUID" -ne 0 ]] || die "não rode como root. Rode como usuário normal; o script pede sudo quando precisa."

ARCH="$(uname -m)"
KERNEL="$(uname -r)"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo 'desconhecido')"
CODENAME="$(. /etc/os-release 2>/dev/null && echo "${VERSION_CODENAME:-?}")"
PYVER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"

printf '      modelo:   %s\n' "$MODEL"
printf '      arch:     %s (kernel %s)\n' "$ARCH" "$KERNEL"
printf '      SO:       Debian/%s | Python %s\n' "$CODENAME" "$PYVER"
printf '      usuário:  %s\n' "$RUN_USER"
printf '      projeto:  %s\n' "$PROJ"

case "$ARCH" in
  aarch64|arm64)
    ok "64-bit: existe wheel pronto do OpenCV no PyPI (nada será compilado)" ;;
  armv7l|armv6l)
    warn "Sistema 32-bit. O PyPI não tem wheel de OpenCV >= 4.8 para armv7l;"
    warn "o pip vai tentar o piwheels e pode falhar ou demorar muito."
    warn "Recomendado: regravar o cartão com Raspberry Pi OS 64-bit (Pi 3B suporta)."
    read -r -p "      Continuar mesmo assim? [s/N] " a
    [[ "${a,,}" == "s" ]] || exit 1 ;;
  *)
    warn "arquitetura inesperada: $ARCH — seguindo em frente" ;;
esac

# espaço em disco: venv (~250 MB) + modelos (~40 MB) + folga p/ snapshots
FREE_MB=$(df -Pm "$PROJ" | awk 'NR==2{print $4}')
printf '      livre:    %s MB\n' "$FREE_MB"
[[ "$FREE_MB" -ge 700 ]] || die "espaço insuficiente (${FREE_MB} MB). Libere pelo menos 700 MB."

TOTAL_RAM=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)
if [[ "$TOTAL_RAM" -lt 1200 ]]; then
  warn "RAM total ${TOTAL_RAM} MB. Se o desktop gráfico estiver ativo, desligue-o:"
  warn "  sudo raspi-config  ->  System Options  ->  Boot / Auto Login  ->  Console"
fi

# relógio: os eventos são gravados com a hora do Pi (que não tem bateria de RTC)
TZ_NOW="$(timedatectl show -p Timezone --value 2>/dev/null || echo '?')"
NTP_OK="$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo '?')"
printf '      hora:     %s (NTP sincronizado: %s)\n' "$TZ_NOW" "$NTP_OK"
[[ "$TZ_NOW" == "America/"* ]] || warn "fuso horário é $TZ_NOW — ajuste com: sudo timedatectl set-timezone America/Sao_Paulo"
[[ "$NTP_OK" == "yes" ]] || warn "relógio ainda não sincronizou; os horários dos eventos podem sair errados"

# ------------------------------------------------------ 2. pacotes do sistema -
step "2/9  Instalando pacotes do sistema (pede sudo)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev python3-pip ca-certificates ffmpeg
ok "python3-venv, python3-dev, ffmpeg instalados"

# ------------------------------------------------------- 3. ambiente virtual --
step "3/9  Ambiente virtual"

# Um .venv copiado de outra máquina (ex.: do Mac) tem caminhos e binários
# inválidos — é a causa nº 1 de "funcionava lá e aqui não".
if [[ -d "$VENV" ]]; then
  if [[ "$REINSTALL" -eq 1 ]] || ! "$PY" -c 'import sys' 2>/dev/null; then
    warn "removendo .venv existente (inválido ou --reinstall)"
    rm -rf "$VENV"
  fi
fi
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  ok "criado em $VENV"
else
  ok "reaproveitando o .venv existente"
fi

"$PY" -m pip install --quiet --upgrade pip wheel
ok "pip $("$PY" -m pip --version | awk '{print $2}')"

# ---------------------------------------------------- 4. dependências Python --
step "4/9  Dependências Python (o OpenCV são ~90 MB, tenha paciência)"
"$PY" -m pip install --quiet -r "$PROJ/requirements-pi.txt" \
  || die "falha ao instalar as dependências. Rode sem --quiet para ver o erro:
       $PY -m pip install -r $PROJ/requirements-pi.txt"

CVVER="$("$PY" -c 'import cv2;print(cv2.__version__)')"
"$PY" - <<'EOF' || die "OpenCV instalado não serve para este projeto (veja a mensagem acima)."
import sys, cv2
major, minor = (int(x) for x in cv2.__version__.split(".")[:2])
if (major, minor) < (4, 8):
    sys.exit(f"OpenCV {cv2.__version__} < 4.8: o YuNet 2023mar nao carrega nesta versao.")
for attr in ("FaceDetectorYN", "FaceRecognizerSF"):
    if not hasattr(cv2, attr):
        sys.exit(f"cv2.{attr} ausente: instale opencv-CONTRIB-python-headless.")
EOF
ok "OpenCV $CVVER com FaceDetectorYN + FaceRecognizerSF"

# O wheel do PyPI já traz FFmpeg embutido — sem isso não há RTSP.
if "$PY" - <<'EOF'
import cv2, sys
line = next((l for l in cv2.getBuildInformation().splitlines() if "FFMPEG" in l), "")
sys.exit(0 if "YES" in line.upper() else 1)
EOF
then
  ok "suporte a FFmpeg presente (necessário para RTSP)"
else
  warn "não confirmei o suporte a FFmpeg no OpenCV — o RTSP pode não abrir"
fi

# ----------------------------------------------------------- 5. modelos ONNX --
step "5/9  Modelos (YuNet + SFace)"
if [[ "$USE_INT8" -eq 1 ]]; then
  "$PY" "$PROJ/models/download_models.py" --int8 || die "falha ao baixar os modelos"
else
  "$PY" "$PROJ/models/download_models.py" || die "falha ao baixar os modelos"
fi
ok "modelos prontos"

# ------------------------------------------------------------ 6. config.yaml --
step "6/9  Configuração"
CFG="$PROJ/config.yaml"
if [[ ! -f "$CFG" ]]; then
  cp "$PROJ/config.pi.example.yaml" "$CFG"
  ok "config.yaml criado a partir de config.pi.example.yaml"
else
  ok "config.yaml já existe (mantido como está)"
fi

if [[ -z "$RTSP_URL" ]] && grep -q "SENHA_CODIFICADA" "$CFG" && [[ -t 0 ]]; then
  printf '\n      Cole a URL RTSP da câmera Intelbras (substream, subtype=1).\n'
  printf '      Formato: rtsp://usuario:senha@IP:554/cam/realmonitor?channel=1&subtype=1\n'
  printf '      Senha com @ # : / precisa vir codificada (@ = %%40, # = %%23).\n'
  read -r -p "      URL (Enter para editar depois): " RTSP_URL
fi

if [[ -n "$RTSP_URL" ]]; then
  "$PY" - "$CFG" "$RTSP_URL" <<'EOF'
import re, sys
path, url = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
new, n = re.subn(r'(?m)^(\s*rtsp_url:\s*).*$', lambda m: m.group(1) + '"' + url + '"', text, count=1)
open(path, "w", encoding="utf-8").write(new)
print("      rtsp_url atualizada" if n else "      AVISO: nao achei a linha rtsp_url")
EOF
fi

if [[ "$USE_INT8" -eq 1 ]]; then
  sed -i \
    -e 's#^\(\s*detector:\s*\).*#\1"models/face_detection_yunet_2023mar_int8.onnx"#' \
    -e 's#^\(\s*recognizer:\s*\).*#\1"models/face_recognition_sface_2021dec_int8.onnx"#' \
    "$CFG"
  ok "config apontando para os modelos int8"
  warn "int8 muda a escala do score: recalibre recognition.cosine_threshold"
fi

mkdir -p "$PROJ/data/snapshots"
ok "pasta data/ pronta"

if grep -q "SENHA_CODIFICADA" "$CFG"; then
  warn "a URL da câmera ainda é o exemplo — edite $CFG antes de subir os serviços"
  SKIP_CAM=1
fi

# --------------------------------------------------------- 7. teste da câmera -
step "7/9  Teste da câmera"
if [[ "$SKIP_CAM" -eq 1 ]]; then
  warn "pulado (--skip-camera-test ou URL não configurada)"
else
  if "$PY" "$PROJ/scripts/test_camera.py"; then
    ok "câmera respondendo"
  else
    warn "não consegui ler a câmera. Os serviços serão instalados, mas revise a URL."
    warn "Diagnóstico detalhado:  $PY scripts/check_pi.py"
  fi
fi

# -------------------------------------------------------------- 8. systemd ----
step "8/9  Serviços (systemd)"
if [[ "$DO_SERVICES" -eq 0 ]]; then
  warn "pulado (--no-services)"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  # O `tr -d '\r'` é proteção contra arquivos vindos do Windows: um CRLF numa
  # unit faz o \r entrar no argumento do ExecStart e o serviço não sobe.
  for unit in facial-api.service facial-worker.service facial-cleanup.service; do
    tr -d '\r' < "$PROJ/systemd/$unit" | sed \
        -e "s|__USER__|$RUN_USER|g" \
        -e "s|__DIR__|$PROJ|g" \
        -e "s|__PORT__|$API_PORT|g" \
        -e "s|__RETENTION_DAYS__|$RETENTION_DAYS|g" \
        -e "s|__MAX_MB__|$MAX_MB|g" \
        > "$TMP/$unit"
  done
  tr -d '\r' < "$PROJ/systemd/facial-cleanup.timer" > "$TMP/facial-cleanup.timer"

  sudo cp "$TMP"/facial-*.service "$TMP"/facial-*.timer /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now facial-api.service facial-worker.service >/dev/null
  sudo systemctl enable --now facial-cleanup.timer >/dev/null
  ok "facial-api, facial-worker e facial-cleanup.timer habilitados no boot"

  sleep 4
  for svc in facial-api facial-worker; do
    if systemctl is-active --quiet "$svc"; then
      ok "$svc está rodando"
    else
      warn "$svc NÃO subiu. Veja o motivo com:  sudo journalctl -u $svc -n 40 --no-pager"
    fi
  done
fi

# --------------------------------------------------------------- 9. resumo ----
step "9/9  Pronto"
IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

  ${BOLD}API do Pi:${RESET}    http://${IP:-IP_DO_PI}:${API_PORT}
  ${BOLD}Saúde:${RESET}        http://${IP:-IP_DO_PI}:${API_PORT}/health
  ${BOLD}Preview:${RESET}      http://${IP:-IP_DO_PI}:${API_PORT}/live.jpg

  ${BOLD}No seu PC${RESET} (painel de cadastro e histórico):
     pip install -r requirements-panel.txt
     # no config.yaml do PC:  api.base_url: "http://${IP:-IP_DO_PI}:${API_PORT}"
     streamlit run panel/app.py

  ${BOLD}Comandos úteis no Pi${RESET}
     sudo journalctl -u facial-worker -f      # log do reconhecimento
     sudo journalctl -u facial-api -f         # log da API
     sudo systemctl restart facial-worker     # aplicar mudança no config.yaml
     $PY scripts/check_pi.py                  # diagnóstico + medição de FPS

  Cadastre as pessoas pelo painel antes de esperar reconhecimentos.
  Rostos são dado biométrico sensível: sinalize o ambiente e defina retenção.

EOF
