# Reconhecimento Facial — Câmera IP Intelbras

Cadastra e reconhece rostos de pessoas que passam por um ambiente, usando uma
câmera IP Intelbras (RTSP). Cada passagem é registrada com **foto + horário**.

- **Engine:** YuNet (detecção) + SFace (reconhecimento) via OpenCV DNN — leve,
  roda em CPU, sem GPU/dlib/onnxruntime. Pensado para o **Raspberry Pi 3B**.
- **Painel web:** Streamlit (cadastro ao vivo, histórico, gestão de pessoas).
- **Duas fases:** (1) testar **tudo num computador só**; (2) exportar **o mesmo código**
  para o Pi, deixando só o painel no PC. A única mudança é `api.base_url`.

## Arquitetura

```
Câmera Intelbras ──RTSP──► worker.py ──┐
                                       ├─ data.db (SQLite) + data/snapshots/
                           api.py  ────┘        ▲
                              ▲ HTTP            │ (worker grava data/live.jpg)
                              │
                        panel/app.py (Streamlit)
```

- `worker.py` mantém o stream e faz o reconhecimento (grava eventos + snapshots).
- `api.py` (FastAPI) faz cadastro ao vivo e serve eventos/fotos/preview.
- `panel/app.py` é um cliente HTTP fino (não usa OpenCV nem toca a câmera).

## Componentes

| Arquivo | Papel |
|---|---|
| `core/camera.py` | Leitor RTSP em thread (só o último frame, reconecta sozinho) |
| `core/face_engine.py` | YuNet + SFace + match por cosseno |
| `core/tracker.py` | Rastreia rostos entre frames e guarda os melhores recortes |
| `core/database.py` | SQLite (people / embeddings / events) em WAL |
| `core/storage.py` | Salva snapshots em `data/snapshots/AAAAMMDD/` |
| `worker.py` | Loop de reconhecimento contínuo |
| `api.py` | API HTTP (cadastro, pessoas, eventos, snapshots, live) |
| `panel/app.py` | Painel Streamlit — chamada com correção manual, cadastro, histórico |
| `scripts/test_camera.py` | Testa a conexão (RTSP, webcam USB ou arquivo) |
| `scripts/find_camera.py` | Varre a rede e descobre se a câmera expõe RTSP |
| `scripts/check_pi.py` | Diagnóstico do Pi + medição de FPS real (abre a câmera) |
| `scripts/monitor.py` | Acompanha CPU, temperatura, memória e fila **com o worker rodando** |
| `scripts/calibrate_threshold.py` | Calibra o limiar a partir dos acertos e erros reais |
| `scripts/recognize_batch.py` | Fase 2 do modo captura: reconhece por votação e gera a presença |
| `scripts/set_mode.py` | Alterna entre `realtime` e `captura` sem reiniciar o serviço |
| `tests/test_all.py` | Suíte de regressão (roda sem câmera e sem os modelos) |
| `scripts/cleanup_snapshots.py` | Retenção de snapshots (protege o cartão SD) |
| `models/download_models.py` | Baixa e valida os modelos ONNX |
| `install_pi.sh` | Instalação automatizada no Raspberry Pi |
| `config.pi.example.yaml` | Preset de configuração para Pi 3B |
| **[`GUIA_RASPBERRY_PI.md`](GUIA_RASPBERRY_PI.md)** | **Guia completo do Pi + câmera Intelbras** |
| **[`ARQUITETURA.md`](ARQUITETURA.md)** | **Algoritmos, estruturas de dados e decisões do `core/`** |

---

## Fase 1 — rodar tudo no seu computador

Serve para testar sem o Raspberry Pi: worker, API e painel na mesma máquina.
Funciona em **Windows, Linux ou macOS**.

> ⚠️ Use **Python 3.11 ou mais recente**, baixado de
> [python.org](https://www.python.org/downloads/) — no Windows, marque
> **"Add python.exe to PATH"** no instalador. Evite a versão da Microsoft Store:
> o atalho dela se comporta de forma estranha no Git Bash.

**Windows** (PowerShell ou Git Bash, na pasta do projeto):

```bash
# 1) ambiente virtual
python -m venv .venv
.venv\Scripts\activate            # PowerShell
# source .venv/Scripts/activate   # Git Bash

# 2) dependências (worker+API e painel no mesmo venv, para testar local)
python -m pip install -r requirements-pi.txt -r requirements-panel.txt

# 3) baixar os modelos
python models/download_models.py

# 4) configurar a fonte de vídeo
copy config.example.yaml config.yaml
#   edite config.yaml:
#     câmera IP  -> camera.rtsp_url com a URL RTSP (use subtype=1)
#     webcam     -> camera.rtsp_url: 0
#   e mantenha api.base_url = http://localhost:8000

# 5) testar a câmera
python scripts/test_camera.py     # deve salvar data/test_frame.jpg
```

**Linux ou macOS:** o mesmo, trocando `.venv\Scripts\activate` por
`source .venv/bin/activate` e `copy` por `cp`.

Agora abra **3 terminais** (com o venv ativado em cada um):

```bash
# terminal A — API
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# terminal B — worker (reconhecimento)
python worker.py

# terminal C — painel
python -m streamlit run panel/app.py    # abre http://localhost:8501
```

No painel: **Cadastrar** uma pessoa, depois ver **Reconhecimentos** e **Ao vivo**.

---

## Fase 2 — exportar para o Raspberry Pi

> 📖 O passo a passo detalhado, a configuração da câmera Intelbras e a solução
> de problemas estão em **[GUIA_RASPBERRY_PI.md](GUIA_RASPBERRY_PI.md)**.

Requer **Raspberry Pi OS Bookworm 64-bit** (`uname -m` = `aarch64`). Copie o
projeto **sem** o `.venv` e rode o instalador:

```bash
# do seu PC
rsync -av --exclude .venv --exclude __pycache__ --exclude .git \
      ./rc-realtime-processor/ usuario@IP_DO_PI:~/rc-realtime-processor/

# no Pi
cd ~/rc-realtime-processor
chmod +x install_pi.sh
./install_pi.sh --rtsp "rtsp://admin:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1"
```

O `install_pi.sh` valida o hardware, instala o OpenCV correto, baixa os modelos,
cria o `config.yaml` a partir do preset de Pi, testa a câmera e liga os serviços
`facial-worker`, `facial-api` e a limpeza diária de snapshots.

> ⚠️ **Não use o `python3-opencv` do apt.** No Bookworm ele é a versão 4.6, e o
> detector YuNet `2023mar` só carrega no **OpenCV ≥ 4.8**. Em 64-bit o PyPI tem
> wheel pronto (`opencv-contrib-python-headless`), então nada é compilado.

No **PC** fica só o painel:

```bash
pip install -r requirements-panel.txt
# em config.yaml: api.base_url = http://IP_DO_PI:8000
streamlit run panel/app.py
```

Nenhuma mudança de código entre as fases — só `config.yaml`.

Diagnóstico e monitoramento no Pi:

```bash
.venv/bin/python scripts/check_pi.py     # ambiente, modelos, câmera e FPS real
sudo journalctl -u facial-worker -f      # log ao vivo
curl http://localhost:8000/health        # saúde da API + idade do preview
```

---

## URL RTSP da Intelbras (firmware base Dahua)

```
substream (leve):   rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1
principal (HD):     rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=0
```

Use o **substream** para aliviar a CPU (essencial no Pi 3B). Configure-o na
câmera como **H.264** (não H.265), **640x480**, **10 fps** — o Pi 3B não
decodifica H.265 em software com folga.

Senha com caractere especial precisa vir **codificada na URL** (`@` → `%40`,
`#` → `%23`). É a causa mais comum de falha de conexão.

## Ajuste de performance (Pi 3B) — em `config.yaml`

- `worker.min_interval_seconds`: teto de processamentos por segundo — o **freio
  principal**. ↑ = mais leve.
- `worker.process_every_n_frames`: processa 1 a cada N frames recebidos. ↑ = mais leve.
- `models.detect_width`: ↓ detecta em imagem menor (mais rápido).
- `worker.opencv_threads`: 3 no Pi 3B (deixa 1 núcleo para o decode do RTSP).
- `recognition.cosine_threshold`: calibre com fotos reais (0.363 é o ponto de partida;
  ↑ = mais rígido/menos falsos positivos; ↓ = mais tolerante).
- `worker.event_cooldown_seconds`: janela anti-duplicação por pessoa.
- `storage.live_path`: no Pi aponte para `/dev/shm/...` (tmpfs) — evita ~172 mil
  escritas por dia no cartão SD.

Expectativa realista no Pi 3B: **2 a 3 reconhecimentos por segundo** (o dobro com
`models/download_models.py --int8`). Meça o seu com `scripts/check_pi.py`.

## ⚠️ LGPD / Privacidade

Imagens de rosto são **dado biométrico sensível**. Garanta base legal/consentimento,
sinalize o ambiente ("local monitorado por reconhecimento facial") e defina retenção
e limpeza dos snapshots em `data/snapshots/`.
