# Reconhecimento Facial — Câmera IP Intelbras

Cadastra e reconhece rostos de pessoas que passam por um ambiente, usando uma
câmera IP Intelbras (RTSP). Cada passagem é registrada com **foto + horário**.

- **Engine:** YuNet (detecção) + SFace (reconhecimento) via OpenCV DNN — leve,
  roda em CPU, sem GPU/dlib/onnxruntime. Pensado para o **Raspberry Pi 3B**.
- **Painel web:** Streamlit (cadastro ao vivo, histórico, gestão de pessoas).
- **Duas fases:** (1) testar **tudo no Mac**; (2) exportar **o mesmo código**
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
| `core/database.py` | SQLite (people / embeddings / events) em WAL |
| `core/storage.py` | Salva snapshots em `data/snapshots/AAAAMMDD/` |
| `worker.py` | Loop de reconhecimento contínuo |
| `api.py` | API HTTP (cadastro, pessoas, eventos, snapshots, live) |
| `panel/app.py` | Painel Streamlit |
| `scripts/test_camera.py` | Testa a conexão RTSP |
| `models/download_models.py` | Baixa os modelos ONNX |

---

## Fase 1 — rodar tudo no Mac

> ⚠️ Use **Python 3.11 ou 3.12**. O Python 3.14 do sistema ainda não tem wheels
> de OpenCV. Instale com `brew install python@3.12`.

```bash
cd ~/Desktop/reconhecimento_facial_ia

# 1) ambiente virtual com Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 2) dependências (worker+API e painel no mesmo venv para testar local)
pip install -r requirements-pi.txt -r requirements-panel.txt

# 3) baixar os modelos
python models/download_models.py

# 4) configurar a câmera
cp config.example.yaml config.yaml
#   edite config.yaml: camera.rtsp_url (use o subtype=1) e mantenha
#   api.base_url = http://localhost:8000

# 5) testar a câmera
python scripts/test_camera.py     # deve salvar data/test_frame.jpg
```

Agora abra **3 terminais** (com o venv ativado em cada um):

```bash
# terminal A — API
uvicorn api:app --host 0.0.0.0 --port 8000

# terminal B — worker (reconhecimento)
python worker.py

# terminal C — painel
streamlit run panel/app.py        # abre http://localhost:8501
```

No painel: **Cadastrar** uma pessoa, depois ver **Reconhecimentos** e **Ao vivo**.

---

## Fase 2 — exportar para o Raspberry Pi

No **Pi** (Raspberry Pi OS Bookworm 64-bit, Python 3.11) ficam o worker + API:

```bash
# OpenCV do sistema (evita compilar no Pi):
sudo apt update && sudo apt install -y python3-opencv

python3 -m venv .venv --system-site-packages   # reaproveita o opencv do apt
source .venv/bin/activate
# no requirements-pi.txt, comente a linha opencv-contrib-python e:
pip install -r requirements-pi.txt

python models/download_models.py
cp config.example.yaml config.yaml   # ajuste a rtsp_url

# habilite os serviços (ajuste usuário/caminhos nos arquivos .service)
sudo cp systemd/facial-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now facial-api facial-worker
```

No **PC** fica só o painel:

```bash
pip install -r requirements-panel.txt
# em config.yaml: api.base_url = http://IP_DO_PI:8000
streamlit run panel/app.py
```

Nenhuma mudança de código entre as fases — só `config.yaml`.

---

## URL RTSP da Intelbras (firmware base Dahua)

```
substream (leve):   rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1
principal (HD):     rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=0
```

Use o **substream** para aliviar a CPU (essencial no Pi 3B).

## Ajuste de performance (Pi 3B) — em `config.yaml`

- `worker.process_every_n_frames`: ↑ processa menos frames (mais leve).
- `models.detect_width`: ↓ detecta em imagem menor (mais rápido).
- `recognition.cosine_threshold`: calibre com fotos reais (0.363 é o ponto de partida;
  ↑ = mais rígido/menos falsos positivos; ↓ = mais tolerante).
- `worker.event_cooldown_seconds`: janela anti-duplicação por pessoa.

## ⚠️ LGPD / Privacidade

Imagens de rosto são **dado biométrico sensível**. Garanta base legal/consentimento,
sinalize o ambiente ("local monitorado por reconhecimento facial") e defina retenção
e limpeza dos snapshots em `data/snapshots/`.
