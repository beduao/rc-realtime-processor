# Rodando no Raspberry Pi 3B com câmera Intelbras

Guia completo: do cartão SD ao reconhecimento funcionando. O código já estava
preparado para o Pi; este guia cobre o que faltava e como operar.

---

## 1. O que roda onde

```
Câmera Intelbras ──RTSP──►  RASPBERRY PI 3B          SEU PC
                            ├─ worker.py             └─ panel/app.py (Streamlit)
                            ├─ api.py (porta 8000)      cadastro + histórico
                            └─ data.db + snapshots      aponta p/ http://IP_DO_PI:8000
```

O Pi faz o trabalho pesado e guarda os dados. O painel é só um cliente HTTP —
não precisa de OpenCV nem toca a câmera.

---

## 2. Pré-requisitos (checar antes)

| Item | Exigência | Por quê |
|---|---|---|
| Sistema | **Raspberry Pi OS 64-bit (Bookworm)** | Só no 64-bit existe wheel pronto do OpenCV ≥ 4.8. No 32-bit o pip tenta compilar e isso leva horas (quando dá certo). |
| Modo de boot | **Console, sem desktop** | O desktop come 200–300 MB de 1 GB. `sudo raspi-config` → System Options → Boot / Auto Login → **Console** |
| Rede | **Cabo Ethernet** (preferível) | No Pi 3B o Wi‑Fi compartilha o barramento USB com a rede; stream RTSP contínuo sofre quedas. |
| Fonte | **5 V / 2,5 A original** | Subtensão trava o decode do vídeo de forma intermitente e difícil de diagnosticar. |
| Hora | Fuso e NTP corretos | Os eventos são gravados com a hora do Pi, que **não tem bateria de relógio**. `sudo timedatectl set-timezone America/Sao_Paulo` |
| Cartão SD | Classe 10 / A1, com folga | Snapshots gravam em disco continuamente. |

Verificar o que você tem:

```bash
uname -m                    # precisa dizer aarch64
cat /etc/os-release          # VERSION_CODENAME=bookworm
free -m                      # RAM disponível
timedatectl                  # fuso e sincronização
vcgencmd get_throttled       # 0x0 = fonte e temperatura ok
```

---

## 3. Configurar a câmera Intelbras primeiro

Faça isso **antes** de instalar no Pi: economiza horas de diagnóstico.

Entre na interface web da câmera pelo navegador (`http://IP_DA_CAMERA`). O
firmware Intelbras das linhas VIP/VHD é baseado em Dahua, então os menus têm
esses nomes (variam com o modelo e a versão):

### 3.1 Ajustar o **stream extra** (substream)

`Configurações → Câmera → Vídeo → Stream Extra`

| Parâmetro | Valor | Motivo |
|---|---|---|
| Habilitar | Sim | É o stream que o Pi vai consumir. |
| Codec / Compressão | **H.264** (nunca H.265/HEVC) | O Pi 3B não tem decode acelerado de H.265; em software ele não dá conta. |
| Resolução | **640x480** (ou D1/704x480) | Acima disso a CPU do Pi 3B não acompanha. |
| Taxa de quadros (FPS) | **10** | Mais que isso é desperdício: o Pi processa ~2 frames/s. |
| Tipo de taxa de bits | CBR | Estabiliza a rede. |
| Taxa de bits | 512–1024 kbps | Suficiente nessa resolução. |
| Intervalo de I‑frame (GOP) | 10–20 | I‑frame frequente = reconexão mais rápida. |

#### Se a câmera não tem interface web

Alguns modelos (linha Mibo, por exemplo) expõem RTSP mas **não** sobem servidor
web na rede local — a porta 80 aceita a conexão e derruba assim que você fala
HTTP (`ERR_CONNECTION_RESET` no navegador). Nesse caso você não consegue mexer
nos parâmetros da tabela acima; resta trabalhar com o que a câmera entrega:

1. Veja o que o **app** oferece. Uma opção de qualidade (HD/SD, "alta/baixa
   resolução") normalmente alterna entre stream principal e substream, e é o
   único controle disponível.
2. Descubra o que o substream realmente entrega, rodando no Pi:

   ```bash
   .venv/bin/python scripts/test_camera.py    # imprime a resolução real
   .venv/bin/python scripts/check_pi.py       # mede fps e ms por frame
   ```

3. Compense pelo `config.yaml`. Se o substream vier maior que 640x480, o
   `models.detect_width` já reduz a imagem antes de detectar — o custo extra
   fica só no decode. Se o `check_pi.py` acusar que o Pi não acompanha, aumente
   `worker.min_interval_seconds`.

O que **não** dá para compensar é H.265: se o `check_pi.py` mostrar menos de 3
fps chegando e o app não permitir trocar o codec, aí a saída é a webcam USB
(seção 3.6).

### 3.2 Garantir que o RTSP está ativo

Em `Configurações → Rede → Portas` confirme **RTSP = 554**.

Em alguns firmwares mais novos há um menu de serviços
(`Sistema → Segurança → Serviços` ou `Rede → Acesso à plataforma`) onde o RTSP
e/ou o ONVIF vêm **desabilitados de fábrica** — habilite se for o caso.

> ⚠️ Se a sua câmera for da linha **Mibo** (câmeras de nuvem: iM3, iM4, iM5,
> IMX…), provavelmente ela **não** tem interface web nem RTSP na rede local.
> Vá direto para a **seção 3.5**, que mostra como confirmar isso e quais são as
> alternativas.

### 3.3 Montar a URL RTSP

```
substream (use esta):  rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=1
stream principal:      rtsp://USUARIO:SENHA@IP:554/cam/realmonitor?channel=1&subtype=0
```

**Atenção com a senha** — a URL quebra se ela tiver caracteres especiais sem
codificar. Este é o erro nº 1 nesse tipo de integração:

| Caractere | Escreva |
|---|---|
| `@` | `%40` |
| `#` | `%23` |
| `:` | `%3A` |
| `/` | `%2F` |
| `?` | `%3F` |
| `%` | `%25` |
| espaço | `%20` |

Exemplo: senha `Th@2024#` → `rtsp://admin:Th%402024%23@192.168.1.77:554/cam/...`

Gerar a URL codificada sem errar:

```bash
python3 -c "import urllib.parse as u; print(u.quote(input('senha: '), safe=''))"
```

### 3.4 Testar antes de envolver o Pi

Do seu PC (ou já do Pi, via SSH):

```bash
ffprobe -rtsp_transport tcp "rtsp://admin:SENHA@192.168.1.77:554/cam/realmonitor?channel=1&subtype=1"
```

Se aparecer `Stream #0:0: Video: h264 ... 640x480`, está pronto. Se der
`401 Unauthorized`, a senha está errada ou mal codificada. Se der timeout, é
IP/porta/rede ou o RTSP está desabilitado na câmera.

### 3.5 Se você não consegue acessar a câmera pelo IP (linha Mibo)

Sintoma: `http://IP_DA_CAMERA` não abre nada no navegador e o único acesso é
pelo aplicativo **Mibo Smart**.

A linha **Mibo** (iM3, iM4, iM5, IMX…) é a linha de consumo da Intelbras: são
câmeras de nuvem, pensadas para funcionar pelo app. Diferente das linhas
profissionais VIP/VHD, elas geralmente **não sobem servidor web nem RTSP na rede
local** — a câmera abre uma conexão de saída para a nuvem e é por ali que o app
conversa com ela. Se for esse o caso, não existe URL RTSP para o Pi consumir, e
não é questão de achar o endereço certo.

**Prove antes de desistir**, porque isso varia com modelo e firmware. O
`find_camera.py` usa só a biblioteca padrão do Python na varredura, então roda
de qualquer máquina **que esteja na mesma rede da câmera** — inclusive do seu PC,
antes de mexer no Pi.

```bash
# No Windows (PowerShell ou Git Bash), na pasta do projeto:
python scripts/find_camera.py
#   se "python" não for reconhecido, tente:  py scripts/find_camera.py

# No Raspberry Pi (o Python fica no .venv, não no sistema):
.venv/bin/python scripts/find_camera.py
```

Ele varre a sua sub-rede, lista o que responde e classifica cada equipamento.
Se souber o IP da câmera (o app costuma mostrar em *Configurações do
dispositivo → Informações*, ou veja a lista de clientes DHCP do seu roteador):

```bash
python scripts/find_camera.py --ip 192.168.1.77
# e, se a porta 554 aparecer aberta, teste os caminhos RTSP:
python scripts/find_camera.py --ip 192.168.1.77 --user admin --password "SUA_SENHA"
```

Como ler o resultado:

| Veredito | O que significa |
|---|---|
| `TEM RTSP` | Porta 554 aberta. Rode de novo com `--user` e `--password` para achar o caminho. |
| `sem RTSP, mas com porta de ONVIF/SDK` | Vale procurar no app/interface uma opção de habilitar RTSP. |
| `responde em HTTP mas com RTSP fechado` | Comportamento de câmera de nuvem. Veja a seção 3.6. |
| `não respondeu em nenhuma porta` | IP errado, câmera desligada, ou isolamento de clientes no roteador. |

O teste de credenciais fala RTSP direto no socket (`DESCRIBE` com autenticação
Digest ou Basic), sem precisar de OpenCV. Isso dá a informação que realmente
separa as causas:

| Resposta | Significado |
|---|---|
| `200 OK` | Caminho **e** credencial corretos. O script imprime a linha do `config.yaml`. |
| `401 Unauthorized` | Servidor RTSP saudável; **usuário ou senha recusados**. O caminho nem chegou a ser avaliado. |
| `404 Not Found` | **Credencial aceita!** Só o caminho é que não existe nessa câmera. |
| erro de conexão | Nem chegou a falar RTSP: porta errada, serviço desabilitado ou firewall. |

Se der 401 em tudo, o usuário pode não ser `admin`:

```bash
python scripts/find_camera.py --ip 192.168.15.19 --try-users --password "SUA_SENHA"
```

> ⚠️ `--try-users` tenta vários usuários de fábrica. Algumas câmeras **bloqueiam
> o acesso temporariamente** após várias falhas de login — se acontecer, espere
> alguns minutos ou reinicie a câmera.

O OpenCV só é usado no final, para confirmar a captura de um frame. Sem ele o
script diz explicitamente que não confirmou a imagem, em vez de dar a conclusão
por certa.

Nesse segundo modo ele testa 13 caminhos RTSP conhecidos (Intelbras/Dahua,
Hikvision, ONVIF genérico) e imprime a linha pronta para o `config.yaml`.

Vale também procurar no app, em *Configurações do dispositivo*, por algo como
**RTSP**, **ONVIF**, **Rede local** ou **Modo local**. Alguns firmwares têm a
opção escondida ali. Se existir, habilite e rode o `find_camera.py` de novo.

**Se a porta 554 estiver aberta, ótimo: ela tem RTSP** e a linha Mibo não é
impedimento nesse modelo. Falta descobrir as **credenciais da câmera** — que
não têm nenhuma relação com o login do Windows nem com a conta do app:

- Se a **porta 80 também estiver aberta**, abra `http://IP_DA_CAMERA` no
  navegador. É o caminho mais direto: dá para confirmar/alterar a senha e
  configurar o substream (seção 3.1). Se antes não abriu, provavelmente foi
  tentado o IP errado — use o que o `find_camera.py` apontou.
- Procure uma **etiqueta no corpo da câmera** com *senha*, *password* ou
  *código de verificação*.
- No app, veja *Configurações do dispositivo* → algo como **senha do
  dispositivo** ou **código de verificação**.
- O usuário é quase sempre `admin`.

A porta **37777** aberta é um bom sinal: é a porta do SDK Dahua, o que indica
firmware base Dahua e que o caminho
`/cam/realmonitor?channel=1&subtype=1` deve funcionar.

### 3.6 Alternativas quando a câmera não tem RTSP

Se a porta 554 estiver fechada e não houver opção no app, estas são as saídas —
em ordem de custo/benefício para o Pi 3B:

| Alternativa | Custo | Observações |
|---|---|---|
| **Webcam USB** no próprio Pi | ~R$ 60–150 | **Mais leve de todas**: elimina o decode de H.264, que é o maior consumo de CPU do Pi 3B. Sobra processamento para o reconhecimento. Limita a posição da câmera ao alcance do cabo USB. |
| **Câmera CSI** (Pi Camera / OV5647) | ~R$ 80–150 | Mesma vantagem de CPU, cabo flat curto (15 cm padrão). Precisa de `libcamera`/V4L2 habilitado. |
| **Câmera IP com RTSP** (Intelbras VIP/VHD, ou qualquer ONVIF) | ~R$ 200–400 | O cenário para o qual este projeto foi escrito. Permite posicionar longe do Pi e usar PoE. |
| Manter a Mibo | — | Só se ela expuser RTSP. Não vale tentar capturar a tela do app: instável, alta latência e provavelmente contra os termos de uso. |

**Usar webcam USB ou CSI** — o código já aceita, basta trocar uma linha no
`config.yaml`:

```yaml
camera:
  rtsp_url: 0          # índice do dispositivo (ou "/dev/video0")
  width: 640           # só valem para dispositivo local
  height: 480
  fps: 15
  mjpeg: true          # MJPEG em vez de vídeo cru: essencial em USB 2.0
```

Descobrir qual índice usar, no Pi:

```bash
ls /dev/video*                     # dispositivos disponíveis
v4l2-ctl --list-devices            # qual é qual (sudo apt install v4l-utils)
v4l2-ctl -d /dev/video0 --list-formats-ext   # resoluções e formatos suportados
.venv/bin/python scripts/test_camera.py      # testa e salva data/test_frame.jpg
```

Depois, `sudo systemctl restart facial-worker`. Nada mais muda: cadastro,
reconhecimento, snapshots e painel funcionam igual.

> Com webcam USB, revise `models.detect_width` e `worker.min_interval_seconds`:
> sem o custo do decode de H.264 o Pi consegue processar mais frames por segundo
> do que com a câmera IP. Meça com `scripts/check_pi.py`.

### 3.7 Onde posicionar a câmera

O reconhecimento depende muito mais do enquadramento do que do código:

- **Altura do rosto**, levemente acima da linha dos olhos, não no teto. Câmera
  muito alta enxerga o topo da cabeça e o SFace erra.
- Rosto ocupando **pelo menos 80–100 px** de largura no frame quando a pessoa
  passa (com 640x480, isso é a pessoa a ~2–3 m).
- **Sem contraluz** — janela ou porta de vidro atrás da pessoa arruína o
  contraste do rosto.
- Corredor onde as pessoas passam **de frente** para a câmera, não de perfil.

---

## 4. Instalar no Pi

Copie a pasta do projeto para o Pi (do seu PC):

```bash
# ATENÇÃO: não copie a pasta .venv nem data/ do PC — o .venv não funciona em
# outra máquina e é a causa nº 1 de "no PC funcionava".
rsync -av --exclude .venv --exclude __pycache__ --exclude .git \
      ./rc-realtime-processor/ beatriz@IP_DO_PI:~/rc-realtime-processor/
```

No Pi, via SSH:

```bash
cd ~/rc-realtime-processor
sed -i 's/\r$//' install_pi.sh          # só por segurança, se o arquivo veio do Windows
chmod +x install_pi.sh
./install_pi.sh --rtsp "rtsp://admin:SENHA_CODIFICADA@192.168.1.77:554/cam/realmonitor?channel=1&subtype=1"
```

O script faz, em ordem, e para com mensagem clara se algo falhar:

1. Confere arquitetura, RAM, disco, fuso horário e subtensão.
2. Instala `python3-venv`, `python3-dev` e `ffmpeg` via apt.
3. Cria o `.venv` (descarta um `.venv` inválido copiado de outra máquina).
4. Instala o **OpenCV ≥ 4.9** via wheel do PyPI e **valida** que ele tem
   `FaceDetectorYN`, `FaceRecognizerSF` e FFmpeg.
5. Baixa e valida os modelos ONNX.
6. Cria o `config.yaml` a partir do preset de Pi e grava a URL da câmera.
7. Testa a câmera.
8. Instala e liga os serviços `facial-worker`, `facial-api` e a limpeza diária.

Opções úteis:

```bash
./install_pi.sh --int8              # modelos quantizados: ~2x mais rápido
./install_pi.sh --reinstall         # recria o .venv do zero
./install_pi.sh --skip-camera-test  # instalar sem a câmera à mão
./install_pi.sh --retention-days 15 --max-mb 1000
./install_pi.sh --help
```

### Por que o OpenCV do apt não serve

O `python3-opencv` do Bookworm é a versão **4.6**. O detector
`face_detection_yunet_2023mar.onnx` usa formas de entrada dinâmicas e **só
carrega no OpenCV ≥ 4.8** — no 4.6 ele falha com um erro obscuro de DNN. Por
isso o `requirements-pi.txt` agora fixa `opencv-contrib-python-headless>=4.9,<5`,
que tem wheel pronto para `aarch64` (nada compila). O `core/face_engine.py`
também detecta essa situação e explica o que fazer, em vez de estourar um erro
interno do OpenCV.

---

## 5. Usar: cadastrar pessoas pelo painel (no seu PC)

Sem cadastro, tudo aparece como "Desconhecido". No **seu PC**:

```bash
pip install -r requirements-panel.txt
cp config.example.yaml config.yaml      # se ainda não existir
# edite a linha:  base_url: "http://IP_DO_PI:8000"
streamlit run panel/app.py
```

No painel: **Cadastrar** → digite o nome → capture ~5 amostras com variações
leves (frontal, levemente à esquerda, à direita, com e sem óculos se for o caso)
→ **Concluir**. O worker recarrega a galeria sozinho em até 10 segundos, sem
reiniciar nada.

Verificar o Pi pelo navegador, sem painel:

- `http://IP_DO_PI:8000/health` — versão do OpenCV, nº de pessoas e a idade do
  último preview (se `worker_live_age_seconds` estiver alto ou `null`, o worker
  não está processando)
- `http://IP_DO_PI:8000/live.jpg` — o que a câmera está vendo agora
- `http://IP_DO_PI:8000/events` — últimos reconhecimentos

---

## 6. Desempenho: o que esperar do Pi 3B

Quatro núcleos a 1,2 GHz, 1 GB de RAM, sem GPU utilizável para DNN. Com
substream 640x480 e `detect_width: 320`, a ordem de grandeza é:

| Etapa | Tempo típico |
|---|---|
| Detecção (YuNet, 320 px) | 120–250 ms |
| Embedding (SFace, por rosto) | 150–350 ms |
| **Total por frame com 1 rosto** | **~0,3–0,6 s → 2 a 3 reconhecimentos/s** |

Com os modelos `--int8`, espere aproximadamente o dobro de velocidade.

Meça no **seu** hardware e enquadramento. Há duas ferramentas, com propósitos
diferentes:

**`check_pi.py` — diagnóstico com o serviço PARADO.** Ele abre a câmera para
medir fps, ms de detecção e ms de embedding, e avisa se o `config.yaml` está
pedindo mais do que o Pi entrega.

```bash
sudo systemctl stop facial-worker     # obrigatório com webcam USB
.venv/bin/python scripts/check_pi.py
sudo systemctl start facial-worker
```

Parar o worker é obrigatório com webcam USB, porque dispositivo V4L2 é
exclusivo — dois processos não abrem `/dev/video0` ao mesmo tempo. Com câmera
IP funcionaria sem parar, mas o teste competiria por CPU com o worker e as
medições sairiam pessimistas.

**`monitor.py` — acompanhamento com o serviço RODANDO.** Não toca na câmera:
lê tudo de `/proc`, do `vcgencmd`, do status do worker e do banco.

```bash
.venv/bin/python scripts/monitor.py                 # até Ctrl+C
.venv/bin/python scripts/monitor.py --segundos 300   # observa 5 min e resume
```

```
hora       cpu   temp   livre   swap  worker       cpu    mem   fps  pend  disco
11:23:53   82%  64.2C   180MB    0MB  captura     241%   210MB  15.8    12   34GB
```

O que olhar, e por quê:

| Coluna | Sinal de alerta |
|---|---|
| `temp` | Acima de 80 °C o Pi reduz o clock sozinho e tudo fica mais lento |
| `livre` | Abaixo de ~80 MB começa swap, e swap em cartão SD arruína a latência |
| `swap` | Qualquer valor crescendo é ruim |
| `worker cpu` | Pode passar de 100% (são 4 núcleos, teto 400%) |
| `pend` | Se só cresce, o lote não acompanha a captura |

No fim ele resume a temperatura máxima, se houve subtensão ou throttling, e se
a fila cresceu ou diminuiu no período. **Subtensão é o achado mais importante
ao testar com webcam USB**: a câmera divide o barramento e a alimentação com o
Pi, e fonte fraca causa falhas intermitentes difíceis de diagnosticar de outra
forma. O aviso aparece mesmo que tudo pareça estar funcionando.

### Ajustes em `config.yaml` (reinicie com `sudo systemctl restart facial-worker`)

| Chave | Efeito | Mais leve |
|---|---|---|
| `worker.min_interval_seconds` | Teto de processamentos por segundo. **Freio principal.** | aumentar (0.5 → 1.0) |
| `worker.process_every_n_frames` | Processa 1 a cada N frames recebidos | aumentar |
| `models.detect_width` | Largura em que detecta (caixas são reescaladas) | diminuir (320 → 240) |
| `models.top_k` | Máximo de rostos considerados | diminuir |
| `recognition.min_face_size` | Ignora rostos pequenos (embedding ruim mesmo) | aumentar |
| `worker.draw_annotations` | Desenha caixas e gera o preview ao vivo | `false` desliga |
| `worker.opencv_threads` | Threads da DNN; 3 deixa 1 núcleo para o decode | manter 3 |
| `worker.event_cooldown_seconds` | Janela anti-duplicação por pessoa | aumentar (menos escrita no SD) |

Trocar para os modelos quantizados:

```bash
.venv/bin/python models/download_models.py --int8
# no config.yaml:
#   detector:   "models/face_detection_yunet_2023mar_int8.onnx"
#   recognizer: "models/face_recognition_sface_2021dec_int8.onnx"
sudo systemctl restart facial-worker
```

O int8 muda a escala dos scores — **recalibre o limiar** (seção 7).

---

## 6.5 Grupos e chamada de presença — o modo `captura`

O modo padrão (`realtime`) reconhece cada rosto no instante em que ele aparece.
Isso não funciona quando várias pessoas passam juntas, e a conta explica por quê:

| Rostos no frame | Tempo para processar 1 frame |
|---|---|
| 1 | 0,34 s |
| 3 | 0,91 s |
| 5 | 1,48 s |
| 10 | 2,91 s |
| 20 | 5,76 s |

A detecção custa ~57 ms por frame **independente da quantidade** de rostos — é
uma passada única da rede. O que escala é o reconhecimento: ~285 ms **por
rosto**. E enquanto o worker processa um frame, os que chegam são descartados,
então o grupo inteiro é visto num único instantâneo. Quem estava de perfil,
atrás de alguém ou borrado naquele exato frame não é registrado.

### Como o modo `captura` resolve

Ele separa as duas fases:

```
FASE 1 (ao vivo, barata)          FASE 2 (depois, sem pressa)
detecta ~17 fps                   lê as trilhas pendentes
  -> rastreia cada pessoa           -> reconhece cada recorte
  -> guarda os 3 melhores           -> decide por VOTAÇÃO
     recortes de cada uma           -> grava a presença
```

Três ganhos, não um:

1. A captura acompanha o vídeo mesmo com o corredor cheio.
2. Cada pessoa tem **várias chances** de aparecer bem, não uma só.
3. A decisão sai por **votação** entre os recortes. Um recorte ruim vira voto
   vencido em vez de decisão final — é a correção direta para o falso positivo
   que aparece no modo em tempo real.

### Alternar entre os dois modos

Você não precisa escolher de uma vez: o modo é trocável a qualquer momento, e
o worker aplica **sem reiniciar o serviço** (ele relê a configuração a cada 10 s).

```bash
.venv/bin/python scripts/set_mode.py             # ver o modo atual
.venv/bin/python scripts/set_mode.py captura      # dia normal, muita criança
.venv/bin/python scripts/set_mode.py realtime     # dia de pouco movimento, ou teste
```

Na troca a partir do modo captura, as trilhas que estavam em cena são salvas
antes da transição — ninguém que estava passando naquele instante se perde. E as
trilhas já capturadas continuam pendentes: o lote as processa depois, mesmo que
você tenha voltado para o modo em tempo real.

Para um teste pontual, sem mexer na configuração, dá para fixar o modo na
linha de comando (isso também desliga a troca automática):

```bash
sudo systemctl stop facial-worker
.venv/bin/python worker.py --mode realtime
```

Conferir o que está rodando, de qualquer máquina:

```bash
curl http://IP_DO_PI:8000/health
# "worker_mode": "captura", "tracks_pending": 34
```

O modo em `/health` é o que o worker está **realmente** executando, que pode
diferir do `config.yaml` se alguém tiver iniciado com `--mode`.

Ajustes finos do modo captura, no `config.yaml` (estes exigem reinício):

```yaml
tracking:
  crops_per_track: 3       # mais recortes = mais votos = mais robusto
batch:
  min_votos: 2             # quantos recortes precisam concordar
```

E ligue o lote uma vez:

```bash
sudo systemctl enable --now facial-batch.timer    # a cada 10 min
```

### Operar

```bash
.venv/bin/python scripts/recognize_batch.py              # reconhecer agora
.venv/bin/python scripts/recognize_batch.py --presenca    # presença de hoje
.venv/bin/python scripts/recognize_batch.py --presenca --dia 2026-08-03
.venv/bin/python scripts/recognize_batch.py --reprocessar # após cadastrar gente nova
```

O `--reprocessar` é importante: se alguém for cadastrado depois, as trilhas
antigas podem ser reavaliadas — nada foi perdido, os recortes continuam lá.

### Ajustar

| Sintoma | Ajuste |
|---|---|
| Pessoas conhecidas saindo como "Desconhecido" | `batch.min_votos` para 1; suba `crops_per_track` |
| Ainda há troca de identidade | `min_votos` para 3 (com `crops_per_track: 4`); suba o limiar |
| Trilhas demais, curtas e inúteis | suba `min_track_frames` |
| Uma pessoa virando várias trilhas | suba `max_missing_frames`; baixe `iou_threshold` |
| Duas pessoas viram uma trilha só | suba `iou_threshold` |

> **Presença não é o mesmo que ausência.** Quem não foi identificado pode ter
> sido falha de captura — não trate automaticamente como falta. O relatório
> lista essas pessoas separadamente, e vale conferir as trilhas marcadas como
> "Desconhecido" antes de fechar a chamada.

## 6.55 A chamada no painel, com conferência humana

No painel, aba **Chamada**. É onde a presença deixa de ser saída de máquina e
passa a ser registro conferido.

Escolha o dia, marque quem esteve presente e salve. A regra: **só as diferenças
em relação ao que o sistema detectou são gravadas como correção**. Se você
concorda com o reconhecimento, nada é escrito.

Cada pessoa aparece com a origem da informação:

| Marca | Significado |
|---|---|
| ✅ detectado | o reconhecimento identificou |
| ✏️ marcado presente | você corrigiu: veio, mas o sistema não pegou |
| ✏️ marcado ausente | você corrigiu: o sistema identificou por engano |
| ❔ não identificado | não detectado e não corrigido |

**Fechar a chamada** registra que uma pessoa conferiu. Depois disso ela aparece
como `conferida: true` na API, e as correções ficam travadas até você reabrir.
Só uma chamada conferida deveria alimentar falta em outro sistema.

Duas travas: não dá para fechar com trilhas aguardando reconhecimento (fecharia
incompleta), e não dá para corrigir uma chamada já fechada sem reabrir antes.

### As correções medem o sistema

Este é o efeito colateral mais útil da conferência. A correção nunca sobrescreve
o que o reconhecimento detectou — fica gravada ao lado. Então a diferença entre
os dois é a **taxa de erro medida**, por tipo:

- presente marcado à mão = o sistema **deixou passar** (falso negativo)
- ausente marcado à mão = o sistema **identificou errado** (falso positivo)

O painel mostra esses números ao final da página, e o `recognize_batch.py
--presenca` também. Acumulando alguns dias, eles dizem objetivamente se vale
mexer no limiar, no enquadramento ou no cadastro — sem depender de impressão.

É também o caminho mais barato para a medição de recall: conferir a chamada
todos os dias produz o dado, sem ninguém precisar contar crianças com prancheta.

## 6.6 Integração: endpoint de presença

Para outro sistema (acadêmico, planilha, script) buscar a chamada:

```
GET http://IP_DO_PI:8000/attendance
GET http://IP_DO_PI:8000/attendance?dia=2026-08-20
GET http://IP_DO_PI:8000/attendance?dia=2026-08-20&inicio=07:00&fim=08:00
```

Três campos merecem atenção de quem for consumir:

**`completo`** — vem `false` quando ainda há trilhas na fila de reconhecimento.
A chamada está **incompleta** nesse caso. Quem gravar falta sem checar isso vai
marcar falta de aluno que está apenas na fila. Este é o campo mais importante
da resposta.

**`nao_identificados`** — não é a mesma coisa que ausente. Pode ser falha de
captura, criança que passou fora do enquadramento, ou score abaixo do limiar.
Transformar isso em falta é decisão do outro sistema, e deveria passar por
conferência humana.

**`person_id`** — identificador interno deste sistema. Casar por **nome** é
frágil: homônimos, acentuação e digitação divergente quebram a associação. Para
integração de verdade, o próximo passo é guardar a matrícula do aluno aqui e
casar por ela.

A resposta une as duas origens possíveis — trilhas do modo captura e eventos do
modo realtime — e o campo `fontes` de cada pessoa diz de onde veio. Isso importa
porque cada modo grava em tabela diferente: ler só uma delas devolveria chamada
vazia dependendo do modo em uso naquele dia.

### Protegendo o acesso

A API nasceu sem autenticação. Como este endpoint devolve nomes de crianças com
horário, há agora um token opcional. No `config.yaml` do Pi:

```yaml
api:
  token: "cole-aqui-um-valor-forte"
```

Gere o valor com:

```bash
python -c "import secrets;print(secrets.token_urlsafe(32))"
```

Com o token preenchido, **todas** as rotas passam a exigir o cabeçalho
`X-API-Token` (ou `Authorization: Bearer`), exceto `/health`, que fica aberta
para monitoramento:

```bash
curl -H "X-API-Token: SEU_TOKEN" http://IP_DO_PI:8000/attendance
```

> ⚠️ Dois avisos. O `config.yaml` **do PC** precisa do mesmo token, senão o
> painel passa a receber 401. E token sobre HTTP simples trafega em texto claro
> na rede — em rede de escola compartilhada, isso protege contra acesso casual,
> não contra quem esteja capturando tráfego. HTTPS resolveria, e fica na lista
> de pendências.

## 7. Calibrar o limiar de reconhecimento

`recognition.cosine_threshold: 0.363` é o valor de referência do SFace, não uma
verdade universal. Ele decide entre "é a Maria" e "é um desconhecido".

- Score **alto** = rostos parecidos. `>= limiar` → mesma pessoa.
- Subir o limiar (ex.: 0.45): menos falsos positivos, mais gente conhecida
  virando "Desconhecido".
- Descer (ex.: 0.30): reconhece mais, com risco de confundir pessoas parecidas.

Calibre com os seus dados, não por palpite. Deixe rodar algumas horas e use:

```bash
.venv/bin/python scripts/calibrate_threshold.py            # distribuição dos scores
.venv/bin/python scripts/calibrate_threshold.py --review    # marca acerto/erro e sugere
.venv/bin/python scripts/calibrate_threshold.py --simular 0.55   # efeito antes de aplicar
```

No `--review` ele mostra cada reconhecimento com o link da foto e pergunta se
acertou. Com isso separa a distribuição dos acertos da dos erros e propõe um
limiar entre as duas. As respostas ficam salvas, então dá para revisar aos poucos.

Se as distribuições **se sobrepõem**, ele avisa em vez de inventar um número —
e com razão: nesse caso nenhum limiar separa os dois casos, e mexer nele só
troca falso positivo por falso negativo. O que resolve aí é cadastro e
enquadramento (veja abaixo).

### Falso positivo: identifica outra pessoa como alguém cadastrado

Em ordem de eficácia:

1. **Cadastre mais amostras** da pessoa, variando ângulo, distância e
   iluminação. Mais amostras aumentam o score dela nos acertos, o que permite
   subir o limiar sem perdê-la.
2. **Cadastre também as outras pessoas** que passam. Com só uma pessoa na
   galeria, todo rosto é comparado apenas com ela — não há identidade
   concorrente para "vencer" a comparação. É o cenário mais propício a erro.
3. **Suba o limiar**, guiado pelo `calibrate_threshold.py`.
4. **Aumente `recognition.min_face_size`** (de 50 para 70–80). Rosto pequeno
   gera embedding ruim e é fonte clássica de confusão.
5. **Melhore o enquadramento**: rosto de frente, sem contraluz, na altura dos
   olhos.

---

## 8. Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `facial-worker` não sobe | Modelos ausentes, config inválida, OpenCV errado | `sudo journalctl -u facial-worker -n 50 --no-pager` |
| "nenhum frame em 20s" | URL, senha não codificada, RTSP desabilitado, rede | `ffprobe -rtsp_transport tcp "<url>"` |
| `401 Unauthorized` no ffprobe | Senha errada ou com `@`/`#` sem codificar | Seção 3.3 |
| Conecta e cai toda hora | Wi‑Fi instável, fonte fraca, GOP muito longo | Cabo de rede; `vcgencmd get_throttled` |
| Poucos fps chegando | Substream em H.265, ou resolução alta | Câmera → Stream Extra → H.264, 640x480, 10 fps |
| Tudo "Desconhecido" | Ninguém cadastrado, ou limiar alto demais | Cadastre pelo painel; seção 7 |
| Reconhece a pessoa errada | Limiar baixo, poucas amostras, rosto pequeno | Suba o limiar; recadastre com mais amostras |
| Perde quem passa rápido | Pi processa ~2 frames/s | Diminua `min_interval_seconds`, use `--int8`, aproxime a câmera |
| Painel: "não consegui falar com a API" | `base_url` errado ou firewall | `curl http://IP_DO_PI:8000/health` do PC |
| `/live.jpg` dá 404 | Worker parado ou `draw_annotations: false` | `systemctl status facial-worker` |
| Pi travando / OOM | Desktop gráfico ligado | `sudo raspi-config` → Boot → Console |
| Cartão cheio | Snapshots acumulados | `.venv/bin/python scripts/cleanup_snapshots.py --days 7` |
| `bash: $'\r': command not found` | Arquivo salvo com CRLF (Windows) | `sed -i 's/\r$//' install_pi.sh` |

Comandos do dia a dia:

```bash
sudo journalctl -u facial-worker -f        # log ao vivo do reconhecimento
sudo journalctl -u facial-api -f           # log da API
sudo systemctl restart facial-worker       # aplicar mudança no config.yaml
sudo systemctl status facial-worker facial-api
.venv/bin/python scripts/check_pi.py       # diagnóstico completo
```

O worker imprime um resumo de saúde a cada 5 minutos no journal:

```
[stats] 9.8 fps da câmera | 21 processados | 380 ms/frame | {'frames_received': 2940, 'reconnects': 0, ...}
```

`reconnects` subindo é sinal de rede ou energia instáveis.

---

## 9. Manutenção e cartão SD

O Pi grava em cartão SD, que tem número finito de escritas. O preview ao vivo
vai para `/dev/shm` (memória RAM), não para o cartão — são ~172 mil escritas por
dia evitadas.

### Os três acervos de imagem, e por que têm prazos diferentes

| Onde | O que é | Prazo |
|---|---|---|
| `data/snapshots/AAAAMMDD/` | foto de cada passagem registrada | `--days` (padrão 30) |
| `data/tracks/AAAAMMDD/` | recortes que o lote usa para reconhecer | `--dias-trilhas` (padrão 7) |
| `data/snapshots/amostras/<id>/` | fotos do cadastro de cada pessoa | sem prazo — saem quando a pessoa é excluída |

Os recortes de trilha têm prazo curto porque são **evidência transitória**:
depois que o lote processou a trilha, servem só para auditoria. Já as fotos do
cadastro precisam durar enquanto a pessoa estiver cadastrada, senão o
reconhecimento perderia a referência visual para revisão.

No banco, `events` e `tracks` seguem o prazo longo (`--days`), porque ali mora o
histórico de presença.

O `facial-cleanup.timer` roda todo dia às 03:30. Para ajustar:

```bash
sudo systemctl edit --full facial-cleanup.service   # mude --days e --dias-trilhas
sudo systemctl restart facial-cleanup.timer
systemctl list-timers facial-cleanup.timer          # confirmar o próximo disparo
```

Conferir antes de aplicar:

```bash
.venv/bin/python scripts/cleanup_snapshots.py --dry-run
```

> **Uma trava importante:** se houver trilhas **pendentes** mais antigas que o
> prazo, a limpeza **não** apaga os recortes e avisa. Trilha pendente antiga
> significa que o reconhecimento em lote parou de rodar — apagar os recortes
> nesse caso jogaria fora dado que nunca foi aproveitado. O aviso aponta para
> `systemctl status facial-batch.timer`.

### Excluir uma pessoa

Pelo painel, em **Pessoas**, com duas opções:

- **Anonimizar** — remove a pessoa, os embeddings e **todas as fotos**, mas
  mantém as linhas de passagem com identificação removida. A estatística de
  "alguém passou às 7:42" sobrevive; quem passou, não.
- **Apagar tudo** — remove também os registros de passagem. Sem desfazer.

Nos dois casos as imagens saem do disco, nas duas pastas — snapshots e recortes
de trilha. Para dado biométrico de criança, "excluir" precisa excluir de fato.

Backup do que importa (cadastros e histórico):

```bash
sqlite3 data/data.db ".backup /tmp/backup.db" && scp pi@IP:/tmp/backup.db .
```

---

## 10. LGPD — não é opcional

Imagem de rosto é **dado pessoal sensível** (biométrico) pela Lei 13.709/2018.
Antes de colocar em produção na organização:

- Defina e registre a **base legal** do tratamento e, quando aplicável, colha
  consentimento — para funcionários, avalie legítimo interesse com cuidado, já
  que relação de subordinação enfraquece o consentimento.
- **Sinalize o ambiente** de forma visível: "Local monitorado por reconhecimento
  facial".
- Defina **prazo de retenção** (é o `--days` da limpeza) e documente-o.
- Restrinja quem acessa o painel: hoje a API **não tem autenticação**. Mantenha
  o Pi em rede interna, nunca exposto à internet, e considere adicionar
  autenticação antes de qualquer uso real.
- Tenha um caminho para **atender pedidos de exclusão** (remover a pessoa pelo
  painel apaga os embeddings dela).

Nada aqui é orientação jurídica — vale validar com quem responde por proteção
de dados na organização antes de ligar isso em produção.
