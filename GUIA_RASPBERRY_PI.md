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

Meça no **seu** hardware e enquadramento:

```bash
cd ~/rc-realtime-processor
.venv/bin/python scripts/check_pi.py
```

Ele mede fps da câmera, ms de detecção, ms de embedding, e avisa se o
`config.yaml` está pedindo mais do que o Pi entrega.

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

## 7. Calibrar o limiar de reconhecimento

`recognition.cosine_threshold: 0.363` é o valor de referência do SFace, não uma
verdade universal. Ele decide entre "é a Maria" e "é um desconhecido".

- Score **alto** = rostos parecidos. `>= limiar` → mesma pessoa.
- Subir o limiar (ex.: 0.45): menos falsos positivos, mais gente conhecida
  virando "Desconhecido".
- Descer (ex.: 0.30): reconhece mais, com risco de confundir pessoas parecidas.

Como calibrar com dados reais, não por palpite: cadastre as pessoas, deixe rodar
algumas horas e olhe os scores dos eventos.

```bash
# scores dos acertos (pessoas conhecidas)
sqlite3 data/data.db "SELECT name, ROUND(score,3) FROM events WHERE is_known=1 ORDER BY score LIMIT 20;"
# scores dos que caíram como desconhecidos
sqlite3 data/data.db "SELECT ROUND(score,3) FROM events WHERE is_known=0 ORDER BY score DESC LIMIT 20;"
```

Escolha um limiar **entre** o menor score dos acertos e o maior score dos erros.
Se esses dois números se sobrepõem, o problema não é o limiar: é enquadramento,
iluminação ou poucas amostras no cadastro.

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

O Pi grava em cartão SD, que tem número finito de escritas. O que este projeto
já faz para poupá-lo:

- O preview ao vivo vai para `/dev/shm` (memória RAM), não para o cartão —
  configurado em `storage.live_path`. São ~172 mil escritas por dia evitadas.
- `facial-cleanup.timer` roda todo dia às 03:30 e apaga snapshots antigos
  (padrão: 30 dias / teto de 2 GB), removendo também os registros
  correspondentes no banco.

Ajustar a retenção:

```bash
sudo systemctl edit --full facial-cleanup.service   # mude --days e --max-mb
sudo systemctl restart facial-cleanup.timer
systemctl list-timers facial-cleanup.timer          # confirmar o próximo disparo
```

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
