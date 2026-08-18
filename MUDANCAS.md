# O que mudou em relação ao projeto original

Registro das alterações feitas para o projeto rodar em **Raspberry Pi 3B** com
câmera IP Intelbras. Base: commit `a8e99c9` (versão original do Kevenny).

Resumo: 22 arquivos, ~2.400 linhas adicionadas. Nada da **lógica de
reconhecimento** foi alterado — YuNet + SFace + comparação por cosseno seguem
exatamente como estavam.

---

## 1. Um bloqueador que impedia o projeto de rodar no Pi

### `requirements-pi.txt` — troca do OpenCV

O README original orientava usar o OpenCV do sistema no Pi:

```bash
sudo apt install python3-opencv    # "Bookworm = OpenCV 4.6, já tem YuNet/SFace"
```

**O problema:** o detector `face_detection_yunet_2023mar.onnx` usa formas de
entrada dinâmicas e **só carrega no OpenCV ≥ 4.8**. No 4.6 do apt ele falha com
um erro interno de DNN, difícil de associar à causa. O worker subiria e morreria.

**A correção:** fixar `opencv-contrib-python-headless>=4.9,<5`. Em Raspberry Pi
OS 64-bit (`aarch64`) existe wheel pronto no PyPI, então nada é compilado — foi
justamente o medo de compilar que motivou a escolha original pelo apt. A variante
`headless` evita as dependências de interface gráfica, que o Pi não usa.

### `core/face_engine.py` — diagnóstico em vez de erro obscuro

Adicionado `_resolve_model()`, que antes de carregar:

- verifica a versão do OpenCV e, se for antiga, explica a exigência e o comando
  para corrigir, em vez de deixar o OpenCV estourar;
- procura o modelo `2022mar` como alternativa automática (é o que funciona em
  OpenCV < 4.8), usando-o se estiver presente;
- se nenhum modelo existe, diz qual arquivo falta e qual comando baixa.

**Motivo:** transformar três falhas silenciosas em mensagens acionáveis.

---

## 2. Eficiência — o Pi 3B tem 4 núcleos a 1,2 GHz e 1 GB de RAM

### `core/camera.py` — `read_new()` no lugar de `read()`

O `read()` original devolve **uma cópia** do último frame. O worker o chamava em
loop com `sleep(0.005)`, ou seja ~200 vezes por segundo, enquanto a câmera
entrega ~16 frames por segundo. Resultado: mais de 180 cópias por segundo do
**mesmo** frame. A 640x480x3, cada cópia é 921 KB — cerca de 170 MB/s de memcpy
desperdiçado.

`read_new(last_seq)` usa um contador de sequência e só devolve algo quando existe
frame novo. Por padrão entrega a referência sem copiar — seguro porque a thread
de captura cria um array novo a cada leitura, então o frame já entregue continua
íntegro.

Em teste: 31 frames novos contra 1.332 chamadas sem novidade, no mesmo intervalo
em que a versão antiga teria feito 1.363 cópias.

O `read()` foi mantido, porque a API de cadastro precisa da cópia.

### `worker.py` — `process_every_n_frames` agora faz o que promete

O contador incrementava a cada **iteração do loop**, não a cada frame recebido.
Como o loop girava muito mais rápido que o stream, `process_every_n_frames: 5`
não significava "1 a cada 5 frames da câmera" — descartava iterações que muitas
vezes olhavam o mesmo frame. Agora o contador avança junto com a sequência da
câmera.

### `worker.py` — novo `min_interval_seconds`

Teto explícito de processamentos por segundo. É o freio mais previsível no Pi,
porque não depende do fps que a câmera está entregando naquele momento. No preset
do Pi vem `0.5` (máximo 2 por segundo).

### `worker.py` + config — `opencv_threads`

No Pi 3B, deixar 3 threads para a DNN e 1 núcleo livre para o decode do RTSP dá
latência menor do que usar os 4 na rede neural. Configurável; `0` mantém o
comportamento automático anterior.

### `config.pi.example.yaml` — `top_k`

Cheguei a reduzir de 50 para 20 supondo fluxo esparso. **Revertido para 50**: o
`top_k` é o número máximo de rostos que o detector devolve por frame, e o
excedente é descartado silenciosamente. Em ambiente com grupos — corredor de
escola, por exemplo — isso apagaria pessoas do registro. O custo de manter folga
é baixo, porque o gargalo é o embedding (285 ms por rosto), não a detecção
(57 ms por frame, independente da quantidade).

---

## 3. Proteção do cartão SD

### `storage.live_path` — preview ao vivo em tmpfs

O preview era escrito em `data/live.jpg`, no cartão, a cada 0,5 s — cerca de
**172 mil escritas por dia**. Cartão SD tem número finito de escritas. Agora o
caminho é configurável e o preset do Pi aponta para `/dev/shm/facial-live.jpg`
(memória RAM). A API lê do mesmo lugar, via `live_image_path()` em
`core/config.py`.

### `scripts/cleanup_snapshots.py` + `systemd/facial-cleanup.{service,timer}`

Não havia retenção. Um corredor movimentado gera centenas de MB por dia; quando
o cartão enche, o serviço morre sem mensagem clara. O timer roda todo dia às
03:30, apaga snapshots além do prazo (padrão 30 dias), respeita um teto de
tamanho e remove os registros correspondentes no banco.

Além do aspecto técnico, prazo de retenção definido é exigência prática de LGPD
para dado biométrico.

### `core/database.py` — `purge_events_before()` e `count_events()`

Métodos novos; o esquema do banco não mudou. O `purge` roda o `VACUUM` em conexão
separada e em autocommit, porque o SQLite recusa `VACUUM` dentro de uma transação
— e o `with con` abre uma implicitamente.

---

## 4. Bugs corrigidos

### `worker.py` — preview escrito de forma atômica

O worker escrevia `live.jpg` diretamente enquanto a API podia estar lendo o
mesmo arquivo. Resultado: JPEG cortado servido ao painel de vez em quando.
Agora codifica em memória (`imencode`), escreve em arquivo temporário e faz
`os.replace()`, que é atômico.

> Detalhe pego em teste: a primeira versão desta correção usava um temporário
> com extensão `.tmp`, e o OpenCV escolhe o formato **pela extensão** — o
> `imwrite` falhava. Daí o `imencode` com escrita manual.

### `worker.py` — um frame ruim não derruba mais o serviço

O laço de processamento não tinha tratamento de exceção. Qualquer erro
transitório (frame corrompido, falha momentânea de decode) encerrava o processo.
Agora o frame problemático é registrado e ignorado.

### `systemd/*.service` — três problemas

1. `User=pi` e `WorkingDirectory=/home/pi/reconhecimento_facial_ia`: o Raspberry
   Pi OS Bookworm **não cria mais o usuário `pi`** por padrão, e o caminho não
   corresponde ao nome da pasta. As units agora usam placeholders que o
   `install_pi.sh` substitui pelo usuário e caminho reais.
2. Faltava `StartLimitIntervalSec=0`. Sem isso o systemd desiste depois de
   algumas falhas seguidas — por exemplo, câmera indisponível durante uma queda
   de energia — e o serviço fica morto até alguém intervir manualmente.
   (Vai em `[Unit]`, não em `[Service]`; o `systemd-analyze verify` apontou isso.)
3. `PYTHONUNBUFFERED=1` e `SyslogIdentifier`, para os `print` aparecerem no
   journal na hora e o log ser filtrável.

### `models/download_models.py` — validação do que foi baixado

Os modelos ficam no repositório do OpenCV Zoo via git-lfs. Um download que
devolve um **ponteiro LFS** ou uma **página de erro HTML** era salvo como
`.onnx` e só se manifestava muito depois, como erro incompreensível do OpenCV.

Agora: dois espelhos alternativos, verificação de tamanho mínimo, detecção de
ponteiro LFS e de HTML/JSON disfarçado, escrita via arquivo temporário, e
mensagem dizendo como copiar manualmente se a rede do Pi bloquear o GitHub.
Ganhou também `--int8` (modelos quantizados, ~2x mais rápidos) e `--force`.

---

## 5. Deploy de Windows para Linux

### `.gitattributes` (novo) e normalização no instalador

Os arquivos do repositório estavam com terminação de linha do Windows (CRLF).
Isso quebra de duas formas no Pi:

- script `.sh` com CRLF: `bash: $'\r': command not found`;
- unit systemd com CRLF: o `\r` entra no argumento do `ExecStart` e o serviço
  não sobe.

O `.gitattributes` força LF em todo o projeto, e o `install_pi.sh` ainda passa
`tr -d '\r'` ao gerar as units, como segunda linha de defesa.

---

## 6. Capacidades novas

| Arquivo | O que faz | Por que existe |
|---|---|---|
| `install_pi.sh` | Instalação em 9 etapas verificadas: hardware, pacotes, venv, OpenCV validado, modelos, config, teste de câmera, systemd, resumo | O passo a passo manual do README tinha 8 comandos e nenhuma verificação; qualquer desvio falhava silenciosamente |
| `scripts/check_pi.py` | Diagnóstico e **benchmark**: versão do OpenCV, modelos, RAM, disco, subtensão, temperatura, resolução real da câmera, ms de detecção e de embedding, e se o config pede mais do que o Pi entrega | Sem medir, o ajuste dos parâmetros é chute |
| `scripts/find_camera.py` | Varre a rede, classifica equipamentos e sonda RTSP **falando o protocolo direto no socket** (Digest e Basic), distinguindo 401 (credencial) de 404 (caminho) de 200 (ok) | Descobrir se a câmera serve, e por que não conecta quando não conecta |
| `scripts/calibrate_threshold.py` | Analisa os scores reais, permite marcar acertos e erros, e calcula o limiar — avisando quando as distribuições se sobrepõem, caso em que limiar nenhum resolve | O `cosine_threshold` é o parâmetro que mais afeta o resultado e o mais difícil de acertar no olho |
| `scripts/cleanup_snapshots.py` | Retenção de snapshots e eventos | Cartão SD e LGPD |
| `config.pi.example.yaml` | Preset com todos os parâmetros já ajustados para Pi 3B | Evita ter que descobrir cada valor |
| `GUIA_RASPBERRY_PI.md` | Guia completo: pré-requisitos, câmera Intelbras, instalação, desempenho, calibração, troubleshooting, LGPD | — |

### `core/camera.py` — webcam USB e câmera CSI

`_parse_source()` interpreta a fonte e escolhe o backend: `rtsp://` e arquivos
vão para FFmpeg; `0`, `"1"` ou `/dev/video0` vão para V4L2. Para dispositivo
local, aplica MJPEG antes da resolução (em USB 2.0 o formato cru satura o
barramento e custa CPU).

`camera_from_config()` centraliza a criação da câmera, usada por worker, API e
scripts.

**Motivo:** era o plano B caso a câmera não expusesse RTSP — situação real com
câmeras de nuvem. No fim não foi necessário, mas continua disponível e, no Pi 3B,
é até mais leve: elimina o decode de H.264.

### `api.py` — `/health` útil

Antes retornava `{"ok": true}`. Agora inclui versão do OpenCV, número de pessoas
cadastradas, se há sessão de cadastro ativa e a **idade do último preview** —
que é como se descobre remotamente, pelo navegador, que o worker parou.

---

## 6.5 Pipeline de duas fases (modo `captura`)

Adicionado para o cenário de **grupos** — corredor de escola, chamada de
presença — onde o modo original não funciona. O reconhecimento custa ~285 ms
**por rosto**: com 10 pessoas no frame, um ciclo leva ~2,9 s, e como os frames
que chegam durante o processamento são descartados, o grupo inteiro é julgado
por um único instantâneo.

A detecção, ao contrário, custa ~57 ms por frame **independente da quantidade**
de rostos. O modo `captura` explora isso:

- `core/tracker.py` (novo): associa detecções por IoU ao longo dos frames,
  pontua cada recorte por nitidez × tamanho e guarda os melhores de cada pessoa.
  A nitidez pesa mais porque borrão de movimento degrada o embedding mais do que
  rosto pequeno.
- `worker.py`: ganhou `worker.mode`. Em `captura` ele detecta, rastreia e grava
  recortes — sem reconhecer. O modo `realtime` segue idêntico e é o padrão.
- `scripts/recognize_batch.py` (novo): fase 2. Reconhece cada recorte da trilha
  e decide por **votação**, com score final igual à média dos votos vencedores.
  Também gera o relatório de presença.
- `core/database.py`: tabelas `tracks` e `track_crops`, e a consulta de presença
  (agrupa por pessoa — para chamada importa "foi vista", não quantas vezes).
- `systemd/facial-batch.{service,timer}`: lote a cada 10 min, com `Nice=15` e
  `IOSchedulingClass=idle` para não competir com a captura.

Detalhe que exigiu cuidado: o `alignCrop` do SFace usa os 5 pontos do rosto.
Ao salvar um recorte, esses pontos precisam ser transladados para o sistema de
coordenadas do recorte, senão o alinhamento da fase 2 sairia diferente do da
fase 1 e o limiar calibrado não valeria — falha silenciosa. O teste
`tracker_alinhamento_do_recorte_equivale_ao_frame_inteiro` verifica que o
resultado é idêntico ao pixel, inclusive para rostos nas bordas, e que a margem
de 0,4 usada no recorte é necessária (com 0,05 o alinhamento degrada).

### Troca de modo em execução

O modo é alternável de três formas, em ordem de precedência:

1. `worker.py --mode realtime|captura` — fixa e desliga a troca automática.
2. `scripts/set_mode.py captura` — edita o `config.yaml` preservando os
   comentários; o worker aplica em até 10 s **sem reiniciar**.
3. Editar `worker.mode` no `config.yaml` à mão.

Modelo (37 MB) e câmera são carregados uma vez e reaproveitados entre as trocas,
então alternar não custa reinício. Só o `mode` é relido a quente — os demais
parâmetros continuam exigindo reinício, porque aplicar metade deles no meio da
execução deixaria o processo em estado inconsistente.

Na saída do modo captura, o `finally` salva as trilhas em cena, então a troca
não perde quem estava passando. Valor inválido ou YAML corrompido no momento da
releitura mantêm o modo atual, em vez de derrubar o serviço.

O worker publica `facial-status.json` ao lado do preview (em tmpfs), com o modo
real em execução e o número de trilhas pendentes. A API expõe isso em `/health`
como `worker_mode` e `tracks_pending` — é como se descobre remotamente que
alguém subiu o worker com `--mode` e ele está ignorando o `config.yaml`.

### Encerramento limpo (bug corrigido)

O `finally` que salva as trilhas em cena não rodava com `systemctl stop/restart`,
porque o SIGTERM padrão mata o processo na hora — as trilhas em andamento eram
perdidas a cada reinício. Agora há handler de SIGTERM/SIGINT, e a unit ganhou
`TimeoutStopSec=25`. Coberto pelo teste `worker_sigterm_salva_trilhas_em_andamento`.

## 6.6 Suíte de testes

`tests/test_all.py` (novo): 14 testes que rodam **sem câmera e sem os modelos
ONNX**, então funcionam no Pi e em qualquer máquina. Sem pytest, para não exigir
instalação extra.

Cobrem principalmente o que falharia em silêncio: equivalência do alinhamento,
contagem de frames, votação, escrita atômica do preview, retenção e o
encerramento por SIGTERM.

```bash
python tests/test_all.py            # tudo
python tests/test_all.py tracker    # filtra por nome
```

## 7. O que **não** foi alterado

- **A lógica de reconhecimento**: YuNet, SFace, `alignCrop`, normalização L2,
  comparação por cosseno, escolha do melhor rosto. Intacta.
- **`core/draw.py`, `core/storage.py`, `panel/app.py`**: nenhuma alteração
  funcional (aparecem no diff apenas por causa da normalização de fim de linha).
- **Esquema do banco**: as três tabelas e os índices continuam iguais; só
  ganharam métodos novos.
- **Contratos da API**: todas as rotas e formatos de resposta preservados.
- **Fase 1 no Mac**: continua funcionando. Todos os parâmetros novos têm valor
  padrão que reproduz o comportamento anterior — `min_interval_seconds: 0`,
  `opencv_threads: 0`, `live_path: data/live.jpg`, e a câmera sem `width`/`height`
  se comporta como antes.

---

## 8. Pendências conhecidas

Levantadas ao revisar o fluxo de dados, ainda **não** implementadas:

1. **Amostras de cadastro nunca são apagadas.** Ficam em
   `data/snapshots/enroll/<sessão>/`; a limpeza diária só varre pastas com nome
   de data, então essa escapa.
2. **Apagar uma pessoa não apaga o histórico dela.** `delete_person()` remove
   nome e embeddings, mas os eventos e as fotos continuam no disco — insuficiente
   para atender um pedido de exclusão.
3. **A API não tem autenticação.** Qualquer um na rede acessa `/events`,
   `/snapshots`, `/people` e `/live.jpg`.
4. **Os nomes também vão para o journal do systemd**, numa segunda cópia com
   retenção própria, fora do controle do `cleanup_snapshots.py`.
