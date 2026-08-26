# Arquitetura do `core/` — algoritmos, estruturas e decisões

Documento técnico do núcleo do projeto: 1.189 linhas em 7 módulos. Cada seção
explica **o que** o módulo faz, **como** (algoritmo e estrutura de dados) e
**por que** dessa forma.

Onde a escolha foi do Kevenny na versão original, está indicado. O resto foi
decidido na adaptação para o Raspberry Pi 3B.

---

## 1. O fluxo de dados

```
                    ┌──────────── core/camera.py ────────────┐
câmera ──RTSP/V4L2──▶ thread de captura → 1 slot (último frame)
                    └───────────────────┬───────────────────┘
                                        │ read_new(seq)
                                        ▼
                              core/face_engine.py
                          detect() → lista de rostos
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        modo realtime                              modo captura
   embed() + match() na hora                  core/tracker.py
                    │                      agrupa em trilhas,
                    │                      guarda melhores recortes
                    ▼                                       ▼
            core/database.py ◀──────── fila (tabela tracks) ─┘
            events / tracks                                  │
                    │                          recognize_batch.py
                    │                          embed() + votação
                    ▼                                       │
            core/storage.py ◀──────────────────────────────┘
            imagens em disco
```

A separação em dois modos existe porque o custo do reconhecimento é **por
rosto** (~285 ms no Pi 3B) enquanto o da detecção é **por frame** (~57 ms,
independente de quantos rostos haja). Com muitas pessoas simultâneas, só a
segunda opção acompanha o vídeo.

---

## 2. A representação numérica: o que é um embedding

Todo o reconhecimento se reduz a geometria em 128 dimensões.

O SFace transforma um recorte de rosto alinhado (112×112 pixels) em um vetor de
**128 números float32**. A propriedade útil: rostos da mesma pessoa produzem
vetores próximos; de pessoas diferentes, distantes. "Próximo" aqui é medido por
**similaridade de cosseno**:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

`core/face_engine.py::embed()` normaliza cada vetor para norma 1 (`vec / ‖vec‖`).
Isso não é detalhe de estilo — é o que permite a simplificação central do
projeto: **com vetores normalizados, `‖a‖ = ‖b‖ = 1`, então o cosseno vira
simplesmente o produto interno `a · b`**. Uma divisão por norma no cadastro
elimina duas raízes quadradas em cada comparação futura.

Custo de armazenamento: 128 × 4 bytes = **512 bytes por amostra**. Um cadastro
de 500 alunos com 5 amostras cada ocupa 1,28 MB — cabe folgado na RAM de 1 GB do
Pi.

---

## 3. `core/camera.py` — o padrão "último valor"

**Problema.** O RTSP entrega frames continuamente, mas o reconhecimento leva
centenas de milissegundos. Se os frames fossem enfileirados, a fila cresceria
sem limite e o sistema passaria a reconhecer imagens de minutos atrás.

**Estrutura de dados: um slot, não uma fila.**

```python
self._frame = None    # UM frame, sobrescrito
self._seq = 0         # contador monotônico
self._lock = threading.Lock()
```

Uma thread dedicada lê da câmera em laço infinito e **sobrescreve** o slot. O
frame anterior é descartado. Perder frames aqui é a intenção, não um defeito:
para reconhecer alguém que está passando, o frame mais recente é sempre o mais
útil.

Esse padrão é conhecido como *latest-value* ou *conflated queue*. A alternativa
(fila com descarte no fim) daria o mesmo resultado com mais código.

**O contador de sequência.** O consumidor precisa saber se o frame mudou. Uma
variável de condição resolveria, mas exigiria o consumidor esperar — e ele tem
outras coisas a fazer. Optei por um inteiro monotônico:

```python
def read_new(self, last_seq, copy=False):
    with self._lock:
        if self._frame is None or self._seq <= last_seq:
            return None, last_seq      # nada novo
        frame, seq = self._frame, self._seq
    return (frame.copy() if copy else frame), seq
```

O consumidor guarda o último `seq` que viu e passa de volta. Comparação de
inteiros, custo desprezível. O laço do worker fica:

```python
frame, seq = cam.read_new(seq)
if frame is None:
    time.sleep(0.02)      # nada novo: dorme 20 ms
    continue
```

**Por que 20 ms de sono.** É o intervalo entre verificações. Com processamento
de ~340 ms por frame, 20 ms de latência adicional é 6% — irrelevante. Já um
laço sem sono consumiria um núcleo inteiro só perguntando "mudou?".

**A entrega sem cópia, e o invariante que a sustenta.** `read_new` devolve a
referência do array, não uma cópia. Isso é seguro por duas razões que precisam
valer juntas:

1. `cap.read()` do OpenCV **aloca um array novo** a cada chamada;
2. a contagem de referências do Python mantém vivo o array já entregue, mesmo
   depois de `self._frame` apontar para outro.

Se alguém "otimizar" a captura para reusar um buffer pré-alocado, esse invariante
quebra e o consumidor passa a ver a imagem mudando debaixo dele — um bug de
corrida difícil de rastrear. Está comentado no código por isso.

A versão original chamava `frame.copy()` sempre. Com 640×480×3, cada cópia é
921 KB; o laço chamava ~200 vezes por segundo contra ~16 frames recebidos, então
mais de 180 cópias por segundo eram do **mesmo** frame — cerca de 170 MB/s de
memória movida à toa.

**Seleção de backend.** `_parse_source()` decide o backend pelo tipo da fonte:

| Fonte | Backend | Motivo |
|---|---|---|
| `rtsp://...`, caminho de arquivo | `CAP_FFMPEG` | rede e contêineres de vídeo |
| `0`, `"1"`, `/dev/video0` | `CAP_V4L2` | dispositivo local |

Para dispositivo local, aplica MJPEG **antes** da resolução: no USB 2.0 o vídeo
cru (YUYV) satura o barramento e limita a taxa.

**Reconexão.** O laço trata falha de leitura como fim da conexão: libera o
`VideoCapture`, dorme `reconnect_delay` e reabre. Sem estado a recuperar —
idempotente por construção. Contadores de `frames_received` e `reconnects` ficam
expostos para diagnóstico.

---

## 4. `core/face_engine.py` — detecção, embedding e busca

Dois modelos ONNX do OpenCV Zoo, ambos rodando no módulo DNN do OpenCV em CPU.
Escolha do Kevenny, e boa: dispensa dlib, onnxruntime e GPU.

### 4.1 O formato de saída do YuNet

Cada rosto detectado é uma linha de **15 números float32**:

| Índices | Conteúdo |
|---|---|
| 0–3 | `x, y, largura, altura` da caixa |
| 4–13 | 5 pontos `(x, y)`: olho direito, olho esquerdo, nariz, canto direito da boca, canto esquerdo |
| 14 | confiança do detector |

Esse layout aparece em vários lugares do projeto (recorte, alinhamento,
serialização das trilhas), então está documentado como constante em
`core/tracker.py`.

### 4.2 Detectar em imagem reduzida e reescalar

```python
if self.detect_width and w > self.detect_width:
    scale = self.detect_width / float(w)
    img = cv2.resize(image, (int(w * scale), int(h * scale)))
...
face[:14] = face[:14] / scale     # volta à escala original
```

O custo da detecção cresce com a área da imagem. Reduzir de 640 para 320 px de
largura corta a área em 4× e o tempo proporcionalmente. As coordenadas voltam
divididas pela escala.

Dois detalhes que importam:

- **Só os índices 0 a 13 são divididos.** O índice 14 é confiança, não
  coordenada — dividi-lo corromperia o score.
- **O filtro `min_face_size` é aplicado depois do reescalonamento**, então o
  valor de configuração está em pixels do frame original, que é o que a pessoa
  consegue raciocinar sobre.

### 4.3 A galeria como matriz, e a consequência disso

`core/database.py::load_gallery()` monta três estruturas:

```python
matrix : np.ndarray  (N, 128)   # UMA LINHA POR AMOSTRA, não por pessoa
ids    : list[int]              # person_id de cada linha (repete)
names  : dict[int, str]         # person_id -> nome
```

E a busca é uma única multiplicação matriz-vetor:

```python
sims = gallery_matrix @ vec      # (N,128) @ (128,) -> (N,)
idx  = int(np.argmax(sims))
return gallery_ids[idx], float(sims[idx])
```

Isso substitui um laço Python de N iterações por uma operação BLAS. Com 2.500
amostras são 320 mil multiplicações-acumulações: microssegundos. **A busca
nunca é o gargalo** — o embedding é.

**A consequência importante, que não é óbvia.** Como há uma linha por amostra e
a decisão é o `argmax`, o resultado é o **máximo** sobre todas as amostras de
todas as pessoas. Acrescentar uma amostra só pode **aumentar** esse máximo,
nunca diminuir.

Ou seja: cada amostra nova aumenta monotonicamente a chance de qualquer rosto
cruzar o limiar — tanto o da pessoa certa quanto o de um estranho. É a
justificativa formal para o aviso de redundância no painel: uma amostra quase
idêntica a outra não acrescenta capacidade de reconhecer, mas acrescenta uma
chance extra de confundir. Amostras devem ser **variadas**, não numerosas.

Uma alternativa de projeto seria fazer *média* dos vetores por pessoa (um
centroide, N = número de pessoas). Ficaria mais robusto a falso positivo e mais
rápido, mas perderia a capacidade de reconhecer ângulos distintos, porque a média
de dois pontos distantes na esfera não corresponde a nenhum rosto real. O
desenho atual (max-pooling) é o mesmo do artigo original do SFace.

### 4.4 Alinhamento antes do embedding

```python
aligned = self.recognizer.alignCrop(image, face)
feat = self.recognizer.feature(aligned)
```

`alignCrop` calcula uma transformação de similaridade (rotação, escala e
translação, sem distorção) que leva os 5 pontos detectados a posições canônicas
fixas num quadro 112×112. É o que torna o embedding invariante à rotação da
cabeça no plano e ao tamanho do rosto na imagem.

Esse passo é a razão pela qual `core/tracker.py` precisa **traduzir os
landmarks** ao salvar um recorte — assunto da próxima seção.

### 4.5 Resolução de modelo com verificação de versão

`_resolve_model()` existe porque o YuNet `2023mar` usa formas de entrada
dinâmicas e só carrega no OpenCV ≥ 4.8. Em versão anterior ele falha com um erro
interno de DNN que não indica a causa. A função verifica a versão, procura o
modelo `2022mar` como alternativa e, se nada servir, levanta erro dizendo qual
comando resolve.

---

## 5. `core/tracker.py` — agrupar detecções em trilhas

Este módulo existe só para o modo captura. Ele transforma uma sequência de
detecções independentes em **trilhas**: uma trilha é uma pessoa acompanhada ao
longo dos frames.

### 5.1 Associação por IoU, gulosa

A cada frame chegam D detecções e existem T trilhas ativas. É preciso decidir
qual detecção continua qual trilha. A métrica é **IoU** (interseção sobre união)
entre as caixas:

```python
inter = max(0, x1-x0) * max(0, y1-y0)
iou   = inter / (area_a + area_b - inter)
```

IoU vale 1 para caixas idênticas e 0 para disjuntas. O algoritmo:

```python
pares = [(iou, tid, di) for cada par com iou >= limiar]
pares.sort(reverse=True)                 # maior IoU primeiro
for _, tid, di in pares:
    if tid ja usado or di ja usado: continue
    casa(tid, di)
```

**Complexidade.** Montar os pares é O(T·D); ordenar é O(T·D · log(T·D)). Com
T e D em torno de 50, são ~2.500 pares — alguns microssegundos. Desprezível
frente aos 57 ms da detecção.

**Por que guloso e não atribuição ótima.** O algoritmo Húngaro daria a atribuição
de custo mínimo global, em O(n³). Para pessoas caminhando num corredor, com
deslocamento pequeno entre frames consecutivos, o guloso por IoU acerta
praticamente sempre — é o que o rastreador SORT faz, sem a parte do filtro de
Kalman. Num Pi 3B, gastar ciclos com atribuição ótima para ganhar caso raro não
se paga.

**Por que sem filtro de Kalman.** Kalman prevê a posição futura, o que ajuda
quando há oclusão longa ou movimento rápido com poucos frames. Aqui a detecção
roda a ~17 fps e o deslocamento por frame é pequeno. Fica como caminho de
melhoria se aparecerem trocas de identidade.

**Falha conhecida:** duas pessoas que se cruzam e se ocluem podem ter as trilhas
trocadas. A votação da fase 2 mitiga — uma trilha trocada acumula votos
divididos e cai em "Desconhecido" em vez de afirmar a identidade errada.

### 5.2 Ciclo de vida da trilha

- **Nascimento:** detecção que não casou com nenhuma trilha.
- **Morte:** `frame_atual - ultimo_frame_visto > max_missing_frames`.
- **Descarte:** ao morrer, trilha com menos de `min_track_frames` frames é
  jogada fora. Detecção isolada em um único frame é quase sempre falso positivo
  do detector, e gastar 285 ms de embedding nela é desperdício.

A tolerância a frames faltantes existe porque o detector perde rostos
esporadicamente (pessoa vira a cabeça, alguém passa na frente). Encerrar a
trilha na primeira falha fragmentaria uma pessoa em várias trilhas.

### 5.3 Seleção dos melhores recortes

Cada trilha mantém no máximo K recortes (padrão 3), em lista ordenada por
qualidade:

```python
if len(self.crops) < K:
    self.crops.append((qualidade, recorte, face_local))
    self.crops.sort(reverse=True)
elif qualidade > self.crops[-1][0]:      # melhor que o pior guardado
    self.crops[-1] = (qualidade, recorte, face_local)
    self.crops.sort(reverse=True)
```

Uma lista ordenada de tamanho 3 com reordenação a cada inserção é O(K log K) por
frame. Um *min-heap* daria O(log K), mas com K = 3 a diferença é ruído e a lista
é mais legível. Se K fosse 50, a escolha mudaria.

**A métrica de qualidade:**

```python
qualidade = variancia_do_laplaciano(cinza) * min(1.0, lado / 100.0)
```

O Laplaciano é um operador de segunda derivada: responde forte onde há bordas
nítidas. Sua **variância** sobre o recorte é um estimador clássico de nitidez —
imagem borrada tem poucas bordas fortes, logo variância baixa. Medido no
projeto: padrão nítido dá ~96.000, o mesmo padrão desfocado dá ~560.

O fator de tamanho entra **limitado a 1,0**. Sem o teto, um rosto enorme e
borrado venceria um menor e nítido. Com o teto, tamanho só desempata até 100 px
de lado; acima disso, decide a nitidez.

Nitidez pesa mais que tamanho porque borrão de movimento degrada o embedding mais
do que resolução baixa — e criança correndo produz exatamente borrão de
movimento.

### 5.4 A tradução dos landmarks (o detalhe que falharia em silêncio)

Ao salvar o recorte para a fase 2, é preciso salvar também os 5 pontos. Mas o
`alignCrop` espera pontos no sistema de coordenadas **da imagem que recebe**. Se
salvamos um recorte e mantemos os pontos do frame inteiro, o alinhamento sai
deslocado, o embedding sai diferente, e o limiar calibrado deixa de valer — sem
erro nenhum, apenas com qualidade pior.

```python
x0, y0 = canto superior esquerdo do recorte
local[0] -= x0;  local[1] -= y0          # caixa
for i in range(4, 14, 2):                 # os 5 pares (x, y)
    local[i]     -= x0
    local[i + 1] -= y0
```

A margem de 0,4 (recorte 1,8× a caixa) existe porque o alinhamento precisa de
contexto ao redor do rosto. Está verificado por teste: com margem 0,4 o resultado
é **idêntico ao pixel** ao alinhamento feito do frame inteiro, nas quatro
posições testadas incluindo bordas; com margem 0,05 a diferença média salta para
~15 níveis de intensidade.

Os 15 valores vão para o banco em JSON, arredondados a 3 casas. Precisão de
subpixel além disso não muda o alinhamento e só ocuparia espaço.

---

## 6. `core/database.py` — SQLite como banco e como fila

### 6.1 Por que SQLite, e por que WAL

```python
con.execute("PRAGMA journal_mode=WAL;")
con.execute("PRAGMA synchronous=NORMAL;")
```

**WAL** (Write-Ahead Logging) permite **um escritor e N leitores simultâneos**
sem bloqueio mútuo. É exatamente a topologia do projeto: o worker escreve, a API
e os scripts leem. No modo journal tradicional, uma escrita bloquearia as
leituras.

**`synchronous=NORMAL`** reduz drasticamente as chamadas de `fsync`. A troca:
uma queda de energia no instante errado pode perder as últimas transações. Num
cartão SD, onde cada `fsync` é caro e o desgaste é finito, a troca vale — e o
dado em risco é um registro de passagem, não algo irrecuperável.

Escolha do Kevenny, e apropriada: um banco embutido sem servidor é o certo para
um dispositivo isolado. Postgres exigiria um serviço rodando, mais RAM, e
configuração — sem ganho, porque não há concorrência real de escrita.

### 6.2 Os embeddings como BLOB

```python
blob = np.asarray(vec, dtype=np.float32).tobytes()          # gravar
vec  = np.frombuffer(r["vec"], dtype=np.float32)            # ler
```

512 bytes contíguos, sem serialização intermediária. `frombuffer` cria uma
**view** sobre os bytes, sem copiar — mas essa view é somente-leitura, e é por
isso que `load_gallery` faz `np.vstack(vecs).astype(np.float32)`: o `vstack`
materializa a matriz contígua que o produto interno precisa.

A alternativa (uma coluna por dimensão, ou JSON) custaria espaço e tempo de
parsing sem nenhum ganho — nunca se consulta uma dimensão isolada.

### 6.3 A fila de trilhas

O pipeline de duas fases usa duas tabelas:

```sql
tracks      (id, started_at, ended_at, frames, status, person_id, name, score, votes)
track_crops (id, track_id, path, quality, face)
```

`status` assume `pendente` → `processado` (ou `descartado`). Isso é uma **fila
persistente**: o worker é produtor, o `recognize_batch.py` é consumidor.

Vantagens sobre um broker (Redis, RabbitMQ) neste contexto: durabilidade sem
configuração (sobrevive a reboot), nenhum serviço adicional para manter, e
inspeção trivial com uma consulta SQL. O que falta hoje é **reivindicação
atômica** — dois consumidores simultâneos pegariam as mesmas trilhas. Enquanto
há um consumidor só, não é problema; para múltiplos, o passo é marcar
`processando` no mesmo comando que seleciona.

### 6.4 Migração aditiva

```python
existentes = {r["name"] for r in con.execute("PRAGMA table_info(embeddings)")}
for coluna, tipo in (("snapshot_path","TEXT"), ("created_at","REAL"), ("quality","REAL")):
    if coluna not in existentes:
        con.execute(f"ALTER TABLE embeddings ADD COLUMN {coluna} {tipo}")
```

`CREATE TABLE IF NOT EXISTS` não altera tabela existente, então um banco criado
por versão anterior ficaria sem as colunas novas. A migração é **só aditiva**:
nunca remove nem reescreve. Cadastro antigo continua válido, apenas com
`snapshot_path` nulo — o que a interface reporta explicitamente em vez de fingir
que a foto não existe por outro motivo.

### 6.5 A consulta de presença

O ponto sutil: cada modo grava em tabela diferente. Ler só uma devolveria
chamada vazia dependendo do modo em uso. A consulta une as duas origens antes de
agrupar:

```sql
SELECT person_id, name, MIN(ts), MAX(fim), COUNT(*), MAX(score),
       GROUP_CONCAT(DISTINCT fonte)
FROM (
    SELECT ..., 'captura'  AS fonte FROM tracks WHERE status='processado' ...
    UNION ALL
    SELECT ..., 'realtime' AS fonte FROM events WHERE is_known=1 ...
)
GROUP BY person_id
```

`UNION ALL` em vez de `UNION` porque não há duplicatas a eliminar e `UNION`
custaria uma ordenação. O `GROUP_CONCAT(DISTINCT fonte)` preserva a procedência,
o que permite auditar um dia em que os dois modos foram usados.

### 6.55 A correção manual como fonte separada

Duas tabelas pequenas guardam a intervenção humana na chamada:

```sql
attendance_overrides (dia, person_id, presente, motivo, autor, created_at)
attendance_closures  (dia, closed_at, autor, presentes, correcoes)
```

A chave primária composta `(dia, person_id)` garante uma correção por pessoa por
dia, e o `ON CONFLICT ... DO UPDATE` faz o upsert sem precisar consultar antes.

**Por que tabela separada em vez de uma coluna em `events`.** Se a correção
sobrescrevesse o resultado do reconhecimento, o dado original desapareceria — e
com ele a possibilidade de medir o acerto do sistema. Mantendo os dois lados, a
presença efetiva é calculada na leitura:

```
presente = correção, quando existe;  senão, detecção automática
```

E a diferença entre os dois vira métrica: `presente` marcado à mão onde não houve
detecção é falso negativo; `ausente` marcado à mão onde houve detecção é falso
positivo. É a mesma ideia do `calibrate_threshold.py`, mas aplicada ao nível da
chamada em vez do evento individual.

A regra "correção existe só na discordância" também mantém a tabela pequena: num
dia em que o reconhecimento acerta tudo, ela fica vazia.

### 6.6 VACUUM fora de transação

```python
con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
con.execute("VACUUM")
```

O SQLite recusa `VACUUM` dentro de transação, e o `with con:` do Python abre uma
implicitamente. Daí a conexão separada em autocommit. Bug encontrado por teste.

---

## 7. `core/storage.py` — particionamento por data

```
data/snapshots/AAAAMMDD/HHMMSS_mmm_nome.jpg
```

**Por que particionar por dia.** A retenção se resume a remover diretórios
inteiros, sem varrer arquivo por arquivo comparando data de modificação. Com
milhares de imagens, a diferença é entre uma chamada de sistema e uma varredura
completa. O `cleanup_snapshots.py` aproveita isso: como o nome da pasta é
`AAAAMMDD`, comparação de string equivale a comparação de data.

**O nome do arquivo** tem hora, milissegundo e rótulo. O milissegundo evita
colisão quando duas pessoas são registradas no mesmo segundo.

**`_safe()`** filtra o nome por lista de permissão (`isalnum()` mais `-` e `_`).
Isso serve a dois propósitos: impede que um nome com `../` escape do diretório, e
garante nome de arquivo válido. Acentos sobrevivem, porque `'ã'.isalnum()` é
verdadeiro em Unicode; espaços e pontuação são removidos.

`remove_dir()` valida que o caminho resolvido está sob a base antes de apagar —
proteção contra *path traversal* na remoção.

---

## 8. `core/config.py` — configuração com acesso por atributo

```python
class Config(dict):
    def __getattr__(self, item):
        value = self[item]
        return Config(value) if isinstance(value, dict) else value
```

`dict` com acesso por ponto, recursivo. Permite `cfg.camera.rtsp_url` em vez de
`cfg["camera"]["rtsp_url"]`. Escolha do Kevenny.

**Uma pegadinha do desenho:** `__getattr__` embrulha dicionários em `Config`, mas
`.get()` é o `dict.get` original e devolve dicionário cru. Então
`cfg.worker.get("mode")` funciona (valor escalar), mas
`cfg.get("tracking").iou_threshold` falharia. O código usa consistentemente
`.get(chave, padrao)` para seções opcionais, o que também é o que mantém
compatibilidade quando um `config.yaml` antigo não tem as chaves novas.

**Ordem de resolução:** caminho explícito → variável `FACIAL_CONFIG` →
`config.yaml` → `config.example.yaml`. O último nível permite rodar o projeto
recém-clonado sem criar arquivo.

**Cache com invalidação explícita.** `load_config()` guarda o resultado; a troca
de modo a quente precisa reler o disco, então existe `reload_config()` que limpa
o cache. Deliberadamente só o `worker.mode` é aplicado a quente: recarregar tudo
no meio da execução deixaria o processo com metade dos valores antigos e metade
novos.

**Tradução de erro.** `ConfigError` converte a exceção do PyYAML — dez níveis de
pilha falando de "block mapping" — em mensagem que aponta a linha, mostra o
trecho ao redor e lista as causas comuns. `ConfigError` herda de `Exception`, não
de `SystemExit`, de propósito: o `except Exception` que protege a releitura a
quente precisa capturá-la, senão um YAML momentaneamente inválido derrubaria o
serviço.

---

## 9. `core/draw.py` — anotação

O menor módulo, 31 linhas. Desenha retângulo e rótulo com fundo sólido para o
texto ficar legível sobre qualquer imagem. Cores em BGR (convenção do OpenCV),
verde para conhecido, vermelho para desconhecido. `crop_face()` recorta com
margem proporcional, usado nas miniaturas do cadastro.

---

## 10. Onde está o tempo, e o que escala

Medido no Pi 3B com substream 640×480:

| Operação | Custo | Escala com |
|---|---|---|
| Decode do H.264 | contínuo, ~1 núcleo | resolução |
| `detect()` | 57 ms | área da imagem (não com o nº de rostos) |
| `embed()` | 285 ms | **número de rostos** |
| `match()` | microssegundos | nº de amostras (irrelevante) |
| Escrita de snapshot | ~10 ms | — |
| Consulta de presença | milissegundos | nº de registros do dia |

A leitura prática: **o embedding domina tudo**. É por isso que reduzir
`detect_width` quase não ajuda, que a galeria pode crescer sem preocupação, e que
o modo captura — que paga o embedding uma vez por pessoa em vez de uma vez por
rosto-por-frame — é o que viabiliza o cenário de grupos.

---

## 11. Limitações conhecidas do desenho

Registradas para quem for dar manutenção:

1. **A fila não tem reivindicação atômica.** Um consumidor só, hoje. Múltiplos
   consumidores exigem marcar `processando` no mesmo `UPDATE` que seleciona.
   Enquanto isso não existir, a limpeza tem uma trava: trilha pendente antiga
   bloqueia a remoção dos recortes, porque indica lote parado.
2. **O rastreador pode trocar identidades** quando duas pessoas se cruzam com
   oclusão. A votação mitiga, mas não elimina.
3. **`match()` é max-pooling sobre amostras**, então cadastro inflado aumenta
   falso positivo (seção 4.3).
4. **Sem identificador externo.** O casamento com outro sistema depende de
   `person_id` interno ou de nome, e nome é frágil.
5. **A câmera V4L2 é exclusiva.** Worker e API não podem abrir o mesmo
   `/dev/video0`; contornado com o worker publicando o frame em tmpfs.
6. **Sem HTTPS.** O token opcional protege contra acesso casual, não contra
   captura de tráfego.
