"""Calibra o `recognition.cosine_threshold` a partir dos reconhecimentos reais.

O limiar decide entre "é a Maria" e "é um desconhecido". O valor de fábrica
(0.363) é o ponto de partida do SFace, medido num conjunto de fotos de
referência — não no seu corredor. Iluminação, ângulo, distância e a qualidade do
cadastro deslocam toda a distribuição. Calibrar no chute costuma trocar um
problema pelo outro.

De onde vêm os DADOS
--------------------
Das duas origens, unidas: `events` (modo realtime) e `tracks` (modo captura).
Ler só uma deixava a calibração cega no modo usado em produção.

De onde vem a VERDADE
---------------------
Duas fontes, em ordem de preferência:

1. **Chamadas conferidas** (aba Chamada do painel). É a fonte principal, porque
   sai da operação normal, sem trabalho extra:
     - aluno marcado AUSENTE numa chamada fechada -> as detecções dele naquele
       dia foram identificação equivocada (falso positivo);
     - aluno SEM correção numa chamada FECHADA -> alguém revisou e concordou,
       então as detecções dele estão confirmadas.
   A exigência de a chamada estar **fechada** é essencial: em chamada aberta, a
   ausência de correção significa "ninguém olhou", não "está correto".

2. **Revisão manual** (`--review`), evento por evento. Serve para dias que não
   foram conferidos. Tem precedência sobre o item 1 por ser mais específica.

Uso:
    python scripts/calibrate_threshold.py              # relatório e sugestão
    python scripts/calibrate_threshold.py --review     # marca acerto/erro à mão
    python scripts/calibrate_threshold.py --simular 0.45   # efeito de um limiar
"""

import argparse
import json
import os
import socket
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config_or_exit, project_path  # noqa: E402
from core.database import Database  # noqa: E402

LABELS_FILE = "data/calibracao.json"


def chave(d: dict) -> str:
    """Identificador de uma detecção. `id` repete entre origens, então precisa do par."""
    return f"{d['fonte']}:{d['id']}"


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "IP_DO_PI"


def _migrar_json(db: Database) -> int:
    """Move os rótulos do arquivo JSON antigo para o banco, uma vez.

    Os rótulos manuais moraram num JSON enquanto só este script os escrevia.
    Quando o painel ganhou o botão "não é essa pessoa", manter os dois lugares
    criaria uma TERCEIRA fonte de verdade sobre acerto e erro (chamada, JSON,
    painel) — e divergência entre fontes já causou dois bugs neste projeto.

    O arquivo é renomeado para `.migrado` em vez de apagado: se algo der
    errado, o dado original continua lá.
    """
    path = project_path(LABELS_FILE)
    if not path.exists():
        return 0
    try:
        cru = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    # Chave antiga era só o número, e significava evento.
    normalizado = {(k if ":" in str(k) else f"evento:{k}"): v
                   for k, v in (cru or {}).items()}
    n = db.importar_rotulos(normalizado)
    try:
        path.rename(path.with_suffix(path.suffix + ".migrado"))
    except OSError:
        pass
    if n:
        print(f"  {n} rótulo(s) migrados do {LABELS_FILE} para o banco.")
    return n


def _load_labels(db: Database) -> dict:
    """Rótulos manuais, agora do banco. Migra o JSON antigo na primeira vez."""
    _migrar_json(db)
    return db.detection_labels()


def _save_label(db: Database, chave: str, rotulo: str) -> None:
    fonte, _, ident = chave.partition(":")
    db.set_detection_label(fonte, int(ident), rotulo, autor="review")


def rotulos_da_chamada(db: Database, deteccoes: list[dict]) -> dict:
    """Deriva certo/errado das chamadas CONFERIDAS.

    O dia de uma detecção é calculado no fuso local, igual ao que o painel usa
    ao gravar a correção — comparar com UTC jogaria detecções da noite para o
    dia seguinte.
    """
    fechados = db.closed_days()
    overrides = db.all_overrides()
    labels = {}
    for d in deteccoes:
        if not d["person_id"]:
            continue                     # desconhecido não tem identidade a confirmar
        dia = time.strftime("%Y-%m-%d", time.localtime(d["ts"]))
        if dia not in fechados:
            continue                     # ninguém conferiu: silêncio não é aprovação
        corr = overrides.get((dia, int(d["person_id"])))
        if corr is None:
            labels[chave(d)] = "certo"   # conferido e não corrigido = confirmado
        elif corr == 0:
            labels[chave(d)] = "errado"  # marcado ausente = identificou errado
        # corr == 1 (marcado presente) não rotula detecção: não houve detecção
    return labels


def incluir_rotuladas(db: Database, deteccoes: list[dict]) -> list[dict]:
    """Acrescenta as detecções rotuladas que ficaram fora da janela recente.

    O relatório analisa as N mais recentes, o que é certo para o histograma —
    ele descreve a atividade atual. Mas o cálculo do limiar depende dos
    RÓTULOS, e rótulo não envelhece: cada um é alguém que olhou uma foto e
    decidiu.

    Sem isto, marcar 211 detecções e depois deixar o worker rodando fazia as
    marcações saírem da janela e serem ignoradas em silêncio — o relatório
    dizia "211 manuais" e usava zero. Pior que não ter a informação é exibi-la
    e não usá-la.
    """
    presentes = {chave(d) for d in deteccoes}
    faltando = [k for k in _load_labels(db) if k not in presentes]
    if not faltando:
        return deteccoes
    extras = db.detections_by_keys(faltando)
    if extras:
        print(f"  (+{len(extras)} detecção(ões) rotuladas fora da janela, "
              "incluídas no cálculo)")
    return deteccoes + extras


def juntar_rotulos(db: Database, deteccoes: list[dict]) -> tuple[dict, dict]:
    """Combina as duas fontes. Manual vence, por ser mais específica.

    `origem` conta só o que é UTILIZÁVEL — rótulo cuja detecção existe na
    análise. Contar rótulo que não entra na conta foi exatamente o que criou
    a contradição "211 manuais / 0 erros" na saída.
    """
    presentes = {chave(d) for d in deteccoes}
    derivados = rotulos_da_chamada(db, deteccoes)
    manuais = {k: v for k, v in _load_labels(db).items() if k in presentes}
    orfaos = len(_load_labels(db)) - len(manuais)

    combinado = dict(derivados)
    combinado.update(manuais)
    origem = {"chamada": len(derivados), "manual": len(manuais),
              "total": len(combinado), "orfaos": orfaos}
    return combinado, origem


def _histogram(scores: list[float], largura: int = 40) -> None:
    """Histograma em texto: mostra onde os scores se concentram."""
    if not scores:
        return
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        print(f"      todos em {lo:.3f}")
        return
    faixas = 10
    passo = (hi - lo) / faixas
    for i in range(faixas):
        a, b = lo + i * passo, lo + (i + 1) * passo
        n = sum(1 for s in scores if (a <= s < b or (i == faixas - 1 and s == b)))
        barra = "#" * int(round(n / max(1, len(scores)) * largura))
        print(f"      {a:.3f}–{b:.3f}  {barra} {n}")


def relatorio(db: Database, cfg, limite: int) -> list[dict]:
    limiar = float(cfg.recognition.cosine_threshold)
    deteccoes = db.list_detections(limit=limite)
    conhecidos = [e for e in deteccoes if e["is_known"]]
    desconhecidos = [e for e in deteccoes if not e["is_known"]]
    por_fonte = {}
    for d in deteccoes:
        por_fonte[d["fonte"]] = por_fonte.get(d["fonte"], 0) + 1

    print(f"\nLimiar atual: {limiar}")
    print(f"Reconhecimentos analisados: {len(deteccoes)} "
          f"({len(conhecidos)} identificados, {len(desconhecidos)} desconhecidos)")
    if por_fonte:
        print("  origem: " + ", ".join(f"{v} de {k}" for k, v in por_fonte.items()))
    if not deteccoes:
        print("\n  Nada registrado ainda. Deixe o worker rodando e, no modo "
              "captura, rode o lote:  python scripts/recognize_batch.py")

    if conhecidos:
        scores = sorted(e["score"] for e in conhecidos)
        print(f"\nScores dos IDENTIFICADOS (min {scores[0]:.3f} / "
              f"mediana {scores[len(scores) // 2]:.3f} / max {scores[-1]:.3f}):")
        _histogram(scores)
        print("\n  Um falso positivo aparece aqui como score BAIXO — é o rosto")
        print("  de outra pessoa que mesmo assim passou do limiar.")

    if desconhecidos:
        scores = sorted((e["score"] for e in desconhecidos), reverse=True)
        print(f"\nScores dos DESCONHECIDOS (maior {scores[0]:.3f}):")
        print("  Se você reconhece alguém cadastrado entre estes, o limiar está")
        print("  alto demais (falso negativo).")

    return deteccoes


def review(db: Database, cfg, deteccoes: list[dict], ip: str, porta: int) -> None:
    conhecidos = [d for d in deteccoes if d["is_known"]]
    if not conhecidos:
        print("\nSem identificações para revisar ainda.")
        return

    combinado, _ = juntar_rotulos(db, deteccoes)
    # Do menor para o maior score: os erros se concentram no início, então você
    # encontra os problemas nas primeiras respostas.
    conhecidos.sort(key=lambda d: d["score"])
    pendentes = [d for d in conhecidos if chave(d) not in combinado]

    print(f"\n{len(pendentes)} a revisar à mão "
          f"({len(combinado)} já rotulados, sendo {len(manuais)} manuais e o "
          f"resto vindo de chamadas conferidas).")
    if not pendentes:
        print("  Nada pendente — as chamadas conferidas já cobriram tudo.")
    print("Responda:  s = acertou   n = ERROU (outra pessoa)   p = pular   q = sair\n")

    for d in pendentes:
        print(f"  score {d['score']:.3f}  ->  identificou como '{d['name']}' "
              f"({d['fonte']})")
        if d["snapshot_path"]:
            # Cada origem tem sua rota, porque ficam em bases diferentes.
            base = "/snapshots/" if d["fonte"] == "evento" else "/tracks/"
            print(f"  foto: http://{ip}:{porta}{base}{d['snapshot_path']}")
        try:
            resp = input("  acertou? [s/n/p/q] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(interrompido)")
            break
        if resp == "q":
            break
        if resp in ("s", "n"):
            _save_label(db, chave(d), "certo" if resp == "s" else "errado")
        print()

    combinado, origem = juntar_rotulos(db, deteccoes)
    sugerir(deteccoes, combinado, origem, float(cfg.recognition.cosine_threshold))


def cobertura(db: Database, deteccoes: list[dict], labels: dict,
              origem: dict) -> None:
    """Quanto do material já foi conferido, e o que as correções revelaram.

    Existe para responder "minhas marcações adiantaram alguma coisa?". O
    relatório de limiar mostra o RESULTADO; este mostra a base sobre a qual
    ele foi calculado — que é o que diz se dá para confiar nele e o que falta
    para melhorar.
    """
    por_chave = {chave(d): d for d in deteccoes}
    rotuladas = [k for k in labels if k in por_chave]
    certos = [k for k in rotuladas if labels[k] == "certo"]
    errados = [k for k in rotuladas if labels[k] == "errado"]

    total = len(deteccoes)
    pct = (len(rotuladas) / total * 100) if total else 0.0

    print("\n--- Cobertura da conferência ---")
    print(f"  {len(rotuladas)} de {total} detecções conferidas ({pct:.0f}%)")
    print(f"    de chamadas fechadas: {origem['chamada']}")
    print(f"    marcadas à mão:       {origem['manual']}")
    if origem.get("orfaos"):
        print(f"  ⚠ {origem['orfaos']} rótulo(s) apontam para detecção que não")
        print("    existe mais — provavelmente apagada pela retenção. O "
              "trabalho")
        print("    de marcá-las se perdeu junto com as fotos.")
    print(f"  veredito: {len(certos)} acerto(s), {len(errados)} erro(s)")

    # O que as correções de identidade revelam: quais cadastros o sistema
    # confunde entre si. Isso é mais acionável que o limiar, porque aponta
    # exatamente quem recadastrar.
    detalhes = db.detection_labels_detalhados()
    confusoes = Counter()
    sem_alvo = 0
    for k in errados:
        info = detalhes.get(k) or {}
        disse = por_chave[k].get("name") or "?"
        if info.get("pessoa_correta"):
            confusoes[(disse, info["nome_correto"])] += 1
        else:
            sem_alvo += 1

    if confusoes:
        # Score POR PAR: é o que distingue "o sistema é ruim em geral" de
        # "um cadastro específico está contaminando tudo". Sem essa quebra, a
        # sobreposição parece um problema difuso e a recomendação vira
        # genérica — quando muitas vezes um par responde por quase tudo.
        scores_par = {}
        for k in errados:
            info = detalhes.get(k) or {}
            if not info.get("pessoa_correta"):
                continue
            par = (por_chave[k].get("name") or "?", info["nome_correto"])
            scores_par.setdefault(par, []).append(por_chave[k]["score"])

        print("\n  Confusões identificadas (o sistema disse -> era):")
        for (disse, era), n in confusoes.most_common(8):
            ss = scores_par.get((disse, era), [])
            faixa = (f"  scores {min(ss):.3f}–{max(ss):.3f}" if ss else "")
            print(f"    {disse} -> {era}: {n}x{faixa}")
        print("  Par que se repete é problema de cadastro das DUAS pessoas —")
        print("  mais amostras de ambas, com ângulos variados, separa melhor")
        print("  que mexer no limiar.")

        # Um par dominante muda completamente o diagnóstico.
        if len(confusoes) >= 1:
            (disse, era), n = confusoes.most_common(1)[0]
            fatia = n / max(1, sum(confusoes.values()))
            if fatia >= 0.7 and n >= 5:
                print(f"\n  ⚠ {fatia*100:.0f}% das confusões são o MESMO par: "
                      f"{disse} -> {era}.")
                print("    Isso não é 'o reconhecimento está ruim', é um")
                print(f"    cadastro específico. Antes de qualquer ajuste, abra")
                print(f"    as amostras de '{disse}' no painel e confira de quem")
                print("    é o rosto nelas. Cadastro feito na frente da câmera")
                print("    com outra pessoa em quadro pode ter capturado a")
                print("    pessoa errada — e aí a galeria de um contém o rosto")
                print("    do outro, o que produz exatamente este padrão.")
    if sem_alvo:
        print(f"\n  {sem_alvo} erro(s) sem dizer de quem era. Informar a pessoa")
        print("  certa ao marcar transforma o rótulo em par (detecção -> quem")
        print("  é), que é o que permite medir; só 'errado' diz o que NÃO é.")


def sugerir(deteccoes: list[dict], labels: dict, origem: dict,
            limiar_atual: float) -> None:
    por_chave = {chave(d): d for d in deteccoes}
    certos = [por_chave[k]["score"] for k, v in labels.items()
              if v == "certo" and k in por_chave]
    errados = [por_chave[k]["score"] for k, v in labels.items()
               if v == "errado" and k in por_chave]

    print("\n--- Sugestão de limiar ---")
    print(f"  rótulos: {origem['chamada']} de chamadas conferidas + "
          f"{origem['manual']} manuais")
    print(f"  acertos: {len(certos)} | erros: {len(errados)}")

    if not certos and not errados:
        print("\n  Sem rótulo nenhum, não há o que calcular. Dois caminhos:")
        print("    1. Confira e FECHE chamadas no painel (recomendado — sai da")
        print("       operação normal e cobre os dois modos).")
        print("    2. Rotule evento por evento: --review")
        return

    if certos:
        print(f"  menor score de um ACERTO:  {min(certos):.3f}")
    if errados:
        print(f"  maior score de um ERRO:    {max(errados):.3f}")

    if not errados:
        print("\n  Nenhum falso positivo entre os rotulados. Se eles acontecem,")
        print("  marque o aluno como ausente na chamada do dia e feche-a.")
        return
    if not certos:
        print(f"\n  Só há erros rotulados. Suba o limiar acima de {max(errados):.3f}")
        print("  e verifique se as pessoas certas continuam sendo reconhecidas.")
        return

    pior_erro, pior_acerto = max(errados), min(certos)
    if pior_erro < pior_acerto:
        sugerido = round((pior_erro + pior_acerto) / 2, 3)
        margem = pior_acerto - pior_erro
        print(f"\n  As duas distribuições estão SEPARADAS "
              f"({pior_erro:.3f} < {pior_acerto:.3f}).")
        print(f"  Limiar sugerido: {sugerido}   (atual: {limiar_atual})")
        print(f"  Margem: {margem:.3f} entre o pior erro e o pior acerto.")
        print("  Ponto médio: fica o mais longe possível do pior caso de cada")
        print("  lado, o que dá a maior tolerância a variação futura.")

        # A margem é a medida de QUALIDADE da calibração, e ela costuma
        # ENCOLHER conforme se rotula mais. Isso não é piora: é o estimador
        # encontrando casos difíceis que já existiam e ainda não tinham
        # aparecido. Margem larga com poucos rótulos é falsa sensação de
        # folga — daí o aviso.
        if margem < 0.05:
            print("\n  ⚠ Margem estreita: qualquer variação de luz ou ângulo")
            print("    atravessa esse vão. Trate o limiar como provisório e")
            print("    priorize melhorar cadastro e enquadramento.")
        if len(errados) < 5 or len(certos) < 10:
            print("\n  Base ainda pequena. Conforme você conferir mais, essa")
            print("    margem tende a ENCOLHER — é o esperado, e significa que")
            print("    o número está ficando mais realista, não pior.")
        print("\n  No config.yaml:")
        print("    recognition:")
        print(f"      cosine_threshold: {sugerido}")
        print("\n  Confira o efeito antes de aplicar:")
        print(f"    python scripts/calibrate_threshold.py --simular {sugerido}")
        print("  Depois:  sudo systemctl restart facial-worker")
    else:
        print(f"\n  ⚠ As distribuições se SOBREPÕEM "
              f"({pior_acerto:.3f} <= {pior_erro:.3f}).")
        print("  Nenhum limiar separa os dois casos — mexer nele só troca")
        print("  falso positivo por falso negativo. O que resolve de verdade:")
        print("    1. cadastrar MAIS amostras da pessoa, com ângulos e luz variados;")
        print("    2. cadastrar também as outras pessoas que passam — assim o")
        print("       rosto delas casa com elas, não com quem já está cadastrado;")
        print("    3. aumentar recognition.min_face_size (rosto pequeno gera")
        print("       embedding ruim e é fonte clássica de confusão);")
        print("    4. melhorar o enquadramento: rosto de frente, sem contraluz.")


def simular(db: Database, cfg, limite: int, novo: float) -> None:
    deteccoes = db.list_detections(limit=limite)
    labels, _ = juntar_rotulos(db, deteccoes)
    conhecidos = [d for d in deteccoes if d["is_known"]]
    rejeitados = [d for d in conhecidos if d["score"] < novo]

    print(f"\nCom cosine_threshold = {novo} "
          f"(atual: {cfg.recognition.cosine_threshold}):")
    print(f"  {len(rejeitados)} das {len(conhecidos)} identificações passariam a "
          f"'Desconhecido'.")

    certos = sum(1 for d in rejeitados if labels.get(chave(d)) == "certo")
    errados = sum(1 for d in rejeitados if labels.get(chave(d)) == "errado")
    if certos or errados:
        print(f"  Dentre as rotuladas: eliminaria {errados} erro(s) "
              f"e perderia {certos} acerto(s).")
        if errados and not certos:
            print("  Ou seja: só ganho, sem perda — bom candidato.")
        elif certos:
            print("  Perder acerto significa criança presente virando não "
                  "identificada. Pese isso contra o ganho.")
    else:
        print("  (nenhuma das afetadas está rotulada — confira e feche chamadas "
              "no painel para saber quais eram erro de verdade)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra o limiar de reconhecimento.")
    ap.add_argument("--review", action="store_true",
                    help="revisa à mão o que as chamadas conferidas não cobriram")
    ap.add_argument("--simular", type=float, metavar="LIMIAR",
                    help="mostra o efeito de um limiar sem aplicá-lo")
    ap.add_argument("--limite", type=int, default=500,
                    help="quantos reconhecimentos analisar (padrão 500)")
    args = ap.parse_args()

    cfg = load_config_or_exit()
    db = Database(cfg.storage.db_path)
    porta = int(cfg.api.get("port", 8000))

    if args.simular is not None:
        simular(db, cfg, args.limite, args.simular)
    elif args.review:
        deteccoes = incluir_rotuladas(db, relatorio(db, cfg, args.limite))
        review(db, cfg, deteccoes, _local_ip(), porta)
    else:
        deteccoes = relatorio(db, cfg, args.limite)
        deteccoes = incluir_rotuladas(db, deteccoes)
        labels, origem = juntar_rotulos(db, deteccoes)
        cobertura(db, deteccoes, labels, origem)
        sugerir(deteccoes, labels, origem, float(cfg.recognition.cosine_threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
