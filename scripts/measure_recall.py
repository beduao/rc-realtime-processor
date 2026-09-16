#!/usr/bin/env python3
"""Mede o recall: das pessoas que passaram, quantas o sistema identificou.

Recall não sai dos dados do sistema sozinho. Quem passou e não foi detectado não
deixa rastro nenhum — nem evento, nem trilha, nem log. É invisível justamente
por ter falhado. Então a medição precisa de uma lista externa de quem esteve
lá, e este script cruza essa lista com o que o banco registrou.

Duas fontes para essa lista, uma para cada momento do projeto:

  --passagens registro.csv   teste controlado: pessoas conhecidas atravessam o
                             corredor N vezes e você anota. Denominador exato,
                             granularidade por passagem.

  --chamada                  operação real: as chamadas conferidas e fechadas.
                             Aluno marcado presente à mão sem detecção nenhuma
                             é falso negativo. Granularidade por dia.

E reporta as duas métricas separadas, porque elas medem coisas diferentes:

  recall por PASSAGEM  — diagnostica o pipeline. Cada travessia é uma chance.
  recall por DIA       — é o que decide se o produto serve. Para a chamada,
                         basta a criança ser pega UMA vez na manhã.

O segundo é sempre melhor que o primeiro, e por muito. 60% por passagem pode
ser 99% por dia. Olhar só o número por passagem assusta sem motivo; olhar só o
por dia esconde que o pipeline está no limite.

Formato do CSV (cabeçalho opcional, `;` ou `,`):

    nome;hora;rodada
    Aluno A;07:32:10;sozinho
    Aluno B;07:32:18;sozinho
    Aluno A;07:40:02;grupo

`hora` pode ser HH:MM ou HH:MM:SS (assume hoje, ou o dia de --data), ou
data e hora completas (AAAA-MM-DD HH:MM:SS).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config          # noqa: E402
from core.database import Database           # noqa: E402

# Os degraus do funil, em ordem. Cada um tem causa e solução diferentes — é por
# isso que a medição os separa em vez de devolver uma porcentagem só.
FUNIL = [
    ("acerto", "Identificada corretamente"),
    ("errada", "Identificada como OUTRA pessoa"),
    ("desconhecido", "Detectada, não reconheceu (abaixo do limiar)"),
    ("sem_recorte", "Rastreada, nenhum recorte legível"),
    ("nao_detectada", "Nunca detectada"),
]

DIAGNOSTICO = {
    "nao_detectada": "posição/altura da câmera, iluminação, min_face_size, resolução",
    "sem_recorte": "qualidade da captura, top_k, nitidez mínima",
    "desconhecido": "cadastro: mais amostras, ângulos e luz variados",
    "errada": "cadastro de quem NÃO está na galeria; limiar",
}


# --------------------------------------------------------------------------- #
# Leitura do registro de passagens
# --------------------------------------------------------------------------- #
def _parse_hora(valor: str, base_dia: str | None) -> float:
    """Converte a hora anotada em epoch. Aceita HH:MM[:SS] ou data completa."""
    valor = valor.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.mktime(time.strptime(valor, fmt))
        except ValueError:
            pass
    dia = base_dia or time.strftime("%Y-%m-%d")
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return time.mktime(time.strptime(f"{dia} {valor}",
                                             f"%Y-%m-%d {fmt}"))
        except ValueError:
            pass
    raise ValueError(f"hora não reconhecida: {valor!r}")


def ler_passagens(caminho: Path, base_dia: str | None) -> list[dict]:
    texto = caminho.read_text(encoding="utf-8-sig")
    # Detecta o separador em vez de exigir um: planilha em português salva com
    # `;` e é o caso mais provável de quem anotou no Excel.
    amostra = texto.splitlines()[0] if texto.splitlines() else ""
    sep = ";" if amostra.count(";") >= amostra.count(",") else ","

    passagens = []
    for n, linha in enumerate(csv.reader(texto.splitlines(), delimiter=sep), 1):
        if not linha or not linha[0].strip():
            continue
        campos = [c.strip() for c in linha]
        if campos[0].startswith("#"):
            continue                                  # comentário
        if n == 1 and campos[0].lower() in ("nome", "pessoa", "aluno"):
            continue                                  # cabeçalho
        if len(campos) < 2:
            print(f"  [aviso] linha {n} incompleta, ignorada: {linha}")
            continue
        try:
            ts = _parse_hora(campos[1], base_dia)
        except ValueError as exc:
            print(f"  [aviso] linha {n}: {exc}")
            continue
        passagens.append({
            "nome": campos[0],
            "ts": ts,
            "rodada": campos[2] if len(campos) > 2 and campos[2] else "—",
        })
    passagens.sort(key=lambda p: p["ts"])
    return passagens


# --------------------------------------------------------------------------- #
# Classificação de uma passagem
# --------------------------------------------------------------------------- #
def _passagens_proximas(passagens: list[dict], cooldown: float) -> int:
    """Quantas passagens caem dentro do cooldown da anterior da MESMA pessoa.

    São exatamente as que o worker suprime no modo realtime, e portanto as que
    o relatório contaria como "nunca detectada" sem aviso.
    """
    ultima: dict[str, float] = {}
    n = 0
    for p in sorted(passagens, key=lambda x: x["ts"]):
        anterior = ultima.get(p["nome"])
        if anterior is not None and p["ts"] - anterior < cooldown:
            n += 1
        ultima[p["nome"]] = p["ts"]
    return n


def atribuir(passagens: list[dict], deteccoes: list[dict], pessoas: dict,
             janela: float) -> int:
    """Casa detecções com passagens e classifica cada passagem no funil.

    Precisa ser feito em conjunto, não passagem por passagem. Na rodada em
    grupo as travessias ficam a 1 ou 2 segundos uma da outra, e a janela de
    tolerância (que existe porque a hora foi anotada à mão) cobre as vizinhas.
    Classificar isoladamente faz a detecção CORRETA de uma pessoa cair na
    janela da outra e virar "identificada como outra pessoa" — erro inventado
    exatamente na condição mais difícil, onde a medição precisa ser confiável.

    Duas regras resolvem:

    1. **Cada detecção é consumida uma vez.** Mesma ideia da associação gulosa
       do rastreador: percorre em ordem de preferência e vai marcando.
    2. **"Pessoa errada" exige que a identidade atribuída NÃO esteja entre as
       esperadas na janela.** Se a detecção diz "Aluno A" e o Aluno A também
       estava atravessando naquele instante, a detecção é dele — não um erro
       de quem passou junto.

    Devolve quantas detecções sobraram sem dono (indicador de trilha quebrada:
    a mesma travessia rastreada mais de uma vez).
    """
    for p in passagens:
        p["pid"] = pessoas.get(p["nome"].strip().lower())
        p["resultado"] = None
        p["score"] = None
        p["visto"] = None
        p["ambiguo"] = False

    livres = list(deteccoes)
    for d in livres:
        d["_usada"] = False

    def candidatas(d, filtro):
        return sorted(
            (p for p in passagens
             if p["resultado"] is None
             and abs(d["ts"] - p["ts"]) <= janela
             and filtro(p)),
            key=lambda p: abs(d["ts"] - p["ts"]))

    # Fase 1: acertos. Detecção nomeada casa com passagem da mesma pessoa,
    # a mais próxima no tempo. Primeiro porque é a atribuição mais confiável
    # que existe — nome e tempo concordam.
    for d in livres:
        if d["person_id"] is None:
            continue
        alvo = candidatas(d, lambda p: p["pid"] == d["person_id"])
        if alvo:
            p = alvo[0]
            # Contestação: existe passagem de OUTRA pessoa mais próxima no tempo
            # desta detecção do que a pessoa nomeada. Então a detecção pode ser
            # o acerto de quem foi nomeado ou o erro de quem passou junto, e o
            # dado não decide — hora anotada à mão não tem essa resolução.
            # Marcar, em vez de escolher em silêncio, é o que permite o
            # relatório dar faixa em vez de um número otimista.
            # `<=`, não `<`: no protocolo em grupo todos recebem a MESMA hora,
            # então a distância empata e a ambiguidade é total — usar `<`
            # deixaria justamente o pior caso passar sem aviso.
            dist = abs(d["ts"] - p["ts"])
            p["ambiguo"] = any(
                q["resultado"] is None and q["pid"] != d["person_id"]
                and abs(d["ts"] - q["ts"]) <= dist
                for q in passagens)
            p.update(resultado="acerto", score=d["score"],
                     crop=d.get("crop"), fonte=d.get("fonte"), det_ts=d["ts"])
            d["_usada"] = True

    # Fase 2: identidade trocada. Só conta se a pessoa nomeada não estava
    # atravessando dentro desta janela — senão a detecção é dela.
    for d in livres:
        if d["_usada"] or d["person_id"] is None:
            continue
        esperados_na_janela = {p["pid"] for p in passagens
                               if abs(d["ts"] - p["ts"]) <= janela}
        if d["person_id"] in esperados_na_janela:
            continue                      # detecção extra de quem já foi casado
        alvo = candidatas(d, lambda p: True)
        if alvo:
            p = alvo[0]
            p.update(resultado="errada", score=d["score"], visto=d["name"])
            d["_usada"] = True

    # Fase 3: detectada mas não reconhecida, e trilha sem recorte legível.
    for estado, rotulo in (("processado", "desconhecido"),
                           ("descartado", "sem_recorte")):
        for d in livres:
            if d["_usada"] or d["person_id"] is not None or d["status"] != estado:
                continue
            alvo = candidatas(d, lambda p: True)
            if alvo:
                p = alvo[0]
                p.update(resultado=rotulo,
                         score=d["score"] if estado == "processado" else None)
                d["_usada"] = True

    # Sobrou passagem sem detecção nenhuma: nunca foi vista.
    for p in passagens:
        if p["resultado"] is None:
            p["resultado"] = "nao_detectada"

    return sum(1 for d in livres if not d["_usada"])


def ic95(acertos: int, total: int) -> tuple[float, float, float]:
    """Proporção e intervalo de confiança de 95% (aproximação normal).

    Existe para o relatório não sugerir precisão que a amostra não tem: 8 de 10
    passagens não é "80%", é "algo entre 55% e 100%".
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = acertos / total
    h = 1.96 * (p * (1 - p) / total) ** 0.5
    return p, max(0.0, p - h), min(1.0, p + h)


def _barra(n: int, total: int, largura: int = 28) -> str:
    if total == 0:
        return ""
    return "#" * max(1, round(n / total * largura)) if n else ""


# --------------------------------------------------------------------------- #
# Modo teste controlado
# --------------------------------------------------------------------------- #
def medir_passagens(db: Database, cfg, caminho: Path, janela: float,
                    base_dia: str | None) -> int:
    passagens = ler_passagens(caminho, base_dia)
    if not passagens:
        print("Nenhuma passagem lida do arquivo.")
        return 1

    limiar = float(cfg.recognition.cosine_threshold)
    pessoas = {p["name"].strip().lower(): p["id"] for p in db.list_people()}

    inicio = min(p["ts"] for p in passagens) - janela - 1
    fim = max(p["ts"] for p in passagens) + janela + 1
    deteccoes = db.detections_between(inicio, fim)

    print(f"Passagens registradas: {len(passagens)}")
    print(f"Detecções no período:  {len(deteccoes)}")
    print(f"Janela de casamento:   ±{janela:.0f}s | limiar atual {limiar:.3f}")

    # Trilha pendente = capturada e AINDA NÃO reconhecida. Sem esta checagem,
    # ela não casa com nenhum degrau do funil e a passagem cai em "nunca
    # detectada" — reportando como falha de detecção o que é só lote não
    # executado. Aborta em vez de avisar: um relatório inteiro errado é pior
    # que nenhum relatório.
    pendentes = [d for d in deteccoes if d["status"] == "pendente"]
    if pendentes:
        print(f"\n⚠ {len(pendentes)} trilha(s) no período ainda NÃO foram "
              "reconhecidas.")
        print("  No modo captura o reconhecimento roda depois, em lote. Sem")
        print("  isso, estas passagens apareceriam como 'nunca detectada' —")
        print("  culpando a câmera por trabalho que não foi feito.")
        print("\n  Rode primeiro:")
        print("    python scripts/recognize_batch.py")
        print("\n  E depois repita esta medição.")
        return 1

    # O cooldown existe para não gravar 200 eventos da mesma criança, e é certo
    # para a chamada. Para medir POR PASSAGEM ele destrói dado: a mesma pessoa
    # detectada de novo dentro da janela não gera evento, e todos os
    # desconhecidos dividem a mesma chave — vários não reconhecidos juntos
    # viram um evento só. O resto aparece como "nunca detectada".
    cooldown = float((cfg.get("worker") or {}).get("event_cooldown_seconds") or 0)
    if cooldown > 0 and any(d["fonte"] == "evento" for d in deteccoes):
        proximas = _passagens_proximas(passagens, cooldown)
        if proximas:
            print(f"\n⚠ worker.event_cooldown_seconds = {cooldown:.0f}s, e "
                  f"{proximas} passagem(ns) ocorreram a menos que isso da")
            print("  anterior da MESMA pessoa. Esses eventos foram suprimidos")
            print("  na gravação e vão aparecer como 'nunca detectada'.")
            print("\n  Para medir por passagem, use uma das opções:")
            print("    • worker.event_cooldown_seconds: 0  (e reinicie o worker)")
            print("    • modo captura, que grava uma trilha por travessia")
            print("      (é também o modo que roda na escola)")
            print("\n  Continuando, mas os números abaixo subestimam o recall.")

    nao_cadastrados = sorted({p["nome"] for p in passagens
                              if p["nome"].strip().lower() not in pessoas})
    if nao_cadastrados:
        print(f"\n⚠ Sem cadastro no banco: {', '.join(nao_cadastrados)}")
        print("  Essas passagens contam como falha — o sistema não teria como")
        print("  acertar. Se foi engano no nome, corrija o CSV e rode de novo.")

    sobraram = atribuir(passagens, deteccoes, pessoas, janela)
    if sobraram:
        print(f"\n{sobraram} detecção(ões) sem passagem correspondente.")
        print("  Normalmente é trilha quebrada — a mesma travessia rastreada")
        print("  duas vezes. Não afeta a chamada (agrupa por pessoa), mas se")
        print("  for muito, ou a janela está curta, ou faltou anotar passagem.")

    _relatorio_funil(passagens, limiar)
    _relatorio_ambiguos(passagens, cfg.api.host if cfg.api.host != "0.0.0.0"
                        else "IP_DO_PI", int(cfg.api.port))
    _relatorio_por_dia(passagens)
    _relatorio_por_rodada(passagens)
    _relatorio_falhas(passagens, limiar)
    return 0


def _relatorio_funil(passagens: list[dict], limiar: float) -> None:
    total = len(passagens)
    contagem = Counter(p["resultado"] for p in passagens)

    print(f"\n--- Funil, por PASSAGEM ({total} travessias) ---")
    for chave, rotulo in FUNIL:
        n = contagem.get(chave, 0)
        pct = n / total * 100
        print(f"  {rotulo:<45} {n:4d}  {pct:5.1f}%  {_barra(n, total)}")

    acertos = contagem.get("acerto", 0)
    ambiguos = sum(1 for p in passagens
                   if p["resultado"] == "acerto" and p["ambiguo"])
    p, lo, hi = ic95(acertos, total)
    print(f"\n  Recall por passagem: {p*100:.1f}%"
          f"  (95%: {lo*100:.0f}% a {hi*100:.0f}%)")

    if ambiguos:
        pior = (acertos - ambiguos) / total
        print(f"\n  ⚠ {ambiguos} de {acertos} acerto(s) não são atribuíveis com")
        print("    certeza: outra pessoa atravessou no mesmo instante (ou mais")
        print("    perto), então a detecção pode ser o acerto de quem foi")
        print("    nomeado ou o erro de quem passou junto. Hora anotada à mão")
        print("    não separa os dois casos.")

        if ambiguos == acertos:
            print("\n    Com TODOS ambíguos, o recall por passagem não é")
            print("    mensurável neste registro — a faixa iria de 0% ao valor")
            print("    acima, o que não informa nada.")
        else:
            print(f"\n    Recall por passagem: entre {pior*100:.1f}% e {p*100:.1f}%.")

        print("\n    É limite do método, não do sistema: em travessia")
        print("    simultânea a atribuição por horário é indecidível. O que")
        print("    continua válido aqui são os degraus de DETECÇÃO (nunca")
        print("    detectada, sem recorte, não reconhecida) — esses não")
        print("    dependem de saber de quem é o rosto.")
        print("\n    Para medir identificação por passagem:")
        print("      • rodada UM POR VEZ, ~5s de intervalo, onde o horário")
        print("        decide sozinho. É a rodada que dá o número;")
        print("      • ou conferir os recortes listados abaixo, que é a única")
        print("        fonte que resolve de verdade de quem é o rosto.")

    if hi - lo > 0.20 and ambiguos < acertos:
        print("  ⚠ Intervalo largo — amostra pequena para concluir. Umas 30")
        print("    passagens por rodada dão a casa dos 10%; 100 dão ±8pp.")


def _relatorio_ambiguos(passagens: list[dict], ip: str, porta: int) -> None:
    """Lista os acertos indecidíveis com o recorte, para conferência visual.

    É a mesma saída do `--review` da calibração: o dado que falta não está no
    banco, está na foto. Só quem olha o recorte sabe de quem é o rosto.
    """
    casos = [p for p in passagens
             if p["resultado"] == "acerto" and p["ambiguo"]]
    if not casos:
        return

    print(f"\n--- Acertos a conferir na foto ({len(casos)}) ---")
    print("Abra cada recorte e veja se o rosto é de quem o sistema disse.")
    for p in casos[:20]:
        hora = time.strftime("%H:%M:%S", time.localtime(p["det_ts"]))
        junto = sorted({q["nome"] for q in passagens
                        if q is not p and abs(q["ts"] - p["ts"]) <= 2
                        and q["nome"] != p["nome"]})
        crop = p.get("crop")
        # As duas origens ficam em bases diferentes e a API tem uma rota para
        # cada. Montar /snapshots/ para tudo dava 404 em todo recorte vindo do
        # modo captura — ou seja, em tudo que interessa na escola.
        base = "/snapshots/" if p.get("fonte") == "evento" else "/tracks/"
        url = (f"http://{ip}:{porta}{base}{crop}" if crop else "(sem recorte)")
        print(f"  {hora}  disse {p['nome']} ({p['score']:.3f})")
        if junto:
            print(f"            passaram junto: {', '.join(junto)}")
        print(f"            {url}")
    if len(casos) > 20:
        print(f"  … e mais {len(casos) - 20}.")


def _relatorio_por_dia(passagens: list[dict]) -> None:
    """A métrica que decide o produto: a criança foi pega ao menos uma vez?"""
    por_pessoa_dia = defaultdict(list)
    for p in passagens:
        dia = time.strftime("%Y-%m-%d", time.localtime(p["ts"]))
        por_pessoa_dia[(dia, p["nome"])].append(p["resultado"])

    ok = sum(1 for r in por_pessoa_dia.values() if "acerto" in r)
    # Firme = a pessoa teve ao menos um acerto que o horário atribui sem
    # contestação. O outro número herda a ambiguidade do funil: identidade
    # produzida a partir do rosto de outra pessoa aparece aqui como sucesso.
    firmes = defaultdict(list)
    for p_ in passagens:
        dia = time.strftime("%Y-%m-%d", time.localtime(p_["ts"]))
        if p_["resultado"] == "acerto" and not p_["ambiguo"]:
            firmes[(dia, p_["nome"])].append(True)
    ok_firme = sum(1 for k in por_pessoa_dia if firmes.get(k))

    total = len(por_pessoa_dia)
    p, lo, hi = ic95(ok, total)

    print(f"\n--- Por DIA e pessoa ({total} combinações) ---")
    print(f"  Identificada ao menos uma vez: {ok} de {total}")
    print(f"  Recall por dia: {p*100:.1f}%  (95%: {lo*100:.0f}% a {hi*100:.0f}%)")
    print("  ↑ é este que diz se a chamada sai certa: basta acertar uma vez.")
    if ok_firme < ok:
        pf, _, _ = ic95(ok_firme, total)
        print(f"\n  Sem contar os ambíguos: {ok_firme} de {total} ({pf*100:.1f}%).")
        print("  A diferença é gente cujo único acerto veio de detecção que")
        print("  pode ter sido de outra pessoa. Confira os recortes acima.")

    faltaram = sorted(nome for (_, nome), r in por_pessoa_dia.items()
                      if "acerto" not in r)
    if faltaram:
        print(f"  Nunca identificadas no dia: {', '.join(faltaram)}")
        print("  Cada uma dessas é uma falta lançada por engano na chamada.")


def _relatorio_por_rodada(passagens: list[dict]) -> None:
    rodadas = sorted({p["rodada"] for p in passagens})
    if len(rodadas) < 2:
        return

    print("\n--- Por rodada ---")
    resultados = {}
    for r in rodadas:
        grupo = [p for p in passagens if p["rodada"] == r]
        ok = sum(1 for p in grupo if p["resultado"] == "acerto")
        prop, lo, hi = ic95(ok, len(grupo))
        resultados[r] = prop
        print(f"  {r:<14} {ok:3d}/{len(grupo):<3d}  {prop*100:5.1f}%"
              f"  (95%: {lo*100:.0f}%–{hi*100:.0f}%)")

    melhor, pior = max(resultados, key=resultados.get), min(resultados, key=resultados.get)
    dif = resultados[melhor] - resultados[pior]
    if dif >= 0.15:
        print(f"\n  Diferença de {dif*100:.0f}pp entre '{melhor}' e '{pior}'.")
        print("  Queda grande com aglomeração significa que o gargalo é")
        print("  enquadramento e oclusão, não cadastro — mexer em amostras")
        print("  não vai resolver. Atacar posição da câmera e iluminação.")
    elif resultados[pior] < 0.7:
        print("\n  As rodadas vão parecido e ambas baixas: o gargalo está no")
        print("  reconhecimento em si, não na aglomeração. Atacar cadastro.")


def _relatorio_falhas(passagens: list[dict], limiar: float) -> None:
    falhas = [p for p in passagens if p["resultado"] != "acerto"]
    if not falhas:
        print("\nNenhuma falha. Confira se a janela e o registro estão certos —")
        print("100% em teste controlado é raro o suficiente para desconfiar.")
        return

    print(f"\n--- Falhas em detalhe ({len(falhas)}) ---")

    # Entre as não reconhecidas, separa quem chegou perto do limiar de quem não
    # chegou nem perto: a primeira melhora com limiar ou amostra, a segunda não.
    quase = [p for p in falhas if p["resultado"] == "desconhecido"
             and (p["score"] or 0) >= limiar - 0.08]
    longe = [p for p in falhas if p["resultado"] == "desconhecido"
             and (p["score"] or 0) < limiar - 0.08]
    if quase:
        scores = ", ".join(f"{p['score']:.3f}" for p in quase[:8])
        print(f"  {len(quase)} não reconhecida(s) por pouco (a <0.08 do limiar):"
              f" {scores}")
        print("    Faixa recuperável: mais amostras da pessoa devem resolver.")
    if longe:
        piores = sorted(longe, key=lambda p: p["score"] or 0)[:5]
        scores = ", ".join(f"{p['score']:.3f}" for p in piores)
        print(f"  {len(longe)} muito abaixo do limiar: {scores}")
        print("    Score assim é rosto irreconhecível (borrão, perfil, escuro)")
        print("    ou pessoa sem cadastro — limiar não conserta.")

    trocas = [p for p in falhas if p["resultado"] == "errada"]
    if trocas:
        print(f"  {len(trocas)} confundida(s) com outra pessoa:")
        for p in trocas[:8]:
            hora = time.strftime("%H:%M:%S", time.localtime(p["ts"]))
            print(f"    {hora}  {p['nome']} → identificado como "
                  f"{p['visto']} ({p['score']:.3f})")
        print("    Conta duas vezes: falso negativo de quem passou e falso")
        print("    positivo de quem foi nomeado.")

    print("\n  Onde atacar, do degrau mais frequente:")
    contagem = Counter(p["resultado"] for p in falhas)
    for chave, _ in FUNIL:
        if contagem.get(chave) and chave in DIAGNOSTICO:
            print(f"    {contagem[chave]:3d}× {chave:<14} → {DIAGNOSTICO[chave]}")


# --------------------------------------------------------------------------- #
# Modo chamada (operação real)
# --------------------------------------------------------------------------- #
def medir_chamada(db: Database) -> int:
    """Recall a partir das chamadas conferidas e fechadas.

    Aqui a lista externa é a conferência humana: aluno marcado `presente` à mão
    é alguém que estava lá. Se não houve detecção dele naquele dia, o sistema
    perdeu — falso negativo. Só conta dia FECHADO, pela mesma razão da
    calibração: em chamada aberta, ausência de correção é "ninguém olhou".
    """
    fechados = db.closed_days()
    if not fechados:
        print("Nenhuma chamada fechada ainda.")
        print()
        print("O recall em operação sai da conferência: no painel, aba Chamada,")
        print("marque quem faltou ou quem o sistema não pegou e feche o dia.")
        print("Cada dia fechado passa a alimentar esta medição e a calibração.")
        return 1

    overrides = db.all_overrides()
    detectados = defaultdict(set)
    for d in db.list_detections(limit=100000):
        if d["person_id"]:
            dia = time.strftime("%Y-%m-%d", time.localtime(d["ts"]))
            detectados[dia].add(int(d["person_id"]))

    nomes = {p["id"]: p["name"] for p in db.list_people()}
    linhas, fn_total, presentes_total = [], 0, 0

    for dia in sorted(fechados):
        auto = detectados.get(dia, set())
        # Presentes de verdade = detectados e não desmarcados, mais os marcados
        # presente à mão.
        marcados_presente = {pid for (d, pid), v in overrides.items()
                             if d == dia and v == 1}
        marcados_ausente = {pid for (d, pid), v in overrides.items()
                            if d == dia and v == 0}
        presentes = (auto - marcados_ausente) | marcados_presente
        # Falso negativo: estava presente, o sistema não pegou.
        fn = {pid for pid in marcados_presente if pid not in auto}

        if not presentes:
            continue
        linhas.append((dia, len(presentes), len(fn),
                       sorted(nomes.get(p, f"#{p}") for p in fn)))
        fn_total += len(fn)
        presentes_total += len(presentes)

    if not presentes_total:
        print("Chamadas fechadas, mas sem ninguém presente registrado.")
        return 1

    print(f"Chamadas conferidas e fechadas: {len(linhas)}")
    print(f"\n{'dia':<12} {'presentes':>10} {'perdidos':>9}  quem o sistema perdeu")
    for dia, n_pres, n_fn, quem in linhas:
        lista = ", ".join(quem[:4]) + (" …" if len(quem) > 4 else "")
        print(f"{dia:<12} {n_pres:>10} {n_fn:>9}  {lista}")

    ok = presentes_total - fn_total
    p, lo, hi = ic95(ok, presentes_total)
    print(f"\n--- Recall por dia e aluno ---")
    print(f"  {ok} de {presentes_total} identificados pelo sistema")
    print(f"  Recall: {p*100:.1f}%  (95%: {lo*100:.0f}% a {hi*100:.0f}%)")

    if hi - lo > 0.20:
        print("  ⚠ Poucos dias fechados para concluir. Acumule mais conferências.")
    if fn_total:
        perdidos = Counter(nome for *_, quem in linhas for nome in quem)
        repetem = [n for n, c in perdidos.items() if c > 1]
        if repetem:
            print(f"\n  Perdidos em mais de um dia: {', '.join(repetem)}")
            print("  Falha repetida na mesma pessoa é problema de cadastro dela,")
            print("  não do sistema — vale recadastrar com mais ângulos.")
    print("\n  Esta medição não distingue POR QUE cada um foi perdido.")
    print("  Para isso, o teste controlado: --passagens registro.csv")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Mede o recall do reconhecimento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  %(prog)s --chamada
  %(prog)s --passagens registro.csv
  %(prog)s --passagens registro.csv --janela 15 --data 2026-08-28
""")
    ap.add_argument("--passagens", metavar="CSV",
                    help="registro do teste controlado (nome;hora;rodada)")
    ap.add_argument("--chamada", action="store_true",
                    help="mede a partir das chamadas conferidas e fechadas")
    ap.add_argument("--janela", type=float, default=10.0,
                    help="tolerância em segundos entre hora anotada e detecção "
                         "(padrão: 10)")
    ap.add_argument("--data", metavar="AAAA-MM-DD",
                    help="dia das passagens, quando o CSV só tem a hora "
                         "(padrão: hoje)")
    args = ap.parse_args(argv)

    if not args.passagens and not args.chamada:
        ap.error("escolha --passagens CSV ou --chamada")

    cfg = load_config()
    db = Database(cfg.storage.db_path)

    if args.chamada:
        return medir_chamada(db)

    caminho = Path(args.passagens)
    if not caminho.exists():
        print(f"Arquivo não encontrado: {caminho}")
        return 1
    return medir_passagens(db, cfg, caminho, args.janela, args.data)


if __name__ == "__main__":
    raise SystemExit(main())
