"""Operações sobre a chamada CONGELADA — o JSON que `/attendance` devolve.

Quando alguém fecha a chamada, a resposta inteira é guardada como uma versão
(tabela `attendance_versions`). Três lugares precisam ler ou alterar essas
versões e não podem divergir entre si:

  - a API, que monta a chamada e serve a versão congelada;
  - o banco, que precisa tirar uma pessoa excluída de todas as versões;
  - os scripts de presença e de recall, que leem quem estava lá.

Por isso as regras moram aqui, como funções puras sobre o dicionário — sem
banco, sem arquivo, sem rede. Este módulo não importa nada do projeto, o que
evita importação circular com `core.database`.
"""

# Origens de cada linha, na mesma grafia que a API usa.
MANUAL_PRESENTE = "manual_presente"   # o sistema não pegou, uma pessoa marcou
MANUAL_AUSENTE = "manual_ausente"     # o sistema pegou errado, uma pessoa tirou


def linhas(payload: dict) -> list[dict]:
    """Todas as linhas da chamada, presentes primeiro."""
    return list(payload.get("presentes", [])) + list(payload.get("ausentes", []))


def falta_aluno_id(linha: dict) -> bool:
    """Esta linha é um cadastro incompleto que alguém PODE corrigir?

    Só conta quem tem `person_id` e não foi removido. Linha anonimizada não
    tem ID por definição, e aluno removido do cadastro não tem mais como
    ganhar um — acusar "falta ID" nesses casos mandaria alguém (a cozinha, do
    outro lado da integração) atrás de um problema que ninguém resolve.

    Regra única para `recontar` e para `/attendance/presentes`: as duas
    contagens de `sem_aluno_id` não podem divergir.

    `matricula` entra no OU por causa das chamadas CONGELADAS antes da troca de
    campo: o payload delas gravou `matricula`, e sem isto uma versão antiga
    reabriria acusando todo mundo de cadastro incompleto.
    """
    return (linha.get("person_id") is not None
            and not (linha.get("aluno_id") or linha.get("matricula"))
            and not linha.get("removido"))


def recontar(payload: dict) -> dict:
    """Recalcula os totais a partir das listas. Altera e devolve o payload.

    Os totais são DERIVADOS das listas, nunca contados à parte: assim tirar
    uma pessoa de uma versão congelada não deixa número desencontrado da
    lista que ele resume.
    """
    presentes = payload.get("presentes", [])
    todas = linhas(payload)
    payload["total_cadastrados"] = len(todas)
    payload["total_presentes"] = len(presentes)
    payload["sem_aluno_id"] = sum(1 for l in todas if falta_aluno_id(l))
    corr = payload.setdefault("correcoes", {})
    corr["marcados_presentes"] = sum(1 for l in todas
                                     if l.get("origem") == MANUAL_PRESENTE)
    corr["marcados_ausentes"] = sum(1 for l in todas
                                    if l.get("origem") == MANUAL_AUSENTE)
    corr["total"] = sum(1 for l in todas if l.get("correcao"))
    return payload


def tirar_fotos(payload: dict, person_id: int) -> bool:
    """Aluno removido do cadastro mantendo o histórico: só a foto sai.

    A linha continua inteira — nome, `aluno_id`, presente ou ausente — porque
    a chamada é registro escolar. O que identifica pelo ROSTO (o link para a
    foto) é o que sai, junto com a imagem no disco. A linha ganha
    `removido: true` para a tela e quem consome saberem por que não há foto.
    True se mudou alguma coisa.
    """
    mudou = False
    for l in linhas(payload):
        if l.get("person_id") != person_id:
            continue
        mudou = True
        l.pop("foto_url", None)
        l.pop("foto_score", None)
        l["removido"] = True
    if mudou:
        recontar(payload)
    return mudou


def retirar_pessoa(payload: dict, person_id: int, anonimizar: bool,
                   nome_anonimo: str) -> bool:
    """Tira a pessoa da chamada, ou a anonimiza. True se mudou alguma coisa.

    Para os modos de exclusão "apagar" e "anonimizar" — o pedido de exclusão
    completa vale também para o registro congelado, e é a única alteração
    que uma versão fechada aceita sem ser reaberta. O modo padrão, que mantém
    o nome no histórico, usa `tirar_fotos`.

      anonimizar=False -> a linha some e os totais são recalculados;
      anonimizar=True  -> a linha fica, para a contagem do dia continuar
                          certa, mas sem nada que identifique: sem id, sem
                          ID de aluno, sem foto e sem o motivo da correção
                          (texto livre, que pode falar da criança).
    """
    mudou = False
    for chave in ("presentes", "ausentes"):
        nova = []
        for l in payload.get(chave, []):
            if l.get("person_id") != person_id:
                nova.append(l)
                continue
            mudou = True
            if anonimizar:
                l = {k: v for k, v in l.items()
                     if k not in ("foto_url", "foto_score")}
                # `matricula` também, e não só `aluno_id`: versão congelada
                # antes da troca de campo gravou o identificador ali, e
                # anonimizar deixando-o para trás seria anonimizar no nome só.
                l.update(person_id=None, aluno_id=None, matricula=None,
                         nome=nome_anonimo)
                if l.get("correcao"):
                    l["correcao"] = {**l["correcao"], "motivo": None}
                nova.append(l)
        payload[chave] = nova
    if mudou:
        recontar(payload)
    return mudou


def ids_detectados(payload: dict) -> set:
    """Quem o RECONHECIMENTO viu no dia, antes de qualquer correção."""
    return {l["person_id"] for l in linhas(payload)
            if l.get("detectado_pelo_sistema") and l.get("person_id") is not None}


def ids_por_origem(payload: dict, origem: str) -> set:
    return {l["person_id"] for l in linhas(payload)
            if l.get("origem") == origem and l.get("person_id") is not None}
