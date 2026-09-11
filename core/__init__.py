"""Pacote `core` — e um ajuste de saída que precisa acontecer antes de tudo.

Por que há código num __init__.py, que normalmente fica vazio
---------------------------------------------------------------
No Windows, `print()` codifica pela página de código do console — cp1252 no
Brasil. Letra acentuada passa (é, ã, ç estão no cp1252), mas símbolo
tipográfico não: `⚠`, `→`, `•`, `↑` levantam

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u26a0'

e o processo MORRE na linha do print. Não é saída feia — é queda.

Aconteceu de verdade: o `cleanup_snapshots.py` quebrava ao avisar sobre trilha
pendente, justamente na mensagem que existe para proteger dado não processado.
E o projeto usa esses símbolos em ~35 lugares, nos relatórios de calibração,
de recall e de retenção.

Reconfigurar a saída para UTF-8 resolve os 35 de uma vez, e o
`errors="replace"` garante que um console antigo (que não renderize o glifo)
mostre `?` em vez de derrubar o processo. No Linux não muda nada: a saída já é
UTF-8.

O lugar é aqui porque TODO ponto de entrada do projeto — worker, api, painel e
os dez scripts — importa algo de `core` antes de imprimir qualquer coisa. Pôr
a chamada em cada `main()` daria dez oportunidades de esquecer, e o esquecimento
só apareceria no dia em que aquele script tivesse algo a avisar.

O efeito colateral é global e vale saber: qualquer processo que importe `core`
passa a ter stdout/stderr em UTF-8.
"""
import sys


def _saida_utf8():
    for stream in (sys.stdout, sys.stderr):
        # `reconfigure` existe em Python >= 3.7 e só em TextIOWrapper. Sob
        # pytest, notebooks ou redirecionamentos o stream pode ser outro tipo,
        # daí o getattr em vez de chamar direto.
        recfg = getattr(stream, "reconfigure", None)
        if recfg is None:
            continue
        try:
            recfg(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream já detached ou não reconfigurável: seguir sem isso é
            # melhor que impedir o import do pacote.
            pass


_saida_utf8()
