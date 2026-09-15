"""Leitura e escrita de imagem que funcionam com caminho não-ASCII.

O `cv2.imread` e o `cv2.imwrite` do OpenCV entregam o caminho a um
`std::ifstream`/`ofstream`, que no Windows converte a string pela página de
código ANSI local. Caminho com acento não sobrevive à conversão — e o pior é
o modo de falhar:

    cv2.imwrite(caminho_com_acento, img)   -> devolve False, não levanta nada
    cv2.imread(caminho_com_acento)         -> devolve None, não levanta nada

Ou seja: **falha silenciosa**. Um usuário do Windows chamado, por exemplo,
"BeatrizEduão-TI" põe um "ã" em todo caminho absoluto do projeto, e o sistema
passa a rodar sem gravar foto nenhuma, sem uma única mensagem de erro. Foi
exatamente o que aconteceu: o worker publicava o frame (porque já usava
`imencode` + escrita pelo Python, herança do conserto do bug do `.tmp`), a API
tentava lê-lo com `imread`, recebia None, e respondia "ainda sem imagem" —
mandando investigar câmera quando o problema era codificação de caminho.

A solução é fazer o OpenCV trabalhar só com bytes em memória
(`imencode`/`imdecode`) e deixar o acesso ao arquivo para o Python, que lida
com Unicode corretamente.

`escrever` também é ATÔMICA: codifica, grava num temporário ao lado e faz
`os.replace`. Sem isso, quem estiver lendo o arquivo no meio da escrita pega
um JPEG truncado — era o bug do preview servido pela API.

    ATENÇÃO: a extensão do caminho é o que define o formato, tanto aqui quanto
    no OpenCV. O temporário guarda a extensão original e ganha só um sufixo,
    porque nomear o temporário como ".tmp" fazia o `imencode` receber uma
    extensão desconhecida e falhar.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np


def ler(caminho, flags: int = cv2.IMREAD_COLOR):
    """Como `cv2.imread`, mas seguro com caminho não-ASCII.

    Devolve None quando o arquivo não existe ou não é imagem decodificável —
    mesmo contrato do `imread`, para não obrigar quem chama a mudar.
    """
    caminho = Path(caminho)
    try:
        dados = caminho.read_bytes()
    except OSError:
        return None
    if not dados:
        return None
    buf = np.frombuffer(dados, dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _substituir_com_retry(tmp: Path, destino: Path, tentativas: int = 12,
                          espera: float = 0.02) -> bool:
    """`os.replace` resistente ao bloqueio de arquivo aberto do Windows.

    No POSIX, renomear por cima de um arquivo que outro processo tem aberto é
    permitido — o leitor continua com o inode antigo. No Windows isso é negado:

        PermissionError: [WinError 5] Acesso negado

    E acontece de verdade neste projeto: o worker reescreve `facial-frame.jpg`
    várias vezes por segundo enquanto a API o lê a cada 0,8s para o preview do
    cadastro. Quando as duas operações se cruzam, a troca falha — e como a
    publicação do frame ficava fora do `try` do laço, o worker MORRIA no meio
    do reconhecimento.

    O leitor segura o arquivo por microssegundos, então repetir resolve. Doze
    tentativas com 20 ms cobrem ~240 ms, muito acima da janela real. Falhando
    todas, devolve False: perder um frame de preview é irrelevante, derrubar o
    worker não.
    """
    for i in range(tentativas):
        try:
            os.replace(tmp, destino)
            return True
        except PermissionError:
            if i == tentativas - 1:
                return False
            time.sleep(espera)
        except OSError:
            return False
    return False


def escrever(caminho, imagem, params=None) -> bool:
    """Como `cv2.imwrite`, mas segura com caminho não-ASCII e ATÔMICA.

    `params` é a lista de flags do OpenCV, ex.: [cv2.IMWRITE_JPEG_QUALITY, 80].
    Devolve True/False como o `imwrite`.
    """
    caminho = Path(caminho)
    ext = caminho.suffix or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, imagem, params or [])
    except cv2.error:
        # Imagem vazia ou com formato inesperado. Quem chama trata pelo False;
        # deixar escapar daqui já derrubou o worker uma vez.
        return False
    if not ok:
        return False

    # O temporário mantém a extensão para o caso de alguém inspecioná-lo, e
    # fica no MESMO diretório: os.replace só é atômico dentro do mesmo volume.
    tmp = caminho.with_name(f".{caminho.name}.parcial{ext}")
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(buf.tobytes())
        if not _substituir_com_retry(tmp, caminho):
            tmp.unlink(missing_ok=True)
            return False
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
