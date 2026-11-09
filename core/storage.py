"""Gravação de snapshots em data/snapshots/AAAAMMDD/HHMMSS_mmm_<nome>.jpg.

Retorna o caminho relativo à pasta de snapshots (ex.: "20260616/120000_001_joao.jpg"),
que é o que vai para o banco e o que a API usa em /snapshots/{path}.
"""

import time
from pathlib import Path

from .config import project_path
from .imagem import escrever as escrever_imagem


def _safe(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
    return cleaned or "face"


class SnapshotStore:
    def __init__(self, snapshots_dir: str):
        self.base = project_path(snapshots_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, frame, label: str = "face", subdir: str = "") -> str:
        now = time.time()
        prefix = Path(subdir) if subdir else Path(time.strftime("%Y%m%d", time.localtime(now)))
        folder = self.base / prefix
        folder.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%H%M%S", time.localtime(now))
        millis = int((now % 1) * 1000)
        fname = f"{stamp}_{millis:03d}_{_safe(label)}.jpg"
        # escrever_imagem, não cv2.imwrite: com caminho não-ASCII o imwrite
        # devolve False sem levantar nada, e o sistema rodaria sem gravar
        # snapshot nenhum, em silêncio. Agora a falha é audível.
        if not escrever_imagem(folder / fname, frame):
            raise OSError(f"não foi possível gravar o snapshot {folder / fname}")
        return str(prefix / fname)

    def remove_dir(self, subdir: str) -> None:
        """Remove a pasta temporária de uma sessão de cadastro (`enroll/<id>`).

        Só aceita pasta ESTRITAMENTE dentro de `enroll/`. A versão anterior
        aceitava qualquer coisa sob a base de snapshots — inclusive a própria
        base: `enroll/..` resolvia para `data/snapshots` e o rmtree levava
        todas as fotos, e `enroll/../amostras` levava as de cadastro de todo
        mundo. Como o id da sessão vem do cliente, isso estava a uma
        requisição de distância.
        """
        import shutil

        raiz = (self.base / "enroll").resolve()
        target = (self.base / subdir).resolve()
        if target != raiz and target.is_relative_to(raiz) and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
