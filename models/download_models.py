"""Baixa os modelos ONNX (YuNet + SFace) do OpenCV Zoo para esta pasta.

Uso:
    python models/download_models.py            # modelos float32 (padrão)
    python models/download_models.py --int8      # versões quantizadas (Pi 3B)
    python models/download_models.py --force     # rebaixa mesmo se já existir

Os modelos ficam no repositório via git-lfs; validamos o arquivo baixado para
não deixar passar um "ponteiro LFS" ou uma página de erro HTML disfarçada de
.onnx — falha silenciosa que depois aparece como um erro obscuro do OpenCV.
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent

# Dois espelhos do mesmo repositório oficial (OpenCV Zoo). O `raw.github...` é o
# endpoint direto; o `github.com/.../raw/...` redireciona para ele. Alguns
# proxies corporativos liberam só um dos dois, então tentamos ambos.
MIRRORS = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models",
    "https://github.com/opencv/opencv_zoo/raw/main/models",
)


def _urls(subdir: str, name: str) -> list[str]:
    return [f"{base}/{subdir}/{name}" for base in MIRRORS]


# nome do arquivo -> (urls, tamanho mínimo plausível em KB)
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        _urls("face_detection_yunet", "face_detection_yunet_2023mar.onnx"), 100),
    "face_recognition_sface_2021dec.onnx": (
        _urls("face_recognition_sface", "face_recognition_sface_2021dec.onnx"), 5000),
}

# Quantizados: ~4x menores e mais rápidos na CPU do Pi 3B.
# Exigem OpenCV >= 4.8 (o install_pi.sh instala 4.9+).
MODELS_INT8 = {
    "face_detection_yunet_2023mar_int8.onnx": (
        _urls("face_detection_yunet", "face_detection_yunet_2023mar_int8.onnx"), 50),
    "face_recognition_sface_2021dec_int8.onnx": (
        _urls("face_recognition_sface", "face_recognition_sface_2021dec_int8.onnx"), 2000),
}



def _progress(block, block_size, total):
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        sys.stdout.write(f"\r   {pct}%")
        sys.stdout.flush()


def _validate(path: Path, min_kb: int) -> str | None:
    """Retorna uma mensagem de erro, ou None se o arquivo parece um ONNX válido."""
    if not path.exists():
        return "arquivo não foi criado"
    size_kb = path.stat().st_size // 1024
    head = path.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs"):
        return "veio um ponteiro git-lfs, não o modelo"
    if head.lstrip()[:1] in (b"<", b"{"):
        return "veio HTML/JSON (URL errada ou bloqueada), não um .onnx"
    if size_kb < min_kb:
        return f"tamanho suspeito: {size_kb} KB (esperado >= {min_kb} KB)"
    return None


def download(name: str, urls: list[str], min_kb: int, force: bool = False) -> bool:
    dest = DEST / name
    if dest.exists() and not force and _validate(dest, min_kb) is None:
        print(f"[ok] {name} já existe ({dest.stat().st_size // 1024} KB)")
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    problems = []
    for url in urls:
        print(f"[baixando] {name}")
        try:
            urllib.request.urlretrieve(url, tmp, _progress)
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            problems.append(f"{url} -> {exc}")
            print(f"\r[falhou espelho] {exc}")
            continue

        problem = _validate(tmp, min_kb)
        if problem:
            tmp.unlink(missing_ok=True)
            problems.append(f"{url} -> {problem}")
            print(f"\r[falhou espelho] {problem}")
            continue

        tmp.replace(dest)
        print(f"\r[ok] {name} ({dest.stat().st_size // 1024} KB)")
        return True

    print(f"[FALHA] {name}:")
    for p in problems:
        print(f"        {p}")
    print("        Se a rede do Pi bloqueia o GitHub, baixe os .onnx em outra "
          f"máquina e copie para {DEST} (ex.: scp models/*.onnx pi@IP:{DEST}/).")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Baixa YuNet + SFace do OpenCV Zoo.")
    ap.add_argument("--int8", action="store_true",
                    help="baixa também as versões quantizadas (mais rápidas no Pi 3B)")
    ap.add_argument("--force", action="store_true", help="rebaixa mesmo se já existir")
    args = ap.parse_args()

    wanted = dict(MODELS)
    if args.int8:
        wanted.update(MODELS_INT8)

    ok = True
    for name, (urls, min_kb) in wanted.items():
        ok &= download(name, urls, min_kb, force=args.force)

    print("Modelos em:", DEST)
    if not ok:
        print("\nAlgum download falhou. Sem os .onnx o worker não sobe.")
        return 1
    if args.int8:
        print("\nPara usar os quantizados, aponte no config.yaml:")
        print("  models.detector:   models/face_detection_yunet_2023mar_int8.onnx")
        print("  models.recognizer: models/face_recognition_sface_2021dec_int8.onnx")
        print("Recalibre recognition.cosine_threshold depois (o int8 muda o score).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
