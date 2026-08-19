"""Carregamento de configuração a partir de YAML.

A configuração é lida (em ordem de prioridade):
  1. caminho passado em load_config(path)
  2. variável de ambiente FACIAL_CONFIG
  3. <raiz>/config.yaml
  4. <raiz>/config.example.yaml  (fallback para facilitar o primeiro teste)

O objeto retornado permite acesso por atributo (cfg.camera.rtsp_url) e por
chave (cfg["camera"]["rtsp_url"]).
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict com acesso por atributo, recursivo."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return Config(value)
        return value


def _resolve_config_path() -> Path:
    env = os.environ.get("FACIAL_CONFIG")
    if env:
        return Path(env)
    local = ROOT / "config.yaml"
    if local.exists():
        return local
    return ROOT / "config.example.yaml"


_cache: Config | None = None


class ConfigError(Exception):
    """Erro de leitura do config.yaml, já traduzido para linguagem humana."""


def _explicar_erro_yaml(resolved: Path, exc: yaml.YAMLError) -> str:
    """Transforma o traceback do PyYAML numa mensagem acionável.

    O erro cru do PyYAML tem dez níveis de pilha e fala de 'block mapping', o
    que não ajuda quem só quer saber qual linha do arquivo está errada.
    """
    linha = coluna = None
    marca = getattr(exc, "problem_mark", None)
    if marca is not None:
        linha, coluna = marca.line + 1, marca.column + 1

    partes = [f"O arquivo de configuração tem erro de sintaxe:\n  {resolved}"]
    if linha:
        partes.append(f"\nO problema está na LINHA {linha}, coluna {coluna}.")
        try:
            linhas = resolved.read_text(encoding="utf-8").splitlines()
            ini, fim = max(0, linha - 4), min(len(linhas), linha + 2)
            trecho = []
            for i in range(ini, fim):
                marcador = ">>" if i + 1 == linha else "  "
                trecho.append(f"  {marcador} {i + 1:>3} | {linhas[i]}")
            partes.append("\n" + "\n".join(trecho))
        except OSError:
            pass
    partes.append(
        "\nCausas mais comuns:\n"
        "  - indentação inconsistente (YAML exige espaços, NUNCA tabulação, e\n"
        "    todas as chaves de um bloco no mesmo recuo);\n"
        "  - linha de exemplo descomentada sem ajustar o recuo ou sem tirar o\n"
        "    comentário que vinha depois do valor;\n"
        "  - dois-pontos dentro de um valor sem aspas.\n"
        "\nValide o arquivo com:\n"
        "  python -c \"import yaml,sys;yaml.safe_load(open(sys.argv[1]));"
        "print('YAML ok')\" config.yaml"
    )
    return "".join(partes)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Carrega a configuração (com cache quando usa o caminho padrão)."""
    global _cache
    if path is None and _cache is not None:
        return _cache
    resolved = Path(path) if path else _resolve_config_path()
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(_explicar_erro_yaml(resolved, exc)) from None
    except OSError as exc:
        raise ConfigError(
            f"Não consegui ler a configuração:\n  {resolved}\n  {exc}\n\n"
            "Se ainda não criou a sua, copie o exemplo:\n"
            "  cp config.pi.example.yaml config.yaml") from None
    if not isinstance(data, dict):
        raise ConfigError(
            f"O arquivo {resolved} não contém um mapeamento YAML "
            f"(veio {type(data).__name__}). Compare com config.pi.example.yaml.")
    cfg = Config(data)
    if path is None:
        _cache = cfg
    return cfg


def load_config_or_exit(path: str | os.PathLike | None = None) -> Config:
    """Como load_config, mas encerra com a mensagem legível em vez de traceback.

    Usado pelos pontos de entrada (worker e scripts): quem digitou errado no
    config.yaml precisa ver a linha do problema, não a pilha do PyYAML.
    """
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"\n{exc}\n") from None


def project_path(rel: str | os.PathLike) -> Path:
    """Resolve um caminho relativo à raiz do projeto (paths absolutos passam direto)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def config_path() -> Path:
    """Caminho do arquivo de configuração em uso (útil para relê-lo)."""
    return _resolve_config_path()


def reload_config() -> Config:
    """Relê a configuração do disco, ignorando o cache.

    Usado pelo worker para detectar troca de modo sem reiniciar. Só o
    `worker.mode` é aplicado a quente — trocar outros parâmetros no meio da
    execução deixaria o processo num estado inconsistente (metade dos valores
    antigos, metade novos), então esses continuam exigindo reinício.
    """
    global _cache
    _cache = None
    return load_config()


DEFAULT_LIVE_PATH = "data/live.jpg"


def live_image_path(cfg) -> Path:
    """Caminho do preview ao vivo (worker escreve, API serve).

    No Raspberry Pi aponte `storage.live_path` para um caminho em tmpfs
    (ex.: /dev/shm/facial-live.jpg): o arquivo é reescrito ~2x por segundo e
    isso desgasta o cartão SD sem necessidade.
    """
    rel = DEFAULT_LIVE_PATH
    storage = cfg.get("storage") or {}
    if isinstance(storage, dict):
        rel = storage.get("live_path") or DEFAULT_LIVE_PATH
    return project_path(rel)


def frame_image_path(cfg) -> Path:
    """Frame LIMPO (sem anotações) publicado pelo worker, para o cadastro.

    Existe porque um dispositivo V4L2 (webcam USB, câmera CSI) só aceita UM
    processo por vez: com o worker rodando, a API não consegue abrir a câmera
    para o cadastro. Em vez de disputar o dispositivo, ela consome o frame que
    o worker já tem em mãos.
    """
    return live_image_path(cfg).with_name("facial-frame.jpg")
