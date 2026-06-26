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


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Carrega a configuração (com cache quando usa o caminho padrão)."""
    global _cache
    if path is None and _cache is not None:
        return _cache
    resolved = Path(path) if path else _resolve_config_path()
    with open(resolved, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    if path is None:
        _cache = cfg
    return cfg


def project_path(rel: str | os.PathLike) -> Path:
    """Resolve um caminho relativo à raiz do projeto (paths absolutos passam direto)."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
