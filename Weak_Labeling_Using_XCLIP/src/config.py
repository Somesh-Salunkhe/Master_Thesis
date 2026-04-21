# Configuration

from dataclasses import dataclass
from pathlib import Path
import yaml

def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(path: str | Path) -> dict:
    cfg = load_yaml(path)
    base = cfg.get("base_config", None)
    if base:
        base_cfg = load_yaml(base)
        cfg = _deep_update(base_cfg, cfg)
    return cfg

def resolve_project_path(project_root: Path, path_str: str) -> Path:
    p = path_str
    if isinstance(p, str):
        if p.startswith("./"):
            p = p[2:]          
        elif p.startswith("."):
            p = p[1:]          
        if p.startswith("/"):
            p = p[1:]         
    return (project_root / p).resolve()
