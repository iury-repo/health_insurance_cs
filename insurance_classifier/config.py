from pathlib import Path

import yaml


def project_root(marker="pyproject.toml"):
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found.")

ROOT_DIR = project_root()

def load_yaml_config(config_file: str) -> dict:
    config_path = ROOT_DIR / "config" / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config .yaml not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)