from ast import Dict
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path:str) -> Dict[str, Any]:
    config = Path(config_path).resolve()
    
    if not config.exists():
        raise FileNotFoundError(f"Config .yaml not found in {config}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)