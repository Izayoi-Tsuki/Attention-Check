"""读取 config.yaml，路径统一相对项目根目录解析。"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["path"] = str(ROOT / cfg["model"]["path"])
    cfg["storage"]["db_path"] = str(ROOT / cfg["storage"]["db_path"])
    cfg["storage"]["reports_dir"] = str(ROOT / cfg["storage"]["reports_dir"])
    return cfg
