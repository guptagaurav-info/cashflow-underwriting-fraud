from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def p(rel: str) -> Path:
    return ROOT / rel
