"""Download public datasets with checksum verification."""
import hashlib
import urllib.request

from .config import load_config, p


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(spec: dict) -> None:
    path = p(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[ingest] downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], path)
    d = sha256(path)
    if d != spec["sha256"]:
        raise ValueError(f"[ingest] checksum mismatch for {path.name}: {d}")
    print(f"[ingest] {path.name} sha256 ok")


def run(cfg=None):
    cfg = cfg or load_config()
    for k in ("txn_categories", "paysim"):
        fetch(cfg["data"][k])


if __name__ == "__main__":
    run()
