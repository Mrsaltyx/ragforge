"""Ingestion en ligne de commande : ragforge-ingest <fichier ou dossier> [plus de chemins…]"""

import sys
from pathlib import Path

from .bootstrap import build_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage : ragforge-ingest <fichier_ou_dossier> [autres…]")
        sys.exit(1)

    pipeline = build_pipeline()
    for arg in sys.argv[1:]:
        path = Path(arg)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() in {".md", ".txt", ".pdf"} and f.is_file():
                n = pipeline.ingest_file(f)
                print(f"✓ {f.name} — {n} chunks indexés")


if __name__ == "__main__":
    main()
