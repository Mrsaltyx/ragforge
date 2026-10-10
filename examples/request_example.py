"""Requêtes prêtes à l'emploi contre l'API locale.

Lancer l'API d'abord :  uvicorn ragforge.api.main:app --reload
Puis :                 python examples/request_example.py
"""

import httpx

BASE = "http://localhost:8000"


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=60) as client:
        print("== /health ==")
        print(client.get("/health").json())

        print("\n== /query ==")
        resp = client.post("/query", json={"question": "Quelle est la formule du RRF ?", "top_k": 3})
        body = resp.json()
        print(body["answer"][:500])
        print(f"\n({len(body['sources'])} sources, citations fidèles : {body['citation_faithful']})")
        for s in body["sources"]:
            print(f"  [{s['rank']}] {s['document']} (score {s['score']})")

        print("\n== /evaluate ==")
        for metric in client.post("/evaluate").json():
            print(f"  {metric['metric']:28s} {metric['value']:.3f}   {metric['detail']}")


if __name__ == "__main__":
    main()
