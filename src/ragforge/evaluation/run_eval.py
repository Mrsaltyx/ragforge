"""Évaluation sur golden set — les métriques avant toute optimisation.

Métriques :
- recall@k     : le chunk pertinent est-il dans le top-k fusionné ?
- mrr          : rang du premier chunk pertinent (plus il est haut, mieux c'est)
- faithfulness : les citations de la réponse renvoient-elles à des chunks récupérés ?
- answer_hit   : la réponse contient-elle au moins un mot-clé attendu ?

Usage : python -m ragforge.evaluation.run_eval   (ou ragforge-eval)
"""

import json
from pathlib import Path

from ..pipeline import RAGPipeline

GOLDEN_SET = Path(__file__).resolve().parents[3] / "eval" / "golden_set.json"


def run_evaluation(pipeline: RAGPipeline, golden_path: Path = GOLDEN_SET) -> dict:
    cases = json.loads(golden_path.read_text(encoding="utf-8"))["cases"]
    top_k = pipeline.settings.final_top_k

    recall_hits, rr_sum, faithful, answer_hits = 0, 0.0, 0, 0
    for case in cases:
        resp = pipeline.query(case["question"], top_k=top_k)
        sources_text = " ".join(s.excerpt.lower() for s in resp.sources)

        hit = any(kw.lower() in sources_text for kw in case["keywords"])
        recall_hits += int(hit)
        if hit:
            rank = next(
                i for i, s in enumerate(resp.sources, start=1) if any(kw.lower() in s.excerpt.lower() for kw in case["keywords"])
            )
            rr_sum += 1.0 / rank
        faithful += int(resp.citation_faithful)
        answer_hits += int(any(kw.lower() in resp.answer.lower() for kw in case["keywords"]))

    n = len(cases)
    return {
        f"recall@{top_k}": {
            "value": round(recall_hits / n, 3),
            "detail": f"{recall_hits}/{n} questions avec un chunk pertinent dans le top-{top_k}",
        },
        "mrr": {
            "value": round(rr_sum / n, 3),
            "detail": "Mean Reciprocal Rank du premier chunk pertinent",
        },
        "citation_faithfulness": {
            "value": round(faithful / n, 3),
            "detail": f"{faithful}/{n} réponses dont toutes les citations sont des chunks récupérés",
        },
        "answer_hit_rate": {
            "value": round(answer_hits / n, 3),
            "detail": f"{answer_hits}/{n} réponses contenant au moins un mot-clé attendu",
        },
    }


def main() -> None:
    from ..bootstrap import build_pipeline

    report = run_evaluation(build_pipeline())
    for metric, r in report.items():
        print(f"{metric:28s} {r['value']:.3f}   {r['detail']}")


if __name__ == "__main__":
    main()
