# Chunking et évaluation d'un pipeline RAG

Découper les documents en chunks est la première étape de l'indexation. Une coupure à taille fixe tronque les phrases et dégrade le retrieval : le chunk perd son sens. La pratique recommandée est un découpage structure-aware qui empile des phrases entières jusqu'à une taille cible, avec un recouvrement entre chunks successifs pour préserver le contexte aux frontières.

Pourquoi le recouvrement ? Une phrase en début de chunk peut dépendre de la précédente, coupée dans le chunk d'avant. Reprendre les cent derniers caractères du chunk précédent garantit que chaque phrase garde son contexte immédiat, au prix d'un index légèrement plus volumineux.

L'évaluation est l'étape que la plupart des équipes sautent — et la première à regretter. Sans jeu de questions étiquetées (golden set), impossible de dire si une modification améliore ou dégrade le pipeline. Les métriques minimales sont le recall@k et le MRR pour le retrieval, et la citation faithfulness pour la génération : les sources citées dans la réponse doivent correspondre exactement aux chunks récupérés, sinon la citation est hallucinée.

Chaque optimisation — chunking, hybride, reranking — doit être validée sur le golden set avant d'être adoptée. Une métrique sans protocole reproductible est une opinion, pas un résultat.
