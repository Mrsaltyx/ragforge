# Le retrieval hybride : BM25, dense et fusion RRF

La recherche dense encode la question et les documents dans un espace vectoriel, puis mesure la similarité de cosinus. Elle excelle sur les reformulations et les questions conceptuelles, mais rate les termes exacts : codes produits, numéros de dossier, noms propres rares.

La recherche lexicale BM25 fait l'inverse : elle compte les occurrences de mots exacts et ignore le sens. Sur un corpus mixte, ni l'une ni l'autre ne suffit. Le retrieval hybride lance les deux en parallèle et fusionne les classements.

La fusion recommandée en production est le Reciprocal Rank Fusion (RRF). Pour chaque document, le score RRF est la somme de 1 / (k + rang) sur chaque liste de résultats, avec k = 60 comme valeur consensuelle. Le RRF ne demande aucune normalisation des scores : il fusionne les rangs, ce qui rend dense et BM25 directement comparables.

Une fois les candidats fusionnés, un cross-encoder peut les re-scorer en lisant la question et le chunk ensemble. Cette étape de reranking coûte de la latence mais améliore fortement la précision du top 5. Elle se garde uniquement si l'évaluation sur un jeu de questions étiquetées le démontre.
