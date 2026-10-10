# Le Retrieval-Augmented Generation (RAG)

Le RAG est une architecture qui combine un moteur de recherche avec un modèle de langage (LLM). Au lieu de répondre uniquement à partir de ses connaissances figées à l'entraînement, le LLM s'appuie sur des documents récupérés en temps réel. Cela réduit les hallucinations et permet d'interroger des données privées ou récentes.

Un pipeline RAG minimal comporte trois étapes. Premièrement, le retrieval : la question de l'utilisateur sert à chercher les passages les plus pertinents d'un corpus indexé. Deuxièmement, l'augmentation : ces passages sont injectés dans le prompt du LLM comme contexte. Troisièmement, la génération : le LLM rédige une réponse en citant ses sources.

Le choix de la métrique conditionne tout le reste. Sur une tâche de classification déséquilibrée, une accuracy brute est trompeuse ; de même, en RAG, une réponse fluide peut citer des passages qui n'ont jamais été récupérés. La citation faithfulness mesure précisément si les sources annoncées correspondent aux chunks effectivement retrouvés.
