# Assistant IA RAG – Analyse des données de recrutement

Application locale Python permettant d'interroger un fichier de suivi du recrutement en langage naturel avec **Gradio**, **LangChain**, **Ollama** et **ChromaDB**.

> **État du projet :** l'assistant RAG est implémenté. L'intégration du dashboard **Power BI** est envisagée, mais n'est pas encore présente dans le code fourni.

## Fonctionnalités actuelles

- Chargement d'un CSV de suivi des recruteuses.
- Normalisation des noms de colonnes et conversion des indicateurs numériques.
- Indexation vectorielle des lignes avec ChromaDB et `nomic-embed-text`.
- Recherche des quatre documents les plus proches de la question.
- Génération d'une réponse en français avec `qwen2:0.5b` via Ollama.
- Interface Gradio avec exemples de questions et option d'affichage des sources.

**Limite :** la recherche RAG ne garantit pas des totaux ou classements exhaustifs : seuls quatre documents sont transmis au modèle par défaut. Pour des KPI exacts sur toutes les lignes, privilégier des calculs sur le CSV ou le modèle Power BI.

## Structure du projet

```text
projet-recrutement/
├── main.py                 # Application et interface Gradio
├── vector.py               # Préparation CSV et index ChromaDB
├── requirements.txt        # Dépendances Python
├── README.md
├── data/
│   └── FactRecruteuse.csv  # Données source
└── chroma_langchain_db/    # Index créé lors de la première exécution
```

Le fichier `data/FactRecruteuse.csv` n'est pas fourni avec les trois fichiers Python reçus. Le code attend un CSV UTF-8 avec séparateur `;` et les colonnes de recrutement prévues dans `CANON_MAP` de `vector.py`.

## Prérequis

- Python 3.10 ou plus récent (3.11 recommandé).
- Ollama installé et actif : https://ollama.com/download
- Accès au terminal (PowerShell ou terminal VS Code sous Windows).

## Installation (Windows)

Depuis le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install gradio
```

> `gradio` est importé dans `main.py`, mais manque dans le `requirements.txt` fourni. Ajouter `gradio` à ce fichier pour une installation reproductible.

Télécharger les modèles nécessaires :

```powershell
ollama pull qwen2:0.5b
ollama pull nomic-embed-text
```

Vérifier qu'Ollama tourne ; si nécessaire, démarrer `ollama serve` dans un autre terminal (s'il n'est pas déjà lancé).

## Démarrer l'application

```powershell
python main.py
```

Ouvrir **http://127.0.0.1:7860** dans le navigateur. Le premier lancement peut prendre plus de temps pendant la création de l'index.

<img width="1476" height="882" alt="image" src="https://github.com/user-attachments/assets/d1b961f7-3676-4bae-bc1f-9fb68b1d9ca1" />

## Questions possibles

- Combien de candidats ont été contactés en juillet par Ines ?
- Quelle recruteuse a fait le plus d'entretiens en septembre ?

Les réponses sont produites à partir des documents retrouvés, et non d'une agrégation garantie de toute la base.

### Dashboard local

<img width="1143" height="630" alt="Capture d&#39;écran 2025-10-03 000602" src="https://github.com/user-attachments/assets/9a673be7-01d7-4e0a-a5e8-f395893fdea6" />

## Dépannage

| Problème | Vérification |
| --- | --- |
| `ModuleNotFoundError: gradio` | Installer `gradio` et l'ajouter à `requirements.txt`. |
| `FileNotFoundError` sur le CSV | Vérifier `data/FactRecruteuse.csv` et lancer la commande depuis la racine du projet. |
| Erreur de connexion Ollama | Vérifier qu'Ollama est démarré et que les deux modèles sont téléchargés. |
| Résultats incohérents après modification du CSV | L'index existant n'est pas automatiquement actualisé ; le reconstruire après sauvegarde, en supprimant le dossier `chroma_langchain_db` uniquement si l'on souhaite régénérer cet index. |
| Port 7860 occupé | Modifier `server_port` dans `main.py`. |

## Technologies

Python · Gradio · LangChain · Ollama · ChromaDB · Pandas · Power BI (intégration future)
