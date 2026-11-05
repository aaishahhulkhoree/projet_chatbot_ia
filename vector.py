from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import unicodedata
import re

CSV_PATH = "data/FactRecruteuse.csv"

"""Supprime les accents d'une chaîne de caractères."""
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) 
        if unicodedata.category(c) != "Mn"
    )

"""Normalise un nom de colonne : minuscules, espaces compressés."""
def canonize(col: str) -> str:
    col = str(col).replace("\ufeff", "").strip().strip('"').strip("'").strip()
    col = strip_accents(col.lower())
    col = re.sub(r"[''`\-]+", " ", col)
    col = re.sub(r"\s+", " ", col).strip()
    col = col.replace("*", "")
    return col


#mapping des colonnes normalisées vers les noms canoniques
CANON_MAP = {
    "mois": "mois",
    "recruteuse": "recruteuse",
    "nb de candidats contactes": "nb_candidats_contactes",
    "nb d entretiens candidats salaries": "nb_entretiens_salaries",
    "nb d entretiens candidats sous traitants": "nb_entretiens_sous_traitants",
    "nb de candidats recrutes salaries": "nb_recrutes_salaries",
    "nb de candidats integres sous traitants": "nb_integres_sous_traitants",
    "nombre de presentations clients": "nb_presentations_clients",
    "nb de refus cdi salaries": "nb_refus_cdi_salaries",
    "nombre de ko candidat a la suite d une presentation client": "nb_ko_candidat_apres_presentation",
    "nombre de ko client a la suite d une presentation client": "nb_ko_client_apres_presentation",
}

"""Trouve le nom canonique correspondant à une colonne normalisée."""
def match_to_canon(normalized_col: str) -> str | None:
    return CANON_MAP.get(normalized_col)

"""Charge et prépare le DataFrame avec les colonnes canoniques."""
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8-sig",
        quotechar='"',
        skipinitialspace=True,
        engine="python",
    )
    
    #mapping des colonnes brutes vers canoniques
    rename_map = {}
    for raw_col in df.columns:
        normalized = canonize(raw_col)
        canon = match_to_canon(normalized)
        if canon:
            rename_map[raw_col] = canon
    
    df.rename(columns=rename_map, inplace=True)
    
    #Vérification des colonnes requises
    required_cols = list(CANON_MAP.values())
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        raise ValueError(
            f"Colonnes manquantes : {', '.join(missing)}\n"
            f"Colonnes présentes : {', '.join(df.columns)}"
        )

    for col in df.columns:
        if col != "mois" and col != "recruteuse":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    return df

"""Crée ou charge le vector store à partir du DataFrame."""
def create_vector_store(df: pd.DataFrame, db_location: str = "./chroma_langchain_db"):
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    should_add_documents = not os.path.exists(db_location)
    
    vector_store = Chroma(
        collection_name="recrutement",
        embedding_function=embeddings,
        persist_directory=db_location,
    )
    
    if should_add_documents:
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            content = (
                f"Mois: {row['mois']}. "
                f"Recruteuse: {row['recruteuse']}. "
                f"Candidats contactés: {row['nb_candidats_contactes']}. "
                f"Entretiens salariés: {row['nb_entretiens_salaries']}. "
                f"Entretiens sous-traitants: {row['nb_entretiens_sous_traitants']}. "
                f"Recrutés salariés: {row['nb_recrutes_salaries']}. "
                f"Intégrés sous-traitants: {row['nb_integres_sous_traitants']}. "
                f"Présentations clients: {row['nb_presentations_clients']}. "
                f"Refus CDI salariés: {row['nb_refus_cdi_salaries']}. "
                f"KO candidat après présentation: {row['nb_ko_candidat_apres_presentation']}. "
                f"KO client après présentation: {row['nb_ko_client_apres_presentation']}."
            )
            
            document = Document(
                page_content=content,
                metadata={
                    "mois": row["mois"],
                    "recruteuse": row["recruteuse"],
                },
            )
            
            documents.append(document)
            ids.append(str(i))
        
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"{len(documents)} documents ajoutés au vector store")
    else:
        print(f"Vector store chargé depuis {db_location}")
    
    return vector_store