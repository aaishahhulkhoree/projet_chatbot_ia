from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import load_and_prepare_data, create_vector_store, CSV_PATH
import gradio as gr

def create_rag_chain(model_name: str = "qwen2:0.5b"):
    model = OllamaLLM(model=model_name, temperature=0.2)
    template = """Vous êtes un assistant IA spécialisé dans l'analyse des données de recrutement.

Contexte (données retrouvées dans la base) :
{context}

Question utilisateur : {question}

Instructions :
- Répondez de manière claire, précise et synthétique
- Basez-vous UNIQUEMENT sur les données du contexte fourni
- Si l'information n'est pas dans le contexte, dites-le clairement
- Fournissez des chiffres précis et des comparaisons quand c'est pertinent
- Organisez votre réponse de façon structurée

Réponse :"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    return chain

"""Formate les documents récupérés en contexte pour le LLM"""
def format_context(documents: list) -> str:
    if not documents:
        return "Aucune donnée trouvée dans la base."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Document {i}]\n{doc.page_content}")
    
    return "\n\n".join(context_parts)

"""Formate les sources pour l'affichage à l'utilisateur si demandé"""
def format_sources(documents: list) -> str:
    if not documents:
        return "Aucune source disponible."
    
    sources_text = "**SOURCES UTILISÉES**\n\n"
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        sources_text += f"**Source {i}:**\n"
        sources_text += f"- Mois : {metadata.get('mois', 'N/A')}\n"
        sources_text += f"- Recruteuse : {metadata.get('recruteuse', 'N/A')}\n"
        sources_text += f"- Contenu : {doc.page_content[:100]}...\n\n"
    
    return sources_text

"""Traite la question posée par l'utilisateur et génère une réponse."""
def process_question(question: str, show_sources: bool, retriever, chain):
    if not question.strip():
        return "Veuillez poser une question.", ""
    try:
        documents = retriever.invoke(question)
        if not documents:
            return "Aucune donnée pertinente trouvée pour cette question.", ""
        context = format_context(documents)
        response = chain.invoke({
            "context": context,
            "question": question
        })
        formatted_response = f"**RÉPONSE :**\n\n{response.strip()}"
        sources = ""
        if show_sources:
            sources = format_sources(documents)
        
        return formatted_response, sources
        
    except Exception as e:
        return f"**Erreur** : {str(e)}", ""

"""Initialise le système : charge les données, crée le vector store, configure le retriever et le modèle."""
def initialize_system():
    print("\nInitialisation du système...")
    print("─" * 60)
    
    print("Chargement du fichier CSV...")
    df = load_and_prepare_data(CSV_PATH)
    print(f"CSV chargé : {len(df)} lignes de données")
    
    print("Création/chargement du vector store...")
    vector_store = create_vector_store(df)
    
    print("Configuration du retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    print("Retriever configuré (top 4 résultats)")
    
    print("Chargement du modèle Qwen...")
    chain = create_rag_chain()
    print("Modèle qwen2:0.5b chargé et prêt")
    
    print("─" * 60)
    print("Système initialisé avec succès !\n")
    
    return retriever, chain

"""Crée l'interface Gradio pour l'utilisateur."""
def create_gradio_interface(retriever, chain):
    examples = [
        ["Combien de candidats ont été contactés en Juillet par Ines ?"],
        ["Quelle recruteuse a fait le plus d'entretiens en Septembre ?"]
    ]
    
    def gradio_process(question, show_sources):
        return process_question(question, show_sources, retriever, chain)

    with gr.Blocks(theme=gr.themes.Soft(), title="Assistant RAG - Recrutement") as demo:
        gr.Markdown(
            """
            Assistant RAG - Analyse des Données de Recrutement
            
            Posez vos questions sur les données de recrutement et obtenez des réponses précises basées sur vos données CSV.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Votre question",
                    placeholder="Ex: Combien de candidats ont été contactés en Juillet par Ines ?",
                    lines=2
                )
                
                show_sources_checkbox = gr.Checkbox(
                    label="Afficher les sources utilisées",
                    value=False
                )
                
                submit_btn = gr.Button("Rechercher", variant="primary", size="lg")
                clear_btn = gr.Button("Effacer", size="sm")
        
        with gr.Row():
            with gr.Column():
                response_output = gr.Markdown(label="Réponse")
            
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(label="Sources", visible=True)
        
        gr.Examples(
            examples=examples,
            inputs=question_input,
            label="Cliquez sur un exemple pour l'essayer"
        )
        
        submit_btn.click(
            fn=gradio_process,
            inputs=[question_input, show_sources_checkbox],
            outputs=[response_output, sources_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[question_input, response_output, sources_output]
        )
        
        gr.Markdown(
            """
            ---
            **Conseils d'utilisation :**
            - Posez des questions précises sur les recruteuses, les mois, les candidats, etc.
            - Utilisez les noms des recruteuses (Ines, Pauline, Samya, Marième)
            - Demandez des comparaisons, des totaux, des analyses
            """
        )
    
    return demo

def main():
    try:
        retriever, chain = initialize_system()

        demo = create_gradio_interface(retriever, chain)

        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=True,
            show_error=True,
        )

        
    except FileNotFoundError:
        print(f"\nErreur : Le fichier {CSV_PATH} est introuvable.")
        print("Vérifiez que le fichier existe dans le bon répertoire.\n")
    except Exception as e:
        print(f"\nErreur lors de l'initialisation : {e}")
        print("Vérifiez votre configuration et réessayez.\n")


if __name__ == "__main__":
    main()