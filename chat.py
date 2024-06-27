import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3")
    return embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE =  """
Você é um especialista nos documentos fornecidos abaixo. Responda à pergunta com base apenas no seguinte contexto em português:

---

{context}

---


Gere  respostas com base no contexto acima: {question}

"""

def query_rag(query_text: str, db):
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print("Bem-vindo ao sistema de perguntas e respostas. Digite 'sair' para encerrar o chat.")
    while True:
        query_text = input("Pergunta: ")
        if query_text.lower() == 'sair':
            break
        query_rag(query_text, db)

if __name__ == "__main__":
    main()
