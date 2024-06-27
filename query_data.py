import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="llama3")
    return embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE =  """
Você é um especialista nos documentos fornecidos abaixo. Responda à pergunta com base apenas no seguinte contexto. Gere cinco respostas diferentes e, em seguida, combine essas respostas para formar uma única resposta final em português:

Exemplos de Perguntas e Respostas:

1. Pergunta: Quais são as interfaces principais do SDK do Digifort?
   Resposta: O SDK do Digifort é composto por várias interfaces de API destinadas a integrações específicas. As interfaces principais incluem ActiveX, HTTP, e RTSP, cada uma servindo a diferentes propósitos de integração. (Referência: SDK - Readme, pág. 3)

2. Pergunta: O que é a interface ActiveX do Digifort e para que serve?
   Resposta: A interface ActiveX do Digifort permite a visualização ao vivo e o controle PTZ (Pan-Tilt-Zoom) através de controles ActiveX. É recomendada para integrações rápidas em páginas HTML, especialmente usando o Internet Explorer. (Referência: ActiveX for HTML pages, pág. 2-3)

3. Pergunta: Como a interface HTTP do Digifort pode ser utilizada?
   Resposta: A interface HTTP do Digifort permite o acesso a dados do servidor, controle de câmeras PTZ, acionamento de alarmes, e visualização de imagens ao vivo e gravadas através de chamadas HTTP GET/POST. (Referência: SDK - Readme, pág. 4)

4. Pergunta: Quais eventos podem ser monitorados usando variáveis de evento no Digifort?
   Resposta: Eventos que podem ser monitorados usando variáveis de evento incluem falha de comunicação, perda de vídeo do dispositivo, falha de disco do dispositivo, e detecção de movimento, entre outros. (Referência: EventVariables, pág. 6-9)

5. Pergunta: Qual é a funcionalidade da interface RTSP do Digifort?
   Resposta: A interface RTSP do Digifort permite a solicitação de vídeo ao vivo de câmeras do servidor Digifort usando o protocolo RTSP, compatível com qualquer player de mídia que suporte este protocolo. (Referência: Digifort RTSP Interface 7.3.0, pág. 3)

---

{context}

---

Gere cinco respostas com base no contexto acima: {question}

1. 
2. 
3. 
4. 
5. 

---

Baseado nas cinco respostas acima, gere uma única resposta final.
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()