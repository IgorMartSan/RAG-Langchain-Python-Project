## Document Loader
## https://ollama.com/download/windows
## instalar
## Using PyPDF
##pip install pypdf
##pip install langchain-community
## Tutorial https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/custom/

#Install and use ollama
#https://www.youtube.com/watch?v=EMC5QQN_vdU
#Models
#https://www.youtube.com/watch?v=EMC5QQN_vdU


from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("D:\LLM\PDF\ActiveX for HTML pages.pdf")
pages = loader.load_and_split()

### from langchain_community.embeddings import OllamaEmbeddings
###https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
###https://python.langchain.com/v0.1/docs/integrations/text_embedding/ollama/
##A classe Embeddings é uma classe projetada para interagir com modelos de incorporação de texto.

from langchain_community.embeddings import OllamaEmbeddings
def get_embedding_function():
  embeddings = OllamaEmbeddings(model='')


## Adicionar no bando de dadis Chroma
  
def add_choroma(chunks:)