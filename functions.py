from langchain.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings import LlamaCppEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

local_path = "./models/ggml-gpt4all-j-v1.3-groovy.bin"
model_path = "./models/alpaca-ggml-model-q4_0.bin"

llm = GPT4All(model=local_path, n_ctx=1024)
embeddings = LlamaCppEmbeddings(model_path=model_path)


def get_qdrant(collection_name: str):
    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(collection_name, VectorParams(size=4096, distance=Distance.COSINE))
    finally:
        return Qdrant(client, collection_name, embeddings)


def get_chain(collection_name: str):
    qdrant = get_qdrant(collection_name)
    retriever = qdrant.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, max_tokens_limit=256)
    return chain


def add_texts(collection_name: str, text_path: str):
    qdrant = get_qdrant(collection_name)
    loader = TextLoader(text_path, encoding="utf8")
    sources = loader.load()
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
        texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    qdrant.add_texts(texts=texts, metadatas=metadatas)


def get_answer(collection_name: str, question: list[str], chat_history: list[tuple[str, str]]) -> str:
    chain = get_chain(collection_name)
    return chain({"question": question, "chat_history": chat_history})["answer"]
