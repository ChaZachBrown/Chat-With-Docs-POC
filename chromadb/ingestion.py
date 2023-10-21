from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.document_loaders import DirectoryLoader

""" Quick script to load documents into persistent Vector store using local embedding model """

# TODO: get directory from .env. Empty for public repo
directory = ""


def load_docs(directory):
    loader = DirectoryLoader(directory)
    docs = loader.load()
    return docs


def split_docs(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


documents = load_docs(directory)

split_docs = split_docs(documents)
print(len(split_docs))

# TODO: Run embeddings model on GPU
hf_embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large", model_kwargs={"device": "cpu"}
)

print("embedding documents")
Chroma.from_documents(
    documents=split_docs,
    embedding=hf_embeddings,
    persist_directory="DB",
    return_source_documents=True,
)
