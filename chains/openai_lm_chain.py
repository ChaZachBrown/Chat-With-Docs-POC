from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.vectorstores import Chroma

""" OpenAI gpt-3 LLM Interface with Vector store retrieval """

hf_embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large", model_kwargs={"device": "cpu"}
)

db = Chroma(persist_directory="chromadb/DB", embedding_function=hf_embeddings)

llm = ChatOpenAI(verbose=True, temperature=0)

memory = ConversationTokenBufferMemory(
    llm=llm, memory_key="chat_history", max_token_limit=4000
)


def custom_get_chat_history(input_str: str) -> str:
    return input_str


qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    rephrase_question=False,
    memory=memory,
    get_chat_history=custom_get_chat_history,
    return_source_documents=True
)


def run_openai_llm(query):
    return qa(query)
