import os

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceEndpoint


""" Open source LMM Interface """

load_dotenv()

HUGGING_FACE_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def chain():
    """
    Simple method to initiate and return a dedicated Huggingface inference Endpoint chain
    Example:
        hub_chain = hugginface_llm_chain.chain()
        result = hub_chain.run(question="#this is a simple snake game written in python")
    """
    # TODO: add vertor retrivals

    endpoint_url = ""
    hf = HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=HUGGING_FACE_KEY,
        task="text-generation",
    )

    template = """
    {question}
    """
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    return LLMChain(llm=hf, prompt=prompt, verbose=True)

