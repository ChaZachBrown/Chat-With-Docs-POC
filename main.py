import langchain
import streamlit as st
from chains.openai_lm_chain import run_openai_llm
from langchain.callbacks import StreamlitCallbackHandler


def run_ui():
    """ Runs main streamlit chatbot UI """

    langchain.verbose = True
    langchain.debug = True

    st.title("Simple chat")
    prompt = st.text_input("Prompt", key="input", placeholder="Enter your prompt here..")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state["chat_history"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Generating response.."):
            # Display assistant response in chat message container
            assistant_response = run_openai_llm(query=prompt)

            sources = set(
                [doc.metadata["source"] for doc in assistant_response["source_documents"]]
            )

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())

                st.markdown(assistant_response["answer"] + "\n\n Sources: \n" + str(sources))

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response["answer"] + "\n\n Sources: \n" + str(sources)}
        )
        # Add prompt and response to LLM chat history
        st.session_state["chat_history"].append((prompt, assistant_response["answer"]))


if __name__ == "__main__":
    run_ui()

