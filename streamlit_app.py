import streamlit as st
from rag import create_conversational_rag_chain
from utils import generate_random_id, ask_question


def show_ui(chain, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"👋 Hello! I'm Archie. I just read the paper you shared. {prompt_to_user}",
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")

    # User-provided prompt
    if query := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking.."):
                    response = st.write_stream(
                        ask_question(chain, query, st.session_state["session_id"])
                    )
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


def main():
    st.title("📚 Archie - The Arxiv Assistant")
    st.divider()

    if "ready" not in st.session_state:
        st.session_state["ready"] = False

    openai_api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key", type="password", placeholder="sk-xrxycrtuytvty"
    )

    with st.sidebar:
        st.divider()
        st.header("Add Research Paper")
        pdf_url = st.text_input(
            "🔗 Enter Arxiv PDF URL:", value="https://arxiv.org/pdf/2403.11703v1"
        )
        session_id = st.text_input("🆔 Enter Session ID:", value=generate_random_id())

        if st.button("🚀 Create Chatbot"):
            try:
                if not pdf_url:
                    st.error("Please enter a valid arXiv PDF URL.")
                else:
                    chain = create_conversational_rag_chain(
                        pdf_url=pdf_url, openai_api_key=openai_api_key
                    )
                    st.session_state["conversational_rag_chain"] = chain
                    st.session_state["session_id"] = session_id
                    st.success("Chatbot created successfully!")
                    st.session_state["ready"] = True

            except ValueError as e:
                st.error(str(e))
        st.divider()
        st.header("Quick Guide")
        st.write(
            """
            1. Add your OpenAI API Key
            2. Enter the Arxiv PDF URL.
            3. Optional: Have memory w/Session ID.
            4. Click 'Create Chatbot' to begin.
            5. Ask questions in the chat.
        """
        )

    if st.session_state["ready"]:
        show_ui(st.session_state["conversational_rag_chain"])


if __name__ == "__main__":
    main()
