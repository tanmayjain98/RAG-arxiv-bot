import os
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import io

load_dotenv()


def extract_pdf_texts(pdfs):
    """Read and process the content of a PDF file uploaded in Streamlit."""
    for file in pdfs:
        try:
            # Read the file directly from the file-like object
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if there is text on the page
                    text += page_text + "\n"
            return text
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"


def setup_retriever(raw_text, embeddings_model=OpenAIEmbeddings()):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    documents = text_splitter.split_text(raw_text)
    db = FAISS.from_texts(texts=documents, embedding=embeddings_model)
    retriever = db.as_retriever()
    return retriever


# Define a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Use the given context to answer the question concisely."),
        ("human", "{input}"),
    ]
)
# llm = Ollama(model="llama3")
llm = OpenAI()

system_prompt = (
    "You are a language model that acts as a personal assistant to Tanmay."
    "In the context, I have provided you with a bio about Tanmay."
    "Please keep your responses converstaional and concise."
    "Only if needed, use the following pieces of retrieved context to answer the question. Do not use context if it not required to answer."
    "If you don't know the answer, please respectfully decline"
    "\n\n"
    "Context:"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")


# Function to get LLM response using the retrieval chain
def get_llm_response(chain, query):
    # print(query)
    result = chain.invoke({"input": f"{query}"})
    print(result)

    return result

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

    with st.sidebar:
        st.subheader("Your documents")
        # File uploader allows multiple files
        files = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            if files:
                with st.spinner("Processing"):
                    raw_text = extract_pdf_texts(files)
                    retriever = setup_retriever(raw_text=raw_text)
                    st.success("Added PDF knowledge base.")
                    chain = create_history_aware_retriever(
                        llm, retriever, rephrase_prompt
                    )
                    chain.invoke({"input": input, "chat_history":})
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
