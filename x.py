from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import langchain

langchain.debug = True

import requests
from PyPDF2 import PdfReader
import io
import re

from dotenv import load_dotenv

load_dotenv()


def extract_arxiv_id(url):
    """
    Extracts the arXiv ID from a given arXiv PDF URL.

    Parameters:
    url (str): The URL to an arXiv PDF document.

    Returns:
    str: The extracted arXiv ID, or None if no valid ID is found.
    """
    # Regular expression to match the necessary part of the URL
    match = re.search(r"/pdf/(.+?)(\.pdf)?$", url)
    if match:
        return match.group(1)
    else:
        return None


def create_arxiv_retriever(pdf_url):
    arxiv_id = extract_arxiv_id(pdf_url)
    loader = ArxivLoader(query=arxiv_id)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)
    documents = text_splitter.split_documents(documents=documents)
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-l6-v2"
    # )
    vectorstore = FAISS.from_documents(
        documents=documents, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return retriever


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def concatenate_documents(documents):
    """
    Concatenate the page content of each document in the list into a single text string.

    Parameters:
    documents (list of Document): List containing Document objects.

    Returns:
    str: A single string containing all the document contents concatenated together.
    """
    all_text = ""
    for doc in documents:
        all_text += (
            doc.page_content + "\n\n"
        )  # Append each document's content followed by two new lines
    return all_text


def format_metadata(metadata):
    formatted_metadata = ""
    formatted_metadata += f"Title: {metadata.get('Title')}, Authors: {metadata.get('Authors')}, Published: {metadata.get('Published')}, Abstract: {metadata.get('Summary')}.\n"
    return formatted_metadata


def retrieve_with_metadata(retriever, query):
    """
    Retrieve documents along with metadata using an existing VectorStoreRetriever.

    Parameters:
    retriever (VectorStoreRetriever): The retriever instance used for fetching documents.
    query (str): The query string used to retrieve documents.

    Returns:
    tuple: A tuple containing two elements: a list of documents and their corresponding metadata.
    """
    # Retrieve documents using the existing functionality of the retriever
    documents = retriever.invoke(query)
    metadata = documents[0].metadata
    documents = concatenate_documents(documents=documents)

    return documents, metadata


def prepare_prompt_with_metadata(metadata, documents, user_query):
    formatted_metadata = format_metadata(metadata)
    full_context = f"{formatted_metadata}\n\n{documents}"  # Concatenate metadata with document text
    return {"input": user_query, "context": full_context}


# URL of the PDF you want to process
pdf_url = "https://arxiv.org/pdf/2403.11703v1"

llm = OpenAI()
retriever = create_arxiv_retriever(pdf_url=pdf_url)

# ### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are a language model that acts as an academic assistant for question-answering tasks. \
You read technical AI papers and help the human understand it. \
Tone: Concise, easy to understand, human-like converstations. \
Tasks: Summarization, Explain in simple words, Gist of the findings, etc. \
Use the following pieces of retrieved context to answer the question. \
Please give importance to Title, Authors and Abstract. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


user_query = "Who is the author of the paper?"
documents, metadata = retrieve_with_metadata(retriever, user_query)
prompt = prepare_prompt_with_metadata(metadata, documents, user_query)
response = conversational_rag_chain.invoke(
    prompt,
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)
print(response["answer"])
