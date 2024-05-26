from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import langchain

# langchain.debug = True
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
    vectorstore = FAISS.from_documents(
        documents=documents, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    metadata = documents[0].metadata

    return retriever, documents, metadata


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# URL of the PDF you want to process
pdf_url = "https://arxiv.org/pdf/2403.11703v1"

llm = OpenAI()
retriever, documents, metadata = create_arxiv_retriever(pdf_url=pdf_url)

history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is.""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, history_prompt)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant chatbot that helps with understanding research papers from Arxiv.
                Tone: Concise, friendly, easy to understand, Human conversation-like, Talk in third person.
                Rules: Be cautious while reading the Retrieved matching document as it may not be relevant all the time.
                If you are asked random questions or you don't know something, please respectfully decline.""",
        ),
        (
            "system",
            f'Here is a research paper from Arxiv: Title: {metadata.get("Title")}, Authors: {metadata.get("Authors")}, Abstract: {metadata.get("Summary")}',
        ),
        (
            "system",
            "Retrieved matching document \n {context}",
        ),
        (
            "human",
            "{input}",
        ),
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
