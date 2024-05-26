from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import re

from prompts import create_history_prompt, create_qa_prompt

load_dotenv()


def extract_arxiv_id(url):
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


def create_conversational_rag_chain(pdf_url, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    retriever, documents, metadata = create_arxiv_retriever(pdf_url=pdf_url)

    history_prompt = create_history_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_prompt
    )

    qa_prompt = create_qa_prompt(metadata)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
