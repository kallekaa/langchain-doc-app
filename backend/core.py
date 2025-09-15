import os
from dotenv import load_dotenv

load_dotenv() # dotenv_path="../.env"

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)  #api_key=os.getenv("OPENAI_API_KEY")

    retrieval_qa_chat_promt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_promt)

    qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain)

    result = qa.invoke(input={"input":query})
    return result

if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])


