
import asyncio
import os
import ssl
from dotenv import load_dotenv
from typing import Any, Dict, List

import certifi

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma # local alternative for PineCone
from langchain_core.documents import Document
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, 
                    log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"),
                                  embedding=embeddings)

#chroma = Chroma(persist_directory="chroma_db", embeddings=embeddings)
#tavily_extract = TavilyExtract()
#tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    print("Hello from langchain-doc-app!")


if __name__ == "__main__":
    main()
