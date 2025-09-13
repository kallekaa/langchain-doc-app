# To install: pip install tavily-python
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

search_response = client.search(
    query="What are the best practices for implementing RAG in production?"
)
print("\n\n*** search_response: \n\n", search_response)

extract_response = client.extract(
    urls=["https://www.britannica.com/science/general-relativity",
          "https://www.britannica.com/science/cosmological-constant",
          "https://en.wikipedia.org/wiki/Galaxy"]
)
print("\n\n*** extract_response: \n\n", extract_response)

crawl_response = client.crawl(
    url="https://tavily.com/",
    instructions="Show me information about the JS and Python SDKs",
    max_breadth=5,
    extract_depth="basic"
)
print("\n\n*** crawl_response: \n\n",crawl_response)

