from backend.core import run_llm

import streamlit as st

#import os
#from dotenv import load_dotenv
#load_dotenv()

st.header("LangChain Udemy Course - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating responses..."):
        generated_reponse = run_llm(query=prompt)
        sources = set([doc.metadata["source"] for doc in generated_reponse["source_documents"]])

        formatted_response = (
            f"{generated_reponse['result']} \n\n {create_sources_string(sources)}"
        ) 


# def main():
#     print("Hello from langchain-doc-app!")


# if __name__ == "__main__":
#     main()
