import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('groq_api_key')

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = 'llama3.2')
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/")
    st.session_state.doc = st.session_state.loader.load()
    st.session_state.text_split = RecursiveCharacterTextSplitter(chunk_size = 1000 ,chunk_overlap = 200)
    st.session_state.chunk_doc = st.session_state.text_split.split_documents(st.session_state.doc[:70] )
    st.session_state.db = Chroma.from_documents(st.session_state.chunk_doc , st.session_state.embeddings)

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key = os.environ["GROQ_API_KEY"] , model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question in the <context>{context}<context> with good accuracy if the answer is correct then i will tip you $10
Questions : {input}
"""
)

doc_chain = create_stuff_documents_chain(llm = llm , prompt= prompt )
retriver = st.session_state.db.as_retriever()
retirver_chain = create_retrieval_chain(retriver , doc_chain)

prompt = st.chat_input("Enter the prompt here")

if prompt:
    start = time.process_time()
    response = retirver_chain.invoke({'input' : prompt})
    print("Response Time : " , time.process_time()- start)
    st.write(response['answer'])



