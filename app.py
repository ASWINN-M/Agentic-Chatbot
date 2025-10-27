import streamlit as st
import os
from langchain import hub
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.utilities import WikipediaAPIWrapper , ArxivAPIWrapper , anthropic
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun
from langchain.agents import create_react_agent , AgentExecutor
from dotenv import load_dotenv
import time


load_dotenv()
os.environ["LANGCHAIN_TRACING"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = os.getenv('lang_api_key')
os.environ["GROQ_API_KEY"] = os.getenv('groq_api_key')

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = 'nomic-embed-text')
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/")
    st.session_state.doc = st.session_state.loader.load()
    st.session_state.text_split = RecursiveCharacterTextSplitter(chunk_size = 1000 ,chunk_overlap = 200)
    st.session_state.chunk_doc = st.session_state.text_split.split_documents(st.session_state.doc[:] )
    st.session_state.db = Chroma.from_documents(st.session_state.chunk_doc , st.session_state.embeddings)

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key = os.environ["GROQ_API_KEY"] , model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question in the <context>{context}<context> with good accuracy and dont give responses other than the query 
Questions : {input}
"""
)

doc_chain = create_stuff_documents_chain(llm = llm , prompt= prompt )
retriver = st.session_state.db.as_retriever()
retirver_chain = create_retrieval_chain(retriver , doc_chain)

retirver_tool = Tool(
    name="LocalDocRetriever",
    func=lambda query: retirver_chain.invoke({"input": query}),
    description="Useful for retrieving answers from the locally stored LangChain documentation."
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results= 5 , doc_content_chars_max= 500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results= 5 , doc_content_chars_max= 500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
tools = [arxiv , wiki , retirver_tool]
pull = hub.pull("hwchase17/react")
agent = create_react_agent(llm = llm , tools=tools , prompt=pull)
prompt = st.chat_input("Enter the prompt here")
agent_executor = AgentExecutor(agent=agent , tools=tools )

if prompt:
    start = time.process_time()
    response = retirver_tool.invoke({'input' : prompt})
    print("Response Time : " , time.process_time()- start)
    st.write(response['answer'])



