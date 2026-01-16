# ChatGroq RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Groq LLaMA 3.1**.  
This app answers questions using:

- Local LangChain documentation (vector search)
- Wikipedia
- arXiv research papers

---

## Features

- 📚 Local document retrieval using Chroma + Ollama embeddings  
- 🤖 Groq LLaMA 3.1 for ultra-fast inference  
- 🔎 Wikipedia & arXiv integration  
- 🧠 ReAct agent for intelligent tool selection  
- 💬 Streamlit chat interface  
- ⚡ Real-time response speed tracking  

---

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Groq API**
- **Ollama Embeddings**
- **Chroma Vector DB**
- **Wikipedia & arXiv APIs**
- **dotenv**

---

## How It Works

1. Loads LangChain documentation from the web  
2. Splits it into chunks  
3. Converts text into embeddings  
4. Stores them in Chroma vector DB  
5. Uses a ReAct agent to choose between:
   - Local docs
   - Wikipedia
   - arXiv  
6. Generates accurate answers using LLaMA 3.1  

---

## Setup Instructions

### Install Dependencies

```bash
pip install streamlit langchain langchain-groq chromadb ollama python-dotenv
