
# 🔍 AI Log Monitoring App

This is a Streamlit-based AI-powered log monitoring application. It allows users to query structured application logs in natural language and get intelligent answers. Built using LangChain, Ollama models, and Chroma DB for vector-based retrieval.

---

## 🧠 Features

- Log ingestion with regex-based parsing.
- Metadata extraction (timestamp, level, component, UUID).
- Query answering using RAG (Retrieval-Augmented Generation) with LLaMA3.

---

## 📁 Project Structure


---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt

Make sure you have Ollama installed and models like llama3 and mxbai-embed-large pulled

### 2. Build Vector Store
python vector_store.py

###3. Launch Streamlit App
streamlit run main.py

