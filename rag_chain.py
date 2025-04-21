# === rag_chain.py ===
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

def get_qa_chain():
    prompt = PromptTemplate.from_template("""
You are an intelligent log monitoring assistant. Your job is to analyze application logs and answer the user's query based on the given log entries. The logs contain structured information such as timestamps, severity levels (ERROR, INFO, WARN), components (e.g., DB, API, Server), and messages.

Understand the intent of the question and provide a clear, concise, and accurate answer.

You can perform tasks such as:
- Counting specific types of errors or events
- Extracting unique messages or patterns
- Summarizing key events over a period
- Identifying the root cause of issues
- Show logs only if it is present and supports the answer

Question: {question}
Log Data:
{context}

Your Answer:
""")


    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma(persist_directory="./chroma_logs", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 20})
    llm = OllamaLLM(model="llama3.2")
    #print("Vector store built and persisted successfully.")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
