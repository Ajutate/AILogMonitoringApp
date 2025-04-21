# === rag_chain.py ===
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

def get_qa_chain():
    prompt = PromptTemplate.from_template("""
You are an intelligent assistant helping users analyze application logs.

The user can ask questions like:
- "Show me distinct errors from the last 24 hours"
- "Get all ERROR logs from January"
- "Show logs between March 10 and March 15"
- "How many failed login attempts happened last week?"
- "Display ERROR logs related to database connection from February"

### Instructions:
1. Analyze the log data carefully.
2. Only use logs that match the time range or date mentioned in the question.
3. Provide **precise**, **factual**, and **concise** answers based on logs provided.
4. If no relevant logs match the user’s question (especially time/date based), respond with:

   > "No relevant log data found for the query."

5. If the query asks for distinct logs, return only unique error messages.

6. If the user asks for a count, give the number of matching logs.

Question: {question}
Log Data:
{context}

Your Answer:
""")


    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma(persist_directory="./chroma_logs", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})
    llm = OllamaLLM(model="llama3.2", temperature=0)
    #print("Vector store built and persisted successfully.")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
