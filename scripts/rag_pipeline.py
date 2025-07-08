import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load vector store and metadata
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load FAISS index
index = faiss.read_index(f"{VECTOR_STORE_DIR}/faiss_index.index")

# Load metadata
with open(f"{VECTOR_STORE_DIR}/metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Load original data for chunk retrieval
df = pd.read_csv("data/filtered_complaints.csv")

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = []
    
    for i in indices[0]:
        meta = metadatas[i]
        text = df.loc[meta['complaint_id'], 'cleaned_narrative']
        retrieved_chunks.append(text)
    
    return retrieved_chunks

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks[:3])  
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""
    return prompt

from transformers import pipeline

# Load generator (You can choose a lightweight one if running locally)
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)

def generate_answer(question):
    chunks = retrieve_relevant_chunks(question)
    prompt = build_prompt(chunks, question)
    result = generator(prompt)[0]['generated_text']
    return result

