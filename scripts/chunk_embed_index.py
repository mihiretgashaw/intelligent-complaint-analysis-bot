# Import libraries 

import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# === Configuration ===
DATA_PATH = "data/filtered_complaints.csv"

VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # number of characters
CHUNK_OVERLAP = 50

# === Load cleaned data ===
df = pd.read_csv(DATA_PATH)

# Fill missing fields 
df = df.dropna(subset=['cleaned_narrative', 'Product'])

# === Step 1: Chunking using LangChain's RecursiveCharacterTextSplitter ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = []
metadatas = []

for idx, row in df.iterrows():
    chunks = splitter.split_text(row['cleaned_narrative'])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "complaint_id": idx,
            "product": row['Product'],
            "chunk_index": i
        })

print(f"Total chunks created: {len(documents)}")

# === Step 2: Load Embedding Model ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = model.encode(documents, show_progress_bar=True)

# === Step 3: Create FAISS Index ===
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# === Step 4: Save FAISS Index and Metadata ===
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "faiss_index.index"))

with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadatas, f)

print(f"\n Vector store saved in '{VECTOR_STORE_DIR}/' with {len(documents)} vectors.")
