import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Path to your CSV file
clinical_csv_path = "ctg-studies (1).csv"  # <- Ensure this is cleaned and well-formatted

# Output FAISS vector DB path
save_path = "faiss_vector"

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)

        # Drop completely empty rows
        df.dropna(how='all', inplace=True)

        # Combine all columns to form the searchable content
        def row_to_text(row):
            return "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])

        df["combined_text"] = df.apply(row_to_text, axis=1)

        # Add original metadata as metadata
        documents = [
            Document(page_content=row["combined_text"], metadata=row.drop("combined_text").to_dict())
            for _, row in df.iterrows()
        ]
        return documents
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def build_and_save_vector_store(docs, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    if os.path.exists(save_path):
        print(f"⚠️ Vector store already exists at {save_path}. Skipping rebuild.")
        return

    start_time = time.time()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    elapsed_time = round(time.time() - start_time, 2)

    print(f"✅ Vector store saved to: {save_path}")
    print(f"⏱ Time taken: {elapsed_time} seconds")

if __name__ == "__main__":
    docs = preprocess_data(clinical_csv_path)
    if docs:
        build_and_save_vector_store(docs, save_path)
