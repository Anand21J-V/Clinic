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

# Path to your JSON file
clinical_json_path = "ctg-studies.json"  # <- Update with your real file

# Output FAISS vector DB path
save_path = "faiss_index"

def preprocess_data(json_path):
    try:
        df = pd.read_json(json_path)

        def combine_fields(row):
            return (
                f"Study Title: {row.get('Study Title', '')}\n"
                f"Conditions: {row.get('Conditions', '')}\n"
                f"Primary Outcome Measures: {row.get('Primary Outcome Measures', '')}\n"
                f"Sex: {row.get('Sex', '')}\n"
                f"Age: {row.get('Age', '')}\n"
                f"Study Type: {row.get('Study Type', '')}\n"
                f"Sponsor: {row.get('Sponsor', '')}\n"
                f"Locations: {row.get('Locations', '')}\n"
                f"Full Study: {row.to_json()}"
            )

        df['combined_text'] = df.apply(combine_fields, axis=1)
        documents = [Document(page_content=text) for text in df['combined_text'].tolist()]
        return documents
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return []

def build_and_save_vector_store(docs, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    
    # Check if the FAISS index already exists
    if os.path.exists(save_path):
        print(f"⚠️ Vector store already exists at {save_path}. Skipping re-building.")
        return

    # If FAISS index does not exist, proceed with creating it
    start_time = time.time()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    elapsed_time = round(time.time() - start_time, 2)
    
    print(f"✅ Vector store saved to: {save_path}")
    print(f"⏱ Time taken for vector store creation: {elapsed_time} seconds")

if __name__ == "__main__":
    docs = preprocess_data(clinical_json_path)
    if docs:
        build_and_save_vector_store(docs, save_path)
