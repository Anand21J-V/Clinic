import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    print(f"✅ Vector store saved to: {save_path}")

def load_vector_store(save_path="faiss_index"):
    # Load the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the FAISS vector store with the embeddings
    vector_store = FAISS.load_local(save_path, embeddings,allow_dangerous_deserialization=True)
    
    return vector_store

if __name__ == "__main__":
    # Check if vector store already exists, if not, create it
    if not os.path.exists(save_path):
        docs = preprocess_data(clinical_json_path)
        if docs:
            build_and_save_vector_store(docs, save_path)
    else:
        print("Vector store already exists, loading it...")
