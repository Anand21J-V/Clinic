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
clinical_json_path = "clinical_studies.json"  # <- Update with your real file

# Output FAISS vector DB path
save_path = "faiss_index"

# Function to combine input fields into a single text
def combine_input(input_data):
    return (
        f"Study Title: {input_data.get('Study Title', '')}\n"
        f"Conditions: {input_data.get('Conditions', '')}\n"
        f"Primary Outcome Measures: {input_data.get('Primary Outcome Measures', '')}\n"
        f"Sex: {input_data.get('Sex', '')}\n"
        f"Age: {input_data.get('Age', '')}\n"
        f"Study Type: {input_data.get('Study Type', '')}\n"
        f"Sponsor: {input_data.get('Sponsor', '')}\n"
        f"Locations: {input_data.get('Locations', '')}\n"
    )

# Preprocessing function to combine all study features
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
        return df, documents
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None, []

# Function to build the vector store and save it
def build_and_save_vector_store(docs, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    print(f"✅ Vector store saved to: {save_path}")
    return vector_store

# Function to load the existing vector store
def load_vector_store(save_path="faiss_index"):
    if os.path.exists(save_path):
        vector_store = FAISS.load_local(save_path)
        print("✅ Vector store loaded successfully.")
        return vector_store
    else:
        print("❌ Vector store not found!")
        return None

# Function to get the top 10 semantically similar studies
def find_similar_studies(input_data, vector_store, df, top_k=10):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    input_text = combine_input(input_data)
    query_embedding = embeddings.embed_documents([input_text])[0]
    
    results = vector_store.similarity_search_by_vector(query_embedding, k=top_k)
    related_studies = []
    
    for result in results:
        index = result.metadata['index']  # Index of the document in the dataframe
        study_data = df.iloc[index].to_dict()  # Retrieve the full study data
        related_studies.append(study_data)
    
    return related_studies

if __name__ == "__main__":
    # Check if the vector store exists
    vector_store = load_vector_store(save_path)
    df, docs = preprocess_data(clinical_json_path)

    if df is not None and docs:
        if vector_store is None:
            # If vector store doesn't exist, build and save it
            build_and_save_vector_store(docs, save_path)
            vector_store = load_vector_store(save_path)
        
        # Example input data (to be taken from the frontend)
        input_data = {
            "Study Title": "Effect of Drug X on Blood Pressure",
            "Conditions": "Hypertension",
            "Primary Outcome Measures": "Change in systolic blood pressure",
            "Sex": "All",
            "Age": "18-65",
            "Study Type": "Interventional",
            "Sponsor": "Pharma Inc.",
            "Locations": "New York, USA"
        }
        
        # Get top 10 similar studies
        start_time = time.time()
        similar_studies = find_similar_studies(input_data, vector_store, df)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Found {len(similar_studies)} related studies in {elapsed_time:.2f} seconds.")
        
        # Display the output (you can return this in your frontend)
        for study in similar_studies:
            print(study)
    else:
        print("❌ Data not processed successfully.")
