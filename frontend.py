import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
import json

# Load the FAISS vector store
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return vector_store

# Construct combined input from user fields
def create_query(study_title, conditions, outcome, sex, age, study_type, sponsor, location):
    return (
        f"Study Title: {study_title}\n"
        f"Conditions: {conditions}\n"
        f"Primary Outcome Measures: {outcome}\n"
        f"Sex: {sex}\n"
        f"Age: {age}\n"
        f"Study Type: {study_type}\n"
        f"Sponsor: {sponsor}\n"
        f"Locations: {location}"
    )

# Streamlit UI
st.set_page_config(page_title="Clinical Study Semantic Search", layout="wide")
st.title("🔍 Semantic Clinical Study Finder")

# Input fields
with st.form("study_input"):
    col1, col2 = st.columns(2)
    with col1:
        study_title = st.text_input("Study Title")
        conditions = st.text_input("Conditions")
        outcome = st.text_input("Primary Outcome Measures")
        sex = st.text_input("Sex")
    with col2:
        age = st.text_input("Age")
        study_type = st.text_input("Study Type")
        sponsor = st.text_input("Sponsor")
        location = st.text_input("Location")

    submitted = st.form_submit_button("Search")

if submitted:
    with st.spinner("🔎 Searching... Please wait..."):
        vs = load_vector_store()
        query = create_query(study_title, conditions, outcome, sex, age, study_type, sponsor, location)
        results = vs.similarity_search(query, k=10)

    st.success("✅ Top 10 Related Clinical Studies:")
    for idx, doc in enumerate(results, 1):
        st.markdown(f"### 🔹 Result {idx}")
        try:
            data = json.loads(doc.page_content.split("Full Study:")[-1])
            for key, value in data.items():
                st.markdown(f"**{key}**: {value}")
        except Exception:
            st.markdown(doc.page_content)
        st.markdown("---")
