import os
import streamlit as st
from dotenv import load_dotenv
import time

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Streamlit UI Config
st.set_page_config(page_title="Clinical Trial Insight Assistant", layout="wide")
st.title("🧠 Semantic Insight Engine for Clinical Trials")

# Form Inputs
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        study_title = st.text_input("Study Title")
        conditions = st.text_input("Conditions")
        primary_outcome = st.text_input("Primary Outcome Measures")
        sex = st.text_input("Sex")
    with col2:
        age = st.text_input("Age")
        study_type = st.text_input("Study Type")
        sponsor = st.text_input("Sponsor")
        locations = st.text_input("Locations")

    submitted = st.form_submit_button("🔍 Get Top 10 Similar Studies")

# Load Vector DB
def load_vector_db():
    if "vectors" not in st.session_state:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectors = vector_store
        except Exception as e:
            st.error(f"❌ Failed to load vector DB from disk: {e}")

# Process Input and Display JSON Results
if submitted:
    load_vector_db()

    if "vectors" in st.session_state:
        user_input = f"""
        Study Title: {study_title}
        Conditions: {conditions}
        Primary Outcome Measures: {primary_outcome}
        Sex: {sex}
        Age: {age}
        Study Type: {study_type}
        Sponsor: {sponsor}
        Locations: {locations}
        """

        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})

        start = time.process_time()
        results = retriever.invoke(user_input)
        elapsed = round(time.process_time() - start, 2)

        st.markdown("### 🎯 Top 10 Semantically Relevant Studies (JSON Format)")

        for i, doc in enumerate(results, 1):
            st.subheader(f"🔹 Study {i}")
            st.json(doc.metadata)

        st.caption(f"⏱ Time Taken: {elapsed} sec")
    else:
        st.error("Vector DB not available. Please check the backend.")
