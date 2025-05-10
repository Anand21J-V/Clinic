import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables (if needed)
load_dotenv()
groq_api_key = "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"

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Streamlit page config
st.set_page_config(page_title="Clinical Study Semantic Search", layout="wide")
st.title("🧠 Semantic Clinical Study Finder with LLM-Powered Insights")

# Form inputs
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
    submitted = st.form_submit_button("🔍 Search Similar Studies")

# Load vector DB with caching
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Use the provided clinical trial design characteristics to retrieve semantically similar studies from historical data.

Search Context:
{context}

Trial Design Criteria:
{input}

Give the output in clean JSON format containing a list of the top 10 similar studies with their key attributes.
""")

# If submitted, run RAG chain
if submitted:
    with st.spinner("🔎 Searching and analyzing..."):
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        user_input = f"""
        Study Title: {study_title}
        Conditions: {conditions}
        Primary Outcome Measures: {outcome}
        Sex: {sex}
        Age: {age}
        Study Type: {study_type}
        Sponsor: {sponsor}
        Location: {location}
        """

        start = time.process_time()
        result = rag_chain.invoke({"input": user_input})
        elapsed = round(time.process_time() - start, 2)

        st.subheader("📊 Top 10 Semantically Similar Studies")
        try:
            parsed_json = json.loads(result["answer"])
            st.json(parsed_json)
        except Exception as e:
            st.warning("⚠️ LLM output couldn't be parsed as JSON. Showing raw text instead.")
            st.markdown(result["answer"])

        st.caption(f"⏱ Time Taken: {elapsed} sec")
