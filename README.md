Semantic Insight Engine for Clinical Trials

This project is a **Streamlit-based semantic search engine** designed to help users find the top 10 most semantically similar clinical trial studies based on form input fields like study title, condition, outcomes, sponsor, etc.

It uses:
- Sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- FAISS (Facebook AI Similarity Search)
- LangChain's `FAISS` + `HuggingFaceEmbeddings` integration

---

## 🚀 Features

- Accepts clinical trial parameters through a user-friendly Streamlit form.
- Searches for top-10 most semantically similar studies.
- Displays metadata of matched results.
- Outputs results in readable format and JSON.
- Fast retrieval via FAISS vector store.

---

## 🗃️ Project Structure

```bash
clinical-trial-model/
│
├── app.py                     # Main Streamlit UI application
├── build_faiss_index.py      # Script to build FAISS index from JSON data
├── faiss_index/              # Directory storing FAISS vector DB (auto-generated)
├── data/
│   └── clinical_trials.json  # Sample input data (customize as needed)
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/clinical-trial-model.git
cd clinical-trial-model
```

### 2. (Optional) Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Build the FAISS Index

Before launching the app, build the FAISS index using the dataset:

```bash
python build_faiss_index.py
```

This will create a `faiss_index/` directory with saved vector data.

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

The app will launch in your browser. Fill in the form, and it will return top-10 semantically similar studies based on your input.

---

## 📦 Deployment Notes

If deploying to **Streamlit Community Cloud**, make sure to:

1. Add the following in your `requirements.txt`:

   ```text
   streamlit
   langchain
   faiss-cpu
   sentence-transformers
   python-dotenv
   ```

2. Commit the built `faiss_index/` directory or rebuild it on launch.

3. Alternatively, remove `dotenv` if not used for secrets.

---

## 📄 Example Input Fields

* Study Title
* Conditions
* Primary Outcome
* Sex
* Age
* Study Type
* Sponsor
* Locations

---

## 💡 Future Improvements

* Integrate PDF/CSV parsing for automatic input.
* Add filtering/sorting options for results.
* Add authentication & admin panel for uploading new data.

---

## 📬 Contact

Built by [Anand Vishwakarma](mailto:anandvishwakarma21j@gmail.com)
GitHub: [@anand-vishwakarma](https://github.com/anand-vishwakarma)


