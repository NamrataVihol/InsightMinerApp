import streamlit as st
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# ----------------------------
# Load and Cache Resources
# ----------------------------
@st.cache_resource
def load_dataframe():
    return pd.read_csv("arxiv_metadata_final.csv")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("arxiv_10000_faiss.index")

@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ----------------------------
# Helper Functions
# ----------------------------
def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding[0].numpy()

def batch_search(queries, top_k=3):
    results_all = []
    for query in queries:
        embedding = compute_embedding(query, tokenizer, model).astype('float32')
        D, I = index.search(np.array([embedding]), k=top_k)
        results = df.iloc[I[0]].copy()
        results['distance'] = D[0]
        results['query'] = query
        results_all.append(results)
    return pd.concat(results_all).reset_index(drop=True)

# ----------------------------
# Load models and data
# ----------------------------
df = load_dataframe()
index = load_faiss_index()
tokenizer, model = load_embedding_model()
summarizer = load_summarizer()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("🔍 InsightMiner: Academic Paper Search")

# Init session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'queries' not in st.session_state:
    st.session_state.queries = None

# Input & Search
query_input = st.text_area("Enter one or more search queries (one per line):")
top_k = st.slider("Number of results per query:", 1, 10, 3)

if st.button("Search"):
    if query_input.strip():
        st.session_state.queries = query_input.strip().split('\n')
        st.session_state.results = batch_search(st.session_state.queries, top_k=top_k)
    else:
        st.warning("Please enter at least one query.")

# Show Results
if st.session_state.results is not None:
    for i, row in st.session_state.results.iterrows():
        st.markdown(f"### 🔹 {i+1}. {row['title']}")
        st.markdown(f"**Authors:** {row['authors']}")
        st.markdown(f"**Categories:** {row['categories']}")
        st.markdown(f"**Distance:** {round(row['distance'], 4)}")
        st.markdown(f"**Abstract:** {row['abstract']}")

        if st.button("Summarize", key=f"sum_{i}"):
            with st.spinner("Summarizing..."):
                summary = summarizer(
                    row['abstract'],
                    max_length=60,
                    min_length=20,
                    do_sample=False
                )[0]['summary_text']
                st.success(f"**Summary:** {summary}")

        st.markdown("---")
