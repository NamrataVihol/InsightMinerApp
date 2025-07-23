# InsightMiner_GP.py

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# ----------------------------------------
# CACHED LOADERS
# ----------------------------------------

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

# ----------------------------------------
# EMBEDDING HELPER
# ----------------------------------------

def compute_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding[0].numpy()

# ----------------------------------------
# SEARCH FUNCTION
# ----------------------------------------

def batch_search(queries, top_k=3):
    tokenizer, model = load_embedding_model()
    df = load_dataframe()
    index = load_faiss_index()

    query_embeddings = np.array([
        compute_embedding(query, tokenizer, model) for query in queries
    ]).astype('float32')

    all_results = []
    for i, query in enumerate(queries):
        D, I = index.search(query_embeddings[i:i+1], k=top_k)
        results = df.iloc[I[0]].copy()
        results['distance'] = D[0]
        results['query'] = query
        all_results.append(results)

    return pd.concat(all_results).reset_index(drop=True)

# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------

st.set_page_config(page_title="InsightMiner", layout="wide")
st.title("🔍 InsightMiner: Academic Paper Search")

query_input = st.text_area("Enter one or more search queries (one per line):")
top_k = st.slider("Number of results per query:", 1, 10, 3)
summarizer = load_summarizer()

if st.button("Search"):
    if query_input.strip():
        queries = query_input.strip().split('\n')
        results = batch_search(queries, top_k=top_k)

        for i, row in results.iterrows():
            st.markdown(f"### 🔹 {i+1}. {row['title']}")
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**Categories:** {row['categories']}")
            st.markdown(f"**Distance:** {round(row['distance'], 4)}")
            st.markdown(f"**Abstract:** {row['abstract']}")

            if st.button(f"Summarize", key=f"sum_{i}"):
                with st.spinner("Summarizing..."):
                    summary = summarizer(
                        row['abstract'],
                        max_length=60,
                        min_length=20,
                        do_sample=False
                    )[0]['summary_text']
                    st.success(f"**Summary:** {summary}")

            st.markdown("---")
    else:
        st.warning("Please enter at least one query.")
