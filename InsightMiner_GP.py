import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ============ Caching Resources ============
@st.cache_data
def load_dataframe():
    return pd.read_csv("arxiv_metadata_final.csv")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("arxiv_10000_faiss.index")

@st.cache_resource
def load_minilm_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_paraphrase_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ============ Core Functions ============
df = load_dataframe()
def search_papers(query, model, index, df, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k=top_k)
    results = df.iloc[I[0]].copy()
    results["distance"] = D[0]
    return results

def batch_search(queries, model, index, df, top_k=5):
    all_results = []
    query_embeddings = model.encode(queries).astype("float32")
    for i, query in enumerate(queries):
        D, I = index.search(query_embeddings[i:i+1], k=top_k)
        results = df.iloc[I[0]].copy()
        results["distance"] = D[0]
        results["query"] = query
        all_results.append(results)
    return pd.concat(all_results).reset_index(drop=True)

# ============ Load Data/Models ============
df = load_dataframe()
index = load_faiss_index()
minilm_model = load_minilm_model()
paraphrase_model = load_paraphrase_model()
summarizer = load_summarizer()

if "menu" not in st.session_state:
    st.session_state["menu"] = "🔍 Search Papers"

# ============ Streamlit UI ============
st.title("📚 InsightMiner: Academic Paper Explorer")

# --------- Section 1: Search ----------
st.header("🔍 Search Papers")
query_input = st.text_area("Enter one or more search queries (one per line):")
top_k = st.slider("Number of top results:", 1, 10, 3)

if st.button("Search"):
    raw_input = query_input.strip()
    if raw_input:
        queries = [q.strip() for q in raw_input.split('\n') if q.strip()]

        if queries:
            try:
                start = time.time()
                results = batch_search(queries, minilm_model, index, df, top_k=top_k)
                st.session_state['last_results'] = results
                end = time.time()
                st.success(f"🔍 Search completed in {round(end - start, 2)} seconds.")
            except Exception as e:
                st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter at least one valid query.")
    else:
        st.warning("Please enter a query above.")

# ✅ Always render results after rerun
if 'last_results' in st.session_state:
    results = st.session_state['last_results']
    for i, row in results.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
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


            except Exception as e:
                st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter at least one valid query.")
    else:
        st.warning("Please enter a query above.")

# --------- Section 2: Compare Models ----------
st.header("📊 Compare Embedding Models")
comp_query = st.text_input("Query to Compare MiniLM vs Paraphrase-MiniLM:")
compare_k = st.slider("Top K results to compare:", 1, 10, 3)

if st.button("Compare Models"):
    if comp_query.strip():
        with st.spinner("Running model comparisons..."):
            start_time = time.time()
            results_minilm = search_papers(comp_query, minilm_model, index, df, top_k=compare_k)
            time_minilm = time.time() - start_time

            start_time = time.time()
            results_para = search_papers(comp_query, paraphrase_model, index, df, top_k=compare_k)
            time_para = time.time() - start_time

        st.subheader("MiniLM-L6-v2 Results")
        for i, row in results_minilm.iterrows():
            st.markdown(f"🔹 {row['title']} ({round(row['distance'], 4)})")

        st.subheader("Paraphrase-MiniLM-L6-v2 Results")
        for i, row in results_para.iterrows():
            st.markdown(f"🔹 {row['title']} ({round(row['distance'], 4)})")

        st.info(f"⏱️ Time Taken: MiniLM = {time_minilm:.2f}s | Paraphrase-MiniLM = {time_para:.2f}s")
    else:
        st.warning("Please enter a query to compare models.")
