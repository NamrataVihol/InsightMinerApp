
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

# Load data
@st.cache_data
def load_dataframe():
    return pd.read_csv("arxiv_metadata_final.csv")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("arxiv_10000_faiss.index")

# Dictionary of model names
model_names = {
    "MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "Paraphrase-MiniLM": "paraphrase-MiniLM-L6-v2",
    "QA-MiniLM": "multi-qa-MiniLM-L6-dot-v1"
}

df = load_dataframe()
index = load_faiss_index()

# Create combined_text column if not present
if "combined_text" not in df.columns:
    df["combined_text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)

# ---------------------- Streamlit Layout -------------------------

st.title("InsightMiner: Academic Paper Search + Model Comparison")

st.markdown("### 🔍 Single/Batch Paper Search")

query_input = st.text_area("Enter one or more search queries (one per line):")
top_k = st.slider("Number of results per query:", 1, 10, 3)

# Load default model
@st.cache_resource
def load_default_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_default_model()

def batch_search(queries, top_k=5):
    query_embeddings = model.encode(queries).astype("float32")
    all_results = []

    for i, query in enumerate(queries):
        D, I = index.search(query_embeddings[i:i+1], k=top_k)
        results = df.iloc[I[0]].copy()
        results["distance"] = D[0]
        results["query"] = query
        all_results.append(results)

    return pd.concat(all_results).reset_index(drop=True)

if st.button("Search"):
    if query_input.strip():
        queries = query_input.strip().split("\n")
        results = batch_search(queries, top_k=top_k)

        for i, row in results.iterrows():
            st.markdown(f"### 🔹 {i+1}. {row['title']}")
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**Categories:** {row['categories']}")
            st.markdown(f"**Distance:** {round(row['distance'], 4)}")
            st.markdown(f"**Abstract:** {row['abstract'][:500]}...")
            st.markdown("---")
    else:
        st.warning("Please enter at least one query.")

# ---------------------- Comparison Section -------------------------

st.markdown("---")
st.markdown("## 🧪 Compare Embedding Models")

with st.expander("🔬 Run Comparison"):
    selected_models = st.multiselect("Choose models to compare", list(model_names.keys()), default=["MiniLM-L6-v2"])
    compare_query = st.text_input("Enter a search query for comparison:")
    compare_k = st.slider("Top results per model:", 1, 10, 3)

    if st.button("Compare Now"):
        if not compare_query.strip():
            st.warning("Please enter a query to compare.")
        else:
            for label in selected_models:
                st.markdown(f"### 🔹 Results from `{label}`")
                try:
                    with st.spinner(f"Loading {label}..."):
                        model = SentenceTransformer(model_names[label])
                        start = time.time()
                        query_embed = model.encode([compare_query])
                        doc_embeds = model.encode(df["combined_text"].tolist())
                        faiss_index = faiss.IndexFlatL2(doc_embeds.shape[1])
                        faiss_index.add(np.array(doc_embeds).astype("float32"))
                        D, I = faiss_index.search(np.array(query_embed).astype("float32"), k=compare_k)
                        end = time.time()

                        results = df.iloc[I[0]].copy()
                        st.markdown(f"🕒 **Search Time:** {end - start:.2f} seconds")
                        st.markdown(f"📊 **Avg Distance Score:** {np.mean(D[0]):.4f}")

                        for j, row in results.iterrows():
                            st.markdown(f"**{j+1}. {row['title']}**")
                            st.markdown(f"*{row['authors']}* — `{row['categories']}`")
                            st.markdown(f"`Abstract:` {row['abstract'][:300]}...")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Error with {label}: {str(e)}")
