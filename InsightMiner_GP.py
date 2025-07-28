import streamlit as st
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ------------------------- Load Resources -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_model2():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("topic_9_faiss.index")

@st.cache_data
def load_dataframe():
    return pd.read_csv("arxiv_enriched.csv")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

model = load_model()
index = load_faiss_index()
df = load_dataframe()
summarizer = load_summarizer()

# -------------- Navigation Menu --------------
if "menu" not in st.session_state:
    st.session_state["menu"] = "🔍 Search Papers"

menu = st.sidebar.radio(
    "Go to",
    ["🔍 Search Papers", "📊 Compare Embedding Models"],
    index=["🔍 Search Papers", "📊 Compare Embedding Models"].index(st.session_state["menu"])
)
st.session_state["menu"] = menu

# ------------------------- Search Papers -------------------------
if menu == "🔍 Search Papers":
    st.title("📚 InsightMiner: Academic Paper Explorer")
    st.header("🔍 Search Papers")

    # Select topic
    #selected_topic = st.sidebar.selectbox("Select topic:", list(range(10)))

    # Load topic-specific data and FAISS index
    #df = load_dataframe(selected_topic)
    #index = load_faiss_index(selected_topic)

    query_input = st.text_area("Enter one or more search queries (one per line):")
    top_k = st.slider("Number of top results:", 1, 10, 3)

    if st.button("Search"):
        raw_input = query_input.strip()
        if raw_input:
            queries = [q.strip() for q in raw_input.split('\n') if q.strip()]

            if queries:
                try:
                    start = time.time()
                    results = []
                    from sentence_transformers.util import cos_sim
                    query_embeddings = model.encode(queries).astype('float32')

                    for i, query_emb in enumerate(query_embeddings):
                        D, I = index.search(np.array([query_emb]), k=top_k)
                        batch = df.iloc[I[0]].copy()
                        batch['distance'] = D[0]
                        batch['query'] = queries[i]
                        results.append(batch)

                    results = pd.concat(results).reset_index(drop=True)
                    st.session_state['last_results'] = results
                    end = time.time()
                    st.success(f"🔍 Search completed in {round(end - start, 2)} seconds.")
                except Exception as e:
                    st.error(f"Error during search: {e}")
            else:
                st.warning("Please enter at least one valid query.")
        else:
            st.warning("Please enter a query above.")

    # Show previous results after rerun
    if 'last_results' in st.session_state:
        results = st.session_state['last_results']
        for i, row in results.iterrows():
            st.markdown(f"### 🔹 {i+1}. {row['title']}")
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**Categories:** {row['categories']}")
            st.markdown(f"**Keywords:** {row['keywords']}")
            st.markdown(f"**Topic:** {row['topic']}")
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

# ------------------------- Model Comparison -------------------------
elif menu == "📊 Compare Embedding Models":
    st.title("📊 Compare Embedding Models")
    query = st.text_input("Enter a single query to compare:", value="machine learning for healthcare")
    top_k = st.slider("Top results:", 1, 10, 5)

    # Use topic 9 by default for comparison
    #df = load_dataframe(9)
    #index = load_faiss_index(9)

    if st.button("Compare"):
        with st.spinner("Running comparisons..."):
            start = time.time()
            query_emb1 = model.encode([query]).astype('float32')
            D1, I1 = index.search(query_emb1, k=top_k)
            results1 = df.iloc[I1[0]].copy()
            results1['distance'] = D1[0]
            t1 = time.time() - start

            start = time.time()
            model_2 = load_model2()
            query_emb2 = model_2.encode([query]).astype('float32')
            D2, I2 = index.search(query_emb2, k=top_k)
            results2 = df.iloc[I2[0]].copy()
            results2['distance'] = D2[0]
            t2 = time.time() - start

        st.subheader("MiniLM-L6-v2")
        for i, row in results1.iterrows():
            st.markdown(f"**{i+1}. {row['title']}**")
            st.markdown(f"Distance: {round(row['distance'], 4)}")
            st.markdown("---")

        st.subheader("Paraphrase-MiniLM-L6-v2")
        for i, row in results2.iterrows():
            st.markdown(f"**{i+1}. {row['title']}**")
            st.markdown(f"Distance: {round(row['distance'], 4)}")
            st.markdown("---")

        st.success(f"MiniLM took {t1:.2f}s | Paraphrase-MiniLM took {t2:.2f}s")
