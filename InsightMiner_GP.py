#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
#import json

#file_path = "arxiv-metadata-oai-snapshot.json"

#records = []

#with open(file_path, 'r', encoding='utf-8') as f:
   # for line in f:
    #    record = json.loads(line)
     #   selected_record = {
      #      'id': record.get('id'),
       #     'title': record.get('title'),
        #    'abstract': record.get('abstract'),
         #   'categories': record.get('categories'),
          #  'authors': record.get('authors')  # Added here
        #}
        #records.append(selected_record)

#df = pd.DataFrame(records)

#print(df.head())


# In[4]:


#df.info()


# In[5]:


#df.head(5)


# In[7]:


#df.to_csv("arxiv_metadata_clean.csv", index=False)


# In[3]:


#import pandas as pd
# Load full dataset
#df = pd.read_csv("arxiv_metadata_clean.csv")

# Select the first 10,000 rows
#df_small = df.iloc[:10000]

# Save as a new CSV
#df_small.to_csv("arxiv_metadata_10000.csv", index=False)

#print("Saved arxiv_metadata_10000.csv with 10,000 rows.")


# In[4]:


# Load existing CSV
#df1 = pd.read_csv("arxiv_metadata_10000.csv")

# Clean and split categories into lists
#df1['categories'] = df1['categories'].fillna('').apply(lambda x: x.split() if isinstance(x, str) else [])

# Clean and split authors into lists
#df1['authors'] = df1['authors'].fillna('').apply(lambda x: [author.strip() for author in x.split(',')] if isinstance(x, str) else [])


# In[6]:


#print(df1[['categories', 'authors']].head())


# In[7]:


#df1.head()


# In[8]:


# Extract all categories into one flat list
#all_categories = df1['categories'].explode().unique().tolist()

#print("Total unique categories:", len(all_categories))
#print(all_categories[:10])  # Preview first 10 categories


# In[9]:


# Extract all authors into one flat list
#all_authors = df1['authors'].explode().unique().tolist()

#print("Total unique authors:", len(all_authors))
#print(all_authors[:10])  # Preview first 10 authors


# In[10]:


#df1.head()


# In[11]:


#df1.info()


# In[12]:


#print(type(df1['categories'].iloc[0]))
#print(type(df1['authors'].iloc[0]))


# In[13]:


#df1['categories'] = df1['categories'].apply(lambda x: '; '.join(x))
#df1['authors'] = df1['authors'].apply(lambda x: '; '.join(x))


# In[14]:


#print(type(df1['categories'].iloc[0]))
#print(type(df1['authors'].iloc[0]))


# In[15]:


#df1.head()


# In[16]:


#df1.to_csv("arxiv_metadata_final.csv", index=False)
#print("Saved arxiv_metadata_final.csv with clean lists as strings.")
df1 = pd.read_csv("arxiv_metadata_final.csv")

# In[21]:


print(type(df1['title'].iloc[0]))
print(type(df1['abstract'].iloc[0]))


# In[41]:


#pip install sentence-transformers


# In[48]:


#pip install tf-keras


# In[23]:


from sentence_transformers import SentenceTransformer

# Combine title + abstract into a new column
df1['combined_text'] = df1['title'].astype(str) + " " + df1['abstract'].astype(str)
df1['combined_text'].head()


# In[24]:

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu", trust_remote_code=True)
# Load SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()
# Generate embeddings
embeddings = model.encode(df1['combined_text'].tolist(), show_progress_bar=True)

print("✅ Embedding shape:", embeddings.shape)


# In[25]:


#pip install faiss-cpu


# In[26]:


import faiss
import numpy as np


# In[27]:


print(embeddings.shape)  # Should show (10000, 384)


# In[28]:


#Initialize FAISS Index
embedding_dimension = embeddings.shape[1]  # 384 for MiniLM

index = faiss.IndexFlatL2(embedding_dimension)  # Using L2 distance


# In[29]:


# Add Embeddings to the Index
index.add(np.array(embeddings).astype('float32'))
print("Total items in index:", index.ntotal)


# In[30]:


#Save the Index
faiss.write_index(index, "arxiv_10000_faiss.index")
print("Index saved as arxiv_10000_faiss.index")


# In[31]:


# Example Query Search
query = "deep learning for image recognition"
query_embedding = model.encode([query])


# In[32]:


D, I = index.search(np.array(query_embedding).astype('float32'), k=5)  # Top 5 matches

print("Indices of top results:", I)
print("Distances:", D)


# In[33]:


results = df1.iloc[I[0]]
print(results[['id', 'title', 'abstract']])


# In[34]:


#Build the Retrieval Function

# Load FAISS index
index = faiss.read_index("arxiv_10000_faiss.index")

# Load your embedding model
#model = SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# In[35]:


# Define Search Function
def search_papers(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query]).astype('float32')
    
    # Search FAISS index
    D, I = index.search(query_embedding, k=top_k)

    # Retrieve matching rows
    results = df1.iloc[I[0]].copy()
    results['distance'] = D[0]
    
    return results[['id', 'title', 'abstract', 'categories', 'authors', 'distance']]


# In[36]:


results = search_papers("machine learning for healthcare", top_k=5)
print(results)


# In[37]:


# Batch Search System (Multiple Queries at Once)
def batch_search(queries, top_k=5):
    # Encode all queries at once
    query_embeddings = model.encode(queries).astype('float32')
    
    # Search for each query
    all_results = []

    for i, query in enumerate(queries):
        D, I = index.search(query_embeddings[i:i+1], k=top_k)
        results = df1.iloc[I[0]].copy()
        results['distance'] = D[0]
        results['query'] = query
        all_results.append(results)

    return pd.concat(all_results).reset_index(drop=True)


# In[38]:


queries = [
    "neural networks for image processing",
    "optimization techniques in machine learning",
    "physics papers about quantum mechanics"
]

batch_results = batch_search(queries, top_k=3)
print(batch_results[['query', 'title', 'distance']])


# In[39]:


#pip install streamlit


# In[40]:


import streamlit as st

# Streamlit UI layout
st.title("InsightMiner: Academic Paper Search")

# Input box for single or batch queries
query_input = st.text_area("Enter one or more search queries (one per line):")

top_k = st.slider("Number of results per query:", 1, 10, 3)

if st.button("Search"):
    if query_input.strip():
        queries = query_input.strip().split('\n')
        results = batch_search(queries, top_k=top_k)
        
        # Display in Streamlit table
        #st.write(results[['query', 'title', 'authors', 'categories', 'distance']])
        # Loop over each result and display nicely
        for i, row in results.iterrows():
            st.markdown(f"### 🔹 {i+1}. {row['title']}")
            st.markdown(f"**Authors:** {row['authors']}")
            st.markdown(f"**Categories:** {row['categories']}")
            st.markdown(f"**Distance:** {round(row['distance'], 4)}")
            st.markdown(f"**Abstract:** {row['abstract']}")
            st.markdown("---")

    else:
        st.warning("Please enter at least one query.")


# In[ ]:

@st.cache_resource
#def load_model():
 #   return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("arxiv_10000_faiss.index")

@st.cache_data
def load_dataframe():
    return pd.read_csv("arxiv_metadata_final.csv")

# Load only once
#model = load_model()
index = load_faiss_index()
df = load_dataframe()



