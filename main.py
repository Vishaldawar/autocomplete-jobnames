import streamlit as st
import pandas as pd
import numpy as np

from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from gensim.models import word2vec
from sentence_transformers import SentenceTransformer

def findSimilarity(w1, w2):
    """
    Returns normalized levenshtein similarity between two input strings w1 and w2.
    """
    normalized_levenshtein = NormalizedLevenshtein()
    return normalized_levenshtein.similarity(w1, w2)

# @app.get("/match-medicine/")
def match_jobs(query: str):
    """
    Finds the top 3 similar job names based on the input query using levenshtein similarity w1 and w2.
    """

    all_simi = pd.DataFrame(columns=['input','match','score'])
    for idx, row in job_list.iterrows():
        w1 = query
        for col in ['job_name']:
            w2 = row[col]
            simi = findSimilarity(w1, w2.lower())
            all_simi.loc[len(all_simi)] = [w1, w2, simi]
    matches = all_simi.sort_values('score',ascending=False).head(10)[['match','score']].values
    return [match[0] for match in matches], [
        {"job_name": match[0], "similarity_score": np.round(match[1],3)}
        for match in matches
    ]

# @app.get("/match-medicine/")
def match_jobs_word2vec(query: str):
    """
    Finds the top 3 similar job names based on the input query using word2vec embeddings.
    """
    word2vec_model = word2vec.Word2Vec.load("./word2vec_model/word2vec_trained_model.bin")
    tensor = np.zeros(256)
    for w in query.split(" "):
        if w in word2vec_model.wv.key_to_index.keys():
            tensor = tensor + word2vec_model.wv[w]
    all_simi = pd.DataFrame(columns=['input','match','score'])
    for col in embeddings.columns:
        w1 = query
        w2 = col
        v1 = tensor
        v2 = embeddings[w2]
        # simi = findSimilarity(w1, w2.lower())
        simi = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        all_simi.loc[len(all_simi)] = [w1, w2, simi]
    matches = all_simi.sort_values('score',ascending=False).head(10)[['match','score']].values
    return [match[0] for match in matches], [
        {"job_name": match[0], "similarity_score": np.round(match[1],3)}
        for match in matches
    ]

def match_jobs_llm(query: str):
    """
    Finds the top 3 similar job names based on the input query using word2vec embeddings.
    """
    w1 = query
    v1 = llm_model.encode(w1)
    all_simi = pd.DataFrame(columns=['input','match','score'])
    for col in embeddings_llm.columns:
        w2 = col
        v2 = llm_model.encode(w2)
        # simi = findSimilarity(w1, w2.lower())
        simi = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        all_simi.loc[len(all_simi)] = [w1, w2, simi]
    matches = all_simi.sort_values('score',ascending=False).head(10)[['match','score']].values
    return [match[0] for match in matches], [
        {"job_name": match[0], "similarity_score": np.round(match[1],3)}
        for match in matches
    ]

def match_data(matches, method):
    df = jobs[jobs['job_name'].isin(matches)].copy()
    df['method'] = method
    return df


# Load job titles from your scraped data
@st.cache_data
def load_jobs():
    df = pd.read_parquet("all_jobs_db.parquet")
    # return df["job_name"].dropna().unique().tolist()
    return df

# Load embeddings from your saved embeddings
@st.cache_data
def load_embeddings():
    df = pd.read_parquet("./data/embeddings_word2vec.parquet")
    # return df["job_name"].dropna().unique().tolist()
    return df

# Load embeddings from your saved embeddings
@st.cache_data
def load_llm_embeddings():
    df = pd.read_parquet("./data/embeddings_llm.parquet")
    # return df["job_name"].dropna().unique().tolist()
    return df

@st.cache_data
def load_llm_model():
    llm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return llm_model

# Set page style
st.set_page_config(page_title="Live Job Search", layout="centered")

st.markdown("""
    <style>
        .big-font {
            font-size: 26px !important;
            font-weight: 600;
            color: #0a66c2;
        }
        .autocomplete-box {
            border: 1px solid #d3d3d3;
            border-radius: 8px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .job-item {
            margin-bottom: 8px;
            font-size: 18px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)



# Load job titles
jobs = load_jobs()
embeddings = load_embeddings()
embeddings_llm = load_llm_embeddings()
llm_model = load_llm_model()
job_list = jobs[['job_name']].drop_duplicates()

# Create input box with key to track typing
st.markdown(
    # "<h1 style='text-align: center; font-size: 60px; color: #4CAF50;'>Job Finder 🔍</h1>",
        "<h1 style='text-align: center; font-size: 60px; color: #007BFF;'>Job Finder 🔍</h1>",
    unsafe_allow_html=True
)
st.markdown("### 🔍 Type a job title:")

st.text_input("Search for a job", key="search_query", label_visibility="collapsed")
query = st.session_state.get("search_query", "").strip()

# In main layout
algorithm = st.selectbox(
    "Choose a matching algorithm:",
    ["String Similarity Search", "Embedding Similarity Search", "LLM Similarity Search"]
)


# Add buttons
col1, col2 = st.columns(2)

with col1:
    show_titles = st.button("🔎 Show Matching Titles")

with col2:
    show_details = st.button("📄 Show Top 10 Job Details")

# Matching logic
if query:
    # matched_jobs = df[df["title"].str.contains(query, case=False, na=False)]
    method = 'Similarity Search'
    matches, scores = match_jobs(query)
    matched_jobs = match_data(matches, method)
else:
    matched_jobs = pd.DataFrame()

# If first button clicked, show job titles
if show_titles:
    if not matched_jobs.empty:
        st.subheader("✅ Matching Job Titles")
        for title in matched_jobs["job_name"].drop_duplicates().head(10):
            st.markdown(f"- {title}")
    else:
        st.warning("No matching job titles found.")

# Show full job info in table
if show_details:
    if algorithm == 'String Similarity Search':
        method = 'String Similarity'
        matches, scores = match_jobs(query)
    elif algorithm == 'Embedding Similarity Search':
        method = 'Embedding Similarity'
        matches, scores = match_jobs_word2vec(query)
    elif algorithm == 'LLM Similarity Search':
        method = 'LLM Similarity'
        matches, scores = match_jobs_llm(query)
    else:
        method = 'String Similarity'
        matches, scores = match_jobs(query)
    matched_jobs = match_data(matches, method)

    if not matched_jobs.empty:
        st.subheader("📋 Top 10 Job Details")
        
        # Prepare display table
        table = matched_jobs[["job_name", "company", "location", "joburl","method"]].head(10).copy()
        table["joburl"] = table["joburl"].apply(lambda x: f"[View Job]({x})")
        table.rename(columns={
            "job_name": "Job Title",
            "company": "Company",
            "location": "Location",
            "joburl": "Link",
            "method": "Method"
        }, inplace=True)

        st.write(table.to_markdown(index=False), unsafe_allow_html=True)  # Better formatting

        # Optional: Show as interactive table
        # st.dataframe(table, use_container_width=True)

    else:
        st.warning("No job matches to display.")
