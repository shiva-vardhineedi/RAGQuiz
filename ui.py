import streamlit as st
from streamlit_lottie import st_lottie
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Function to load Lottie animations from URL
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Set page configuration for a clean, wide layout
st.set_page_config(page_title="Document Search", layout="wide")

# Load animations
search_animation = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_UJNc2t.json")
loading_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# Load FAISS index and embedding model
@st.cache_resource
def load_faiss_index():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return faiss_index, embedding_model

# Main title and description
st.title("📄 Document Search & Retrieval")
st.markdown("Welcome to the document retrieval system. Enter a query to search through the document collection and retrieve the most relevant sections.")

# Sidebar for user input
with st.sidebar:
    st.header("🔍 Search Query")
    query = st.text_input("Type your search query here")

# Show the search animation if no query is submitted yet
if not query:
    st_lottie(search_animation, height=300, key="search-placeholder")

# Process the query and show results once the search button is clicked
if st.sidebar.button("Run Search"):
    if query:
        # Initialize progress bar
        progress_bar = st.progress(0)

        # Display loading animation during the search
        with st.spinner("Searching... Please wait"):
            st_lottie(loading_animation, height=200, key="loading-animation")

            # Simulate progress (this part can be customized for longer queries)
            for percent_complete in range(100):
                progress_bar.progress(percent_complete + 1)

            # Load FAISS index and perform the search
            faiss_index, embedding_model = load_faiss_index()
            results = faiss_index.similarity_search(query, k=5)

        # Clean result display section
        st.header("🔎 Top 5 Search Results")
        if results:
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1}"):
                    st.write(f"**Snippet:** {result.page_content[:500]}...")  # Preview first 500 characters
                    if result.metadata:
                        st.write(f"**Metadata:** {result.metadata}")
                    st.write("---")
        else:
            st.warning("No results found for your query. Please try a different search.")

    else:
        st.error("Please enter a query to perform the search.")

# Footer with a professional touch
st.markdown("---")
st.markdown("🗂 **About this App**")
st.markdown(
    "This document search application uses **FAISS** indexing to perform fast similarity-based searches "
    "across a document collection. The system allows users to input a search query and retrieves the "
    "most relevant document snippets based on semantic similarity."
)
