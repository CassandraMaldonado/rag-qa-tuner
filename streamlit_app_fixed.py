import streamlit as st
import os
import json
import pickle
import numpy as np
import openai
from typing import List, Dict
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="UChicago MS-ADS Q&A Bot - Diagnostic",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 RAG System Diagnostic Tool")
st.write("Let's check what files you have and debug the loading process...")

# File structure checker
st.header("📁 File Structure Check")

save_dir = st.text_input("RAG System Directory", value="rag_system_export")

if st.button("🔍 Check Files"):
    st.write(f"Checking directory: `{save_dir}`")
    
    if os.path.exists(save_dir):
        st.success(f"✅ Directory `{save_dir}` exists!")
        
        # List all files in the directory
        files = os.listdir(save_dir)
        st.write("📋 Files found:")
        for file in files:
            file_path = os.path.join(save_dir, file)
            file_size = os.path.getsize(file_path)
            st.write(f"- `{file}` ({file_size:,} bytes)")
        
        # Check specific required files
        st.write("\n🎯 Required Files Check:")
        
        required_files = {
            'metadata.json': 'Contains system metadata',
            'chunks.pkl': 'Contains text chunks for search'
        }
        
        optional_files = {
            'faiss_index.bin': 'FAISS vector index',
            'embeddings.npy': 'Precomputed embeddings'
        }
        
        for filename, description in required_files.items():
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                st.success(f"✅ `{filename}` - {description}")
            else:
                st.error(f"❌ `{filename}` - {description} - MISSING!")
        
        for filename, description in optional_files.items():
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                st.info(f"📦 `{filename}` - {description}")
            else:
                st.warning(f"⚠️ `{filename}` - {description} - Not found (optional)")
        
    else:
        st.error(f"❌ Directory `{save_dir}` does not exist!")
        st.write("Current working directory:", os.getcwd())
        st.write("Files in current directory:", os.listdir('.'))

# Metadata inspection
st.header("📊 Metadata Inspection")

if st.button("🔍 Load and Inspect Metadata"):
    metadata_path = os.path.join(save_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.success("✅ Metadata loaded successfully!")
            st.json(metadata)
        except Exception as e:
            st.error(f"❌ Error loading metadata: {e}")
    else:
        st.error("❌ metadata.json not found!")

# Chunks inspection
st.header("📝 Chunks Inspection")

if st.button("🔍 Load and Inspect Chunks"):
    chunks_path = os.path.join(save_dir, "chunks.pkl")
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            st.success(f"✅ Chunks loaded successfully! Found {len(chunks)} chunks")
            
            # Show first few chunks
            st.write("🔍 First 3 chunks preview:")
            for i, chunk in enumerate(chunks[:3]):
                with st.expander(f"Chunk {i+1}"):
                    st.json(chunk)
            
            # Show chunk statistics
            st.write("📈 Chunk Statistics:")
            chunk_types = {}
            text_lengths = []
            
            for chunk in chunks:
                chunk_type = chunk.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                text_lengths.append(len(chunk.get('text', '')))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Chunk Types:**")
                for chunk_type, count in chunk_types.items():
                    st.write(f"- {chunk_type}: {count}")
            
            with col2:
                st.write("**Text Length Stats:**")
                st.write(f"- Average length: {np.mean(text_lengths):.0f} chars")
                st.write(f"- Min length: {min(text_lengths)} chars")
                st.write(f"- Max length: {max(text_lengths)} chars")
                
        except Exception as e:
            st.error(f"❌ Error loading chunks: {e}")
            st.write("Error details:", str(e))
    else:
        st.error("❌ chunks.pkl not found!")

# Search test
st.header("🔍 Search Test")

if st.button("🔍 Initialize Search System"):
    try:
        # Load files
        metadata_path = os.path.join(save_dir, "metadata.json")
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        
        if not os.path.exists(metadata_path) or not os.path.exists(chunks_path):
            st.error("❌ Required files missing!")
            st.stop()
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        st.success("✅ Metadata loaded")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        st.success(f"✅ Chunks loaded: {len(chunks)} total")
        
        # Create TF-IDF vectorizer
        texts = [chunk['text'] for chunk in chunks]
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        chunk_vectors = vectorizer.fit_transform(texts)
        st.success("✅ TF-IDF vectors created")
        
        # Store in session state for testing
        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.chunk_vectors = chunk_vectors
        
        st.success("🎉 Search system initialized successfully!")
        
    except Exception as e:
        st.error(f"❌ Error initializing search: {e}")
        st.write("Full error:", str(e))

# Test search functionality
if 'chunks' in st.session_state:
    st.header("🧪 Test Search")
    
    test_query = st.text_input("Enter a test query:", "tuition cost")
    
    if st.button("🔍 Test Search") and test_query:
        try:
            # Perform search
            query_vector = st.session_state.vectorizer.transform([test_query])
            similarities = cosine_similarity(query_vector, st.session_state.chunk_vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-10:][::-1]
            
            st.write(f"🎯 Search Results for: '{test_query}'")
            
            results_found = False
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:
                    results_found = True
                    chunk = st.session_state.chunks[idx]
                    score = similarities[idx]
                    
                    with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                        st.write(f"**Type:** {chunk.get('chunk_type', 'unknown')}")
                        st.write(f"**Source:** {chunk.get('source_url', 'N/A')}")
                        st.write(f"**Text:** {chunk['text'][:500]}...")
            
            if not results_found:
                st.warning("⚠️ No results found with similarity > 0")
                
        except Exception as e:
            st.error(f"❌ Search error: {e}")

# Sample query tests
st.header("📋 Sample Query Tests")

sample_queries = [
    "What are the deadlines for the in-person program?",
    "Where can I mail my official transcripts?",
    "tuition cost per course",
    "visa sponsorship",
    "MBA program application"
]

selected_query = st.selectbox("Select a sample query to test:", sample_queries)

if st.button("🧪 Test Sample Query") and 'chunks' in st.session_state:
    try:
        query_vector = st.session_state.vectorizer.transform([selected_query])
        similarities = cosine_similarity(query_vector, st.session_state.chunk_vectors).flatten()
        
        # Get top 5 results
        top_indices = similarities.argsort()[-5:][::-1]
        
        st.write(f"🎯 Top 5 Results for: '{selected_query}'")
        
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                chunk = st.session_state.chunks[idx]
                score = similarities[idx]
                
                st.write(f"**{i+1}. Score: {score:.4f}**")
                st.write(f"Type: {chunk.get('chunk_type', 'unknown')}")
                st.write(f"Text: {chunk['text'][:200]}...")
                st.write("---")
    except Exception as e:
        st.error(f"❌ Error testing query: {e}")

# Content search
st.header("🔎 Content Search")
st.write("Search for specific content in your chunks:")

search_term = st.text_input("Search for text containing:", "")

if st.button("🔍 Search Content") and search_term and 'chunks' in st.session_state:
    matches = []
    for i, chunk in enumerate(st.session_state.chunks):
        if search_term.lower() in chunk['text'].lower():
            matches.append((i, chunk))
    
    if matches:
        st.success(f"✅ Found {len(matches)} chunks containing '{search_term}'")
        for i, (idx, chunk) in enumerate(matches[:5]):  # Show first 5 matches
            with st.expander(f"Match {i+1} (Chunk {idx})"):
                st.write(f"**Type:** {chunk.get('chunk_type', 'unknown')}")
                st.write(f"**Source:** {chunk.get('source_url', 'N/A')}")
                st.write(f"**Text:** {chunk['text']}")
    else:
        st.warning(f"⚠️ No chunks found containing '{search_term}'")

# Summary
st.header("📝 Diagnostic Summary")

if st.button("📊 Generate Summary"):
    summary = []
    
    # Check files
    if os.path.exists(save_dir):
        summary.append("✅ RAG directory exists")
        
        if os.path.exists(os.path.join(save_dir, "metadata.json")):
            summary.append("✅ metadata.json found")
        else:
            summary.append("❌ metadata.json missing")
            
        if os.path.exists(os.path.join(save_dir, "chunks.pkl")):
            summary.append("✅ chunks.pkl found")
        else:
            summary.append("❌ chunks.pkl missing")
    else:
        summary.append("❌ RAG directory not found")
    
    # Check session state
    if 'chunks' in st.session_state:
        summary.append(f"✅ Search system loaded with {len(st.session_state.chunks)} chunks")
    else:
        summary.append("❌ Search system not initialized")
    
    st.write("**Diagnostic Results:**")
    for item in summary:
        st.write(item)
    
    # Recommendations
    st.write("\n**Recommendations:**")
    if "❌ RAG directory not found" in summary:
        st.write("1. 🔧 Make sure your `rag_system_export` folder is in the same directory as this script")
        st.write("2. 📁 Check the exact folder name and path")
    elif "❌ metadata.json missing" in summary or "❌ chunks.pkl missing" in summary:
        st.write("1. 🔧 Make sure all required files are in the RAG directory")
        st.write("2. 📊 Re-export your RAG system if files are missing")
    elif "❌ Search system not initialized" in summary:
        st.write("1. 🔧 Click 'Initialize Search System' button above")
        st.write("2. 🐛 Check error messages for specific issues")
    else:
        st.write("1. ✅ System looks good! Try the sample queries above")
        st.write("2. 🔧 If search isn't working, check the search test results")
