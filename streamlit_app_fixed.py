import streamlit as st
import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict
import time

# Page configuration with UChicago branding
st.set_page_config(
    page_title="UChicago MS-ADS Q&A Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UChicago theme (keeping your original styling)
st.markdown("""
<style>
    /* UChicago Colors */
    :root {
        --uchicago-maroon: #800000;
        --uchicago-dark-gray: #767676;
        --uchicago-light-gray: #D6D6CE;
        --uchicago-white: #FFFFFF;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #800000 0%, #A01010 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #D6D6CE;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F8F8;
    }
    
    /* Question suggestions */
    .question-suggestions {
        background-color: #F8F8F8;
        border-left: 4px solid #800000;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .suggestion-button {
        background-color: white;
        border: 2px solid #800000;
        color: #800000;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .suggestion-button:hover {
        background-color: #800000;
        color: white;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #800000;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .bot-message {
        background-color: #F8F8F8;
        border: 1px solid #D6D6CE;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-info {
        color: #800000;
        font-weight: bold;
    }
    
    /* Footer */
    .footer {
        background-color: #800000;
        color: white;
        padding: 1rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: 10px;
    }
    
    /* Loading animation */
    .loading {
        color: #800000;
        font-style: italic;
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        border: 1px solid #D6D6CE;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Answer box styling */
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #D6D6CE;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Source box styling */
    .source-box {
        background-color: #F8F8F8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #800000;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class SavedRAGSystem:
    """RAG system that loads from pre-saved files."""
    
    def __init__(self):
        self.chunks = []
        self.embedding_model = None
        self.vector_store = None
        self.openai_client = None
        self.is_loaded = False
        self.metadata = {}
    
    def load_system(self, save_dir="rag_system_export"):
        """Load the pre-saved RAG system."""
        try:
            if not os.path.exists(save_dir):
                st.error(f"❌ Directory '{save_dir}' not found. Please upload your saved RAG system.")
                return False
            
            with st.spinner("📂 Loading saved RAG system..."):
                # Load metadata
                metadata_path = os.path.join(save_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    st.success(f"✓ Loaded metadata: {self.metadata.get('total_chunks', 0)} chunks")
                
                # Load chunks
                chunks_path = os.path.join(save_dir, "chunks.pkl")
                if os.path.exists(chunks_path):
                    with open(chunks_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    st.success(f"✓ Loaded {len(self.chunks)} text chunks")
                else:
                    st.error("❌ chunks.pkl not found")
                    return False
                
                # Load embedding model
                model_name = self.metadata.get('embedding_model', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                st.success(f"✓ Loaded embedding model: {model_name}")
                
                # Load FAISS index
                index_path = os.path.join(save_dir, "faiss_index.bin")
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
                    st.success("✓ Loaded FAISS index")
                else:
                    st.error("❌ faiss_index.bin not found")
                    return False
                
                # Load embeddings
                embeddings_path = os.path.join(save_dir, "embeddings.npy")
                if os.path.exists(embeddings_path):
                    embeddings = np.load(embeddings_path)
                    st.success(f"✓ Loaded embeddings: {embeddings.shape}")
                else:
                    st.error("❌ embeddings.npy not found")
                    return False
                
                # Create vector store
                self.vector_store = {
                    'index': index,
                    'embeddings': embeddings,
                    'chunks': self.chunks,
                    'metadata': self.metadata
                }
                
                self.is_loaded = True
                st.success("🎉 RAG system loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            return False
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI client."""
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            # Test the connection
            self.openai_client.models.list()
            return True
        except Exception as e:
            st.error(f"❌ OpenAI setup failed: {e}")
            return False
    
    def search_chunks(self, query: str, k: int = 6):
        """Search for relevant chunks using the saved system."""
        if not self.is_loaded or not self.vector_store:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search with more results initially for better boosting
            search_k = min(k * 3, len(self.chunks))
            scores, indices = self.vector_store['index'].search(query_embedding, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    result = self.chunks[idx].copy()
                    result['semantic_score'] = float(score)
                    results.append(result)
            
            # Apply enhanced boosting (same logic as your original system)
            enhanced_results = self._apply_enhanced_boosting(query, results)
            enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return enhanced_results[:k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def _apply_enhanced_boosting(self, query, results):
        """Apply the same enhanced boosting logic from your saved system."""
        query_lower = query.lower()
        
        # Enhanced keyword mappings for better results
        keyword_boosts = {
            'deadline': ['deadline', 'date', 'apply', 'due', 'portal', 'open', 'september', 'cohort', 'filled'],
            'address': ['transcript', 'mail', 'address', 'send', 'official', 'university of chicago', 'cityfront', '455', 'suite 950'],
            'mba': ['mba', 'booth', 'joint', 'dual', 'centralized', 'full-time mba application'],
            'visa': ['visa', 'sponsorship', 'international', 'f-1', 'in-person', 'full-time', 'only the', 'eligible'],
            'tuition': ['tuition', 'cost', 'fee', 'price', '$', 'dollar', 'per course', 'total'],
            'scholarship': ['scholarship', 'financial aid', 'funding', 'grant', 'data science institute', 'alumni'],
            'contact': ['contact', 'appointment', 'advisor', 'advising', 'schedule', 'portal', 'http']
        }
        
        for result in results:
            text_lower = result['text'].lower()
            final_score = result['semantic_score']
            
            # Priority boost for key facts (micro-chunks)
            if result.get('chunk_type') == 'key_fact' or result.get('chunk_type') == 'micro':
                final_score *= 1.8
            
            # Query-specific boosting
            for category, keywords in keyword_boosts.items():
                if any(keyword in query_lower for keyword in keywords):
                    matches = sum(1 for keyword in keywords if keyword in text_lower)
                    if matches > 0:
                        # Different boost levels based on category importance
                        if category in ['deadline', 'address', 'mba', 'visa']:
                            final_score *= (1 + 0.4 * matches)  # Higher boost for specific queries
                        else:
                            final_score *= (1 + 0.2 * matches)  # Standard boost
            
            result['final_score'] = final_score
        
        return results
    
    def generate_answer(self, query: str, chunks: List[Dict]):
        """Generate enhanced answers using the same logic as your original system."""
        if not self.openai_client:
            return "❌ OpenAI client not configured"
        
        # Build enhanced context
        context_parts = []
        key_facts = [c for c in chunks if c.get('chunk_type') in ['key_fact', 'micro']]
        regular_chunks = [c for c in chunks if c.get('chunk_type') not in ['key_fact', 'micro']]
        
        if key_facts:
            context_parts.append("CRITICAL SPECIFIC INFORMATION:")
            for i, chunk in enumerate(key_facts[:5]):
                context_parts.append(f"{i+1}. {chunk['text']}")
            context_parts.append("")
        
        if regular_chunks:
            context_parts.append("ADDITIONAL CONTEXT:")
            for i, chunk in enumerate(regular_chunks[:4]):
                context_parts.append(f"Source {i+1}: {chunk['text']}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt with specific instructions (same as your original)
        prompt = f"""You are an expert assistant for the MS in Applied Data Science program at the University of Chicago.

OFFICIAL PROGRAM INFORMATION:
{context}

QUESTION: {query}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:

1. FOR DEADLINE QUESTIONS: Look for information about "application portal", "September 2025", "2026 entrance", "Events & Deadlines", "cohort filled", "portal may close early". Include ALL of these specific details if found.

2. FOR TRANSCRIPT MAILING QUESTIONS: Look for the complete mailing address including "University of Chicago", "Attention: MS in Applied Data Science Admissions", "455 N Cityfront Plaza Dr., Suite 950", "Chicago, Illinois 60611". Include the COMPLETE address.

3. FOR MBA PROGRAM QUESTIONS: Look for information about "Joint MBA/MS", "Booth", "centralized joint-application process", "Chicago Booth Full-Time MBA application", "MBA/MS in Applied Data Science as their program of interest". Include ALL Booth-specific details.

4. FOR VISA SPONSORSHIP QUESTIONS: Look for distinctions between "Online program" and "In-Person, Full-Time program", specifically "Only the In-Person, Full-Time program is Visa eligible". Include this exact distinction.

5. ALWAYS include specific details like:
   - Exact addresses with all components
   - Specific dates and years (2025, 2026)
   - Exact program names and distinctions
   - Complete application process details
   - Any qualifying statements ("may close early", "Only the In-Person")

6. If the context contains any of these specific details, you MUST include them in your answer verbatim.

Answer the question comprehensively using the exact information provided:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for UChicago's MS in Applied Data Science program. You must provide complete, exact answers including all specific details like addresses, dates, program distinctions, and qualifying statements. Never summarize or omit important specifics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.05,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if "quota" in str(e).lower():
                return "❌ OpenAI API quota exceeded. Please add credits to your account."
            elif "401" in str(e):
                return "❌ Invalid OpenAI API key."
            else:
                return f"❌ Error: {str(e)}"
    
    def ask_question(self, query: str):
        """Complete Q&A pipeline."""
        if not self.is_loaded:
            return "❌ System not loaded", []
        
        relevant_chunks = self.search_chunks(query, 6)
        
        if not relevant_chunks:
            return "❌ No relevant information found.", []
        
        answer = self.generate_answer(query, relevant_chunks)
        return answer, relevant_chunks

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def main():
    # Header (keeping your original design)
    st.markdown("""
    <div class="main-header">
        <h1>🎓 UChicago MS-ADS Q&A Bot</h1>
        <p>Your intelligent assistant for the Master's in Applied Data Science program</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Setup")
        
        # System loading section
        st.markdown("#### 📂 Load Saved System")
        save_dir = st.text_input(
            "RAG System Directory", 
            value="rag_system_export",
            help="Path to your saved RAG system folder"
        )
        
        if st.button("📥 Load System", type="primary"):
            if not st.session_state.rag_system:
                st.session_state.rag_system = SavedRAGSystem()
            
            success = st.session_state.rag_system.load_system(save_dir)
            if success:
                st.session_state.system_ready = "loaded"
        
        # API Key input
        st.markdown("#### 🔑 OpenAI Configuration")
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key to enable Q&A"
        )
        
        if api_key and st.session_state.system_ready == "loaded":
            if st.button("🔗 Connect OpenAI"):
                if st.session_state.rag_system.setup_openai(api_key):
                    st.session_state.system_ready = "ready"
                    st.success("✅ OpenAI connected!")

        # System status
        if st.session_state.system_ready == "ready":
            st.success("✅ System Fully Ready!")
            
            # System metrics (keeping your original design)
            if st.session_state.rag_system:
                rag = st.session_state.rag_system
                st.markdown("### 📊 System Info")
                
                if rag.metadata:
                    st.metric("Pages Scraped", rag.metadata.get('total_pages', 'N/A'))
                    st.metric("Text Chunks", len(rag.chunks))
                    st.metric("Vector Dimensions", rag.metadata.get('vector_dimension', 'N/A'))
                
        elif st.session_state.system_ready == "loaded":
            st.warning("⚠️ System loaded - need OpenAI key")
        else:
            st.warning("⚠️ System not loaded")

        # Quick tips (keeping your original content)
        st.markdown("### 💡 Usage Tips")
        st.markdown("""
        - Use specific questions for better results
        - Ask about tuition, scholarships, deadlines, etc.
        - The system uses pre-trained data from UChicago website
        """)

        # Contact info (keeping your original design)
        st.markdown("### 📞 DSI Contact")
        st.markdown("""
        **In-Person Program:**  
        Jose Alvarado  
        Associate Director
        
        **Online Program:**  
        Patrick Vonesh  
        Senior Assistant Director
        """)

    # Main content area (keeping your original layout)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Optimized question suggestions (keeping your original questions)
        st.markdown("""
        <div class="question-suggestions">
            <strong>🎯 Optimized Questions (Click to use):</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Your original optimized questions
        optimized_questions = [
            "What are the deadlines for the in-person program?",
            "Where can I mail my official transcripts?",
            "How do I apply to the MBA/MS program?",
            "Does the Master's in Applied Data Science Online program provide visa sponsorship?",
            "What is the exact tuition cost per course and total for the program?",
            "What scholarships are available including Data Science Institute Scholarship?",
            "What are the minimum TOEFL and IELTS English Language Requirements?",
            "How many courses must you complete to earn the Master's degree?",
            "Is the MS in Applied Data Science program STEM/OPT eligible?",
            "How do I schedule an advising appointment?"
        ]
        
        # Create clickable buttons for each question
        selected_question = None
        cols = st.columns(2)
        for i, question in enumerate(optimized_questions):
            with cols[i % 2]:
                if st.button(f"Q{i+1}: {question[:40]}...", key=f"q_{i}", help=question):
                    selected_question = question

        # Question input
        if selected_question:
            user_question = st.text_area(
                "Your Question:", 
                value=selected_question,
                height=100,
                key="question_input"
            )
        else:
            user_question = st.text_area(
                "Your Question:", 
                placeholder="Type your question about the MS in Applied Data Science program...",
                height=100,
                key="question_input"
            )

        # Ask button
        if st.button("🔍 Get Answer", type="primary", disabled=(st.session_state.system_ready != "ready")):
            if user_question and st.session_state.rag_system:
                with st.spinner("🔍 Searching and generating answer..."):
                    answer, sources = st.session_state.rag_system.ask_question(user_question)
                    
                    # Display answer in styled box
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 📝 Answer")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources[:3]):
                                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                st.markdown(f"**Source {i+1}** (Relevance: {source.get('final_score', 0):.3f})")
                                st.markdown(f"**Type:** {source.get('chunk_type', 'regular')}")
                                st.markdown(f"**URL:** {source.get('source_url', 'N/A')}")
                                st.markdown(f"**Content:** {source['text'][:300]}...")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'sources': sources,
                        'timestamp': time.strftime('%H:%M:%S')
                    })

        # Display chat history (keeping your original design)
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:80]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'])
                    
                    if chat['sources']:
                        st.markdown("**Sources:**")
                        for j, source in enumerate(chat['sources'][:3]):
                            relevance = source.get('final_score', source.get('semantic_score', 0))
                            st.markdown(f"- **Source {j+1}** (Relevance: {relevance:.3f}): [{source.get('title', 'Unknown')}]({source.get('source_url', '#')})")

    # Right column (keeping your original design)
    with col2:
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_ready == "ready" and st.session_state.rag_system:
            rag = st.session_state.rag_system
            
            # Status metrics (keeping your original styling)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System Status", "🟢 Online")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Sources", f"{rag.metadata.get('total_pages', 'N/A')} pages")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Knowledge Chunks", f"{len(rag.chunks)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System Status", "🔴 Offline")
            st.markdown('</div>', unsafe_allow_html=True)

        # Quick actions (keeping your original design)
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.button("📥 Download Chat History"):
            if st.session_state.chat_history:
                chat_data = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=chat_data,
                    file_name=f"uchicago_msads_chat_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        # Help section (keeping your original content)
        st.markdown("### ❓ Need Help?")
        st.markdown("""
        **Setup Steps:**
        1. Upload your `rag_system_export` folder
        2. Click "Load System"
        3. Enter your OpenAI API key
        4. Click "Connect OpenAI"
        5. Start asking questions!
        
        **Best Practices:**
        - Be specific in your questions
        - Use the optimized question templates
        - Ask one question at a time
        """)

    # Footer (keeping your original design)
    st.markdown("""
    <div class="footer">
        <p>🎓 University of Chicago Data Science Institute | MS in Applied Data Science Program</p>
        <p>Powered by RAG (Retrieval-Augmented Generation) Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()