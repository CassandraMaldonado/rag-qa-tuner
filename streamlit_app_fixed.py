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
    page_title="UChicago MS-ADS Q&A Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
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
    
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #D6D6CE;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .source-box {
        background-color: #F8F8F8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #800000;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #D6D6CE;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class ImprovedRAGSystem:
    """Improved RAG system with better search and guaranteed LLM answers."""
    
    def __init__(self):
        self.chunks = []
        self.openai_client = None
        self.is_loaded = False
        self.metadata = {}
        self.vectorizer = None
        self.chunk_vectors = None
    
    def load_system(self, save_dir="rag_system_export"):
        """Load the RAG system with detailed feedback."""
        try:
            if not os.path.exists(save_dir):
                st.error(f"❌ Directory '{save_dir}' not found.")
                return False
            
            with st.spinner("📂 Loading RAG system..."):
                # Load metadata
                metadata_path = os.path.join(save_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    st.success(f"✓ Loaded metadata: {self.metadata.get('total_chunks', 0)} chunks expected")
                
                # Load chunks
                chunks_path = os.path.join(save_dir, "chunks.pkl")
                if os.path.exists(chunks_path):
                    with open(chunks_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    st.success(f"✓ Loaded {len(self.chunks)} text chunks")
                    
                    # Show chunk type distribution
                    chunk_types = {}
                    for chunk in self.chunks:
                        chunk_type = chunk.get('chunk_type', 'unknown')
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    st.info(f"📊 Chunk types: {dict(chunk_types)}")
                else:
                    st.error("❌ chunks.pkl not found")
                    return False
                
                # Create improved TF-IDF vectors
                with st.spinner("🔍 Creating search index..."):
                    texts = [chunk['text'] for chunk in self.chunks]
                    self.vectorizer = TfidfVectorizer(
                        max_features=10000,
                        stop_words='english',
                        ngram_range=(1, 3),
                        lowercase=True,
                        min_df=1,
                        max_df=0.95
                    )
                    self.chunk_vectors = self.vectorizer.fit_transform(texts)
                    st.success("✓ Created enhanced TF-IDF search index")
                
                self.is_loaded = True
                st.success("🎉 RAG system loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            return False
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI client with validation."""
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            response = self.openai_client.models.list()
            st.success("✓ OpenAI API connected successfully")
            return True
        except Exception as e:
            st.error(f"❌ OpenAI setup failed: {e}")
            return False
    
    def search_chunks(self, query: str, k: int = 8):
        """Enhanced search with lower threshold and better boosting."""
        if not self.is_loaded or not self.vectorizer:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            top_indices = similarities.argsort()[-k*3:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    result = self.chunks[idx].copy()
                    result['semantic_score'] = float(similarities[idx])
                    results.append(result)
            
            enhanced_results = self._apply_enhanced_boosting(query, results)
            enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return enhanced_results[:k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def _apply_enhanced_boosting(self, query, results):
        """Apply comprehensive boosting for UChicago MS-ADS queries."""
        query_lower = query.lower()
        
        # Keywords for boosting - FIXED VERSION
        tuition_words = ['tuition', 'cost', 'costs', 'fee', 'fees', 'price', 'pricing', 'dollar', 'dollars', 'course', 'total', 'financial', 'payment', 'expense', 'expensive', 'money', 'pay', 'amount', 'credit', 'unit']
        scholarship_words = ['scholarship', 'scholarships', 'aid', 'funding', 'grant', 'grants', 'institute', 'alumni', 'merit', 'assistance', 'support', 'award', 'awards', 'fellowship', 'fellowships']
        deadline_words = ['deadline', 'deadlines', 'date', 'dates', 'apply', 'application', 'due', 'portal', 'september', 'cohort', 'filled', 'open', '2025', '2026', 'events', 'when', 'timeline', 'schedule', 'admission', 'round', 'early', 'late', 'close', 'closing']
        
        for result in results:
            text_lower = result['text'].lower()
            final_score = result['semantic_score']
            
            # Boost for key facts
            chunk_type = result.get('chunk_type', 'regular')
            if chunk_type in ['key_fact', 'micro']:
                final_score *= 3.0
            elif chunk_type == 'important':
                final_score *= 2.0
            
            # Apply targeted boosting for the 3 problematic queries
            # Tuition queries
            tuition_query_match = any(word in query_lower for word in tuition_words)
            tuition_text_match = any(word in text_lower for word in tuition_words)
            if tuition_query_match and tuition_text_match:
                final_score *= 3.5
            
            # Scholarship queries  
            scholarship_query_match = any(word in query_lower for word in scholarship_words)
            scholarship_text_match = any(word in text_lower for word in scholarship_words)
            if scholarship_query_match and scholarship_text_match:
                final_score *= 3.5
            
            # Deadline queries
            deadline_query_match = any(word in query_lower for word in deadline_words)
            deadline_text_match = any(word in text_lower for word in deadline_words)
            if deadline_query_match and deadline_text_match:
                final_score *= 3.5
            
            # Other category boosts (unchanged)
            if 'transcript' in query_lower and 'transcript' in text_lower:
                final_score *= 2.5
            if 'mba' in query_lower and 'mba' in text_lower:
                final_score *= 2.5
            if 'visa' in query_lower and 'visa' in text_lower:
                final_score *= 2.5
            
            # Exact phrase matching
            if 'tuition cost' in query_lower and 'tuition cost' in text_lower:
                final_score *= 3.0
            if 'scholarship available' in query_lower and 'scholarship' in text_lower:
                final_score *= 3.0
            if 'deadline' in query_lower and 'deadline' in text_lower:
                final_score *= 3.0
            
            # Length adjustment
            text_length = len(result['text'])
            if text_length < 50:
                final_score *= 0.7
            elif text_length > 200:
                final_score *= 1.1
            
            result['final_score'] = final_score
        
        return results
    
    def generate_answer(self, query: str, chunks: List[Dict]):
        """Generate comprehensive LLM answers."""
        if not self.openai_client:
            return "❌ OpenAI client not configured. Please add your API key."
        
        if not chunks:
            return "❌ No relevant information found in the knowledge base."
        
        # Build context
        context_parts = []
        key_facts = [c for c in chunks if c.get('chunk_type') in ['key_fact', 'micro']]
        important_chunks = [c for c in chunks if c.get('chunk_type') == 'important']
        regular_chunks = [c for c in chunks if c.get('chunk_type') not in ['key_fact', 'micro', 'important']]
        
        if key_facts:
            context_parts.append("CRITICAL SPECIFIC INFORMATION:")
            for i, chunk in enumerate(key_facts[:6]):
                context_parts.append(f"{i+1}. {chunk['text']}")
            context_parts.append("")
        
        if important_chunks:
            context_parts.append("IMPORTANT DETAILS:")
            for i, chunk in enumerate(important_chunks[:4]):
                context_parts.append(f"• {chunk['text']}")
            context_parts.append("")
        
        if regular_chunks:
            context_parts.append("ADDITIONAL CONTEXT:")
            for i, chunk in enumerate(regular_chunks[:4]):
                context_parts.append(f"Source {i+1}: {chunk['text']}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt
        prompt = f"""You are an expert assistant for the MS in Applied Data Science program at the University of Chicago. You must provide complete, accurate, and helpful answers based on the official program information provided.

OFFICIAL PROGRAM INFORMATION:
{context}

STUDENT QUESTION: {query}

CRITICAL INSTRUCTIONS - Follow these exactly:

1. FOR DEADLINE QUESTIONS: Include ALL specific details about application portal opening, exact dates (September 2025, 2026 entrance), "Events & Deadlines", cohort capacity, and any warnings about early closure.

2. FOR TRANSCRIPT MAILING QUESTIONS: Provide the COMPLETE mailing address including all components: "University of Chicago", department name, street address, suite number, city, state, and ZIP code.

3. FOR MBA/JOINT PROGRAM QUESTIONS: Include details about Joint MBA/MS programs, Chicago Booth requirements, centralized application processes, and specific application instructions.

4. FOR VISA SPONSORSHIP QUESTIONS: Clearly distinguish between program types and their visa eligibility. Be specific about which programs do/don't provide sponsorship.

5. FOR TUITION AND COST QUESTIONS: Include specific dollar amounts, per-course costs, total program costs, and any additional fees mentioned.

6. FOR SCHOLARSHIP QUESTIONS: Include specific scholarship names, amounts, eligibility criteria, and application processes.

7. ALWAYS:
   - Use exact information from the context
   - Include specific details like addresses, dates, costs, and requirements
   - Provide complete answers, not summaries
   - If multiple pieces of information relate to the question, include all relevant details
   - Be helpful and informative while staying accurate

Answer the student's question comprehensively using the exact information provided:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert assistant for UChicago's MS in Applied Data Science program. Provide complete, accurate answers with all specific details from the context. Never summarize or omit important information like addresses, dates, costs, or requirements."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "billing" in error_msg:
                return "❌ OpenAI API quota exceeded. Please add credits to your account or check your billing."
            elif "401" in error_msg or "authentication" in error_msg:
                return "❌ Invalid OpenAI API key. Please check your API key."
            elif "rate_limit" in error_msg:
                return "❌ Rate limit exceeded. Please wait a moment and try again."
            else:
                return f"❌ Error generating answer: {str(e)}"
    
    def ask_question(self, query: str):
        """Complete Q&A pipeline with guaranteed LLM response."""
        if not self.is_loaded:
            return "❌ System not loaded", []
        
        relevant_chunks = self.search_chunks(query, 8)
        
        if relevant_chunks:
            answer = self.generate_answer(query, relevant_chunks)
        else:
            answer = f"I couldn't find specific information about '{query}' in the MS in Applied Data Science knowledge base. This might be because:\n\n1. The information isn't available in the current data\n2. Try rephrasing your question\n3. Contact the program directly for the most current information\n\nProgram contacts:\n- In-Person Program: Jose Alvarado, Associate Director\n- Online Program: Patrick Vonesh, Senior Assistant Director"
            relevant_chunks = []
        
        return answer, relevant_chunks

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def main():
    # Header with UChicago branding
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 30px;">
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/uchicago_logo.png" 
                 alt="University of Chicago" style="height: 120px; background-color: white; padding: 15px; border-radius: 10px;">
            <div style="text-align: center;">
                <h1 style="margin: 0; color: white;">MS in Applied Data Science</h1>
                <p style="margin: 5px 0; color: #D6D6CE;">Intelligent Q&A Assistant</p>
                <p style="margin: 0; color: #D6D6CE; font-size: 0.9rem;">University of Chicago Data Science Institute</p>
            </div>
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/dsi_logo.png" 
                 alt="Data Science Institute" style="height: 100px; background-color: white; padding: 15px; border-radius: 10px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Setup")
        
        # System loading
        st.markdown("#### 📂 Load RAG System")
        save_dir = st.text_input(
            "RAG System Directory", 
            value="rag_system_export",
            help="Path to your saved RAG system folder"
        )
        
        if st.button("📥 Load System", type="primary"):
            st.session_state.rag_system = ImprovedRAGSystem()
            success = st.session_state.rag_system.load_system(save_dir)
            if success:
                st.session_state.system_ready = "loaded"
        
        # OpenAI setup
        st.markdown("#### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key and st.session_state.system_ready == "loaded":
            if st.button("🔗 Connect OpenAI"):
                if st.session_state.rag_system.setup_openai(api_key):
                    st.session_state.system_ready = "ready"

        # System status
        if st.session_state.system_ready == "ready":
            st.success("✅ System Ready!")
            
            if st.session_state.rag_system:
                rag = st.session_state.rag_system
                st.markdown("### 📊 System Info")
                st.metric("Text Chunks", len(rag.chunks))
                if rag.metadata:
                    st.metric("Pages Scraped", rag.metadata.get('total_pages', 'N/A'))
                st.info("🚀 Enhanced TF-IDF + LLM Mode")
                
        elif st.session_state.system_ready == "loaded":
            st.warning("⚠️ Add OpenAI API key to enable Q&A")
        else:
            st.warning("⚠️ Load your RAG system first")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Question input
        user_question = st.text_area(
            "Your Question:", 
            placeholder="Ask anything about the MS in Applied Data Science program...",
            height=100
        )

        # Ask button
        if st.button("🔍 Get Answer", type="primary", disabled=(st.session_state.system_ready != "ready")):
            if user_question and st.session_state.rag_system:
                with st.spinner("🤖 Generating answer..."):
                    start_time = time.time()
                    answer, sources = st.session_state.rag_system.ask_question(user_question)
                    response_time = time.time() - start_time
                    
                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 📝 Answer")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display search info
                    if sources:
                        st.info(f"📊 Found {len(sources)} relevant sources (response time: {response_time:.1f}s)")
                        
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(sources[:5]):
                                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                st.markdown(f"**Source {i+1}** (Score: {source.get('final_score', 0):.4f})")
                                st.markdown(f"**Type:** {source.get('chunk_type', 'regular')}")
                                st.markdown(f"**Text:** {source['text'][:400]}...")
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("📊 No specific sources found - generated general response")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'sources': len(sources) if sources else 0,
                        'timestamp': time.strftime('%H:%M:%S')
                    })

        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Recent Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                with st.expander(f"Q: {chat['question'][:60]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'][:500] + ("..." if len(chat['answer']) > 500 else ""))

    # Right column
    with col2:
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_ready == "ready":
            st.metric("Status", "🟢 Online")
            st.metric("Questions Asked", len(st.session_state.chat_history))
        else:
            st.metric("Status", "🔴 Setup Required")

        # Quick actions
        st.markdown("### ⚡ Actions")
        if st.button("🔄 Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # Footer with official UChicago branding
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background-color: #800000; color: white; text-align: center; border-radius: 10px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/uchicago_logo.png" 
                 alt="UChicago" style="height: 60px; background-color: white; padding: 8px; border-radius: 6px;">
            <span style="font-weight: bold;">University of Chicago</span>
        </div>
        <p style="margin: 5px 0;">Data Science Institute | MS in Applied Data Science</p>
        <p style="margin: 0; font-size: 0.9rem; color: #D6D6CE;">Official Program Information Assistant</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
