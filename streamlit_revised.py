import streamlit as st
import os
import json
import pickle
import numpy as np
import openai
from typing import List, Dict
import time
import datetime
import re
import streamlit.components.v1 as components

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

# Config.
st.set_page_config(
    page_title="UChicago MS-ADS Q&A Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    .speech-controls {
        display: flex;
        gap: 10px;
        align-items: center;
        margin: 10px 0;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .speech-button {
        background-color: #800000;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .speech-button:hover {
        background-color: #A01010;
        transform: scale(1.05);
    }
    
    .speech-button.recording {
        background-color: #dc3545;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .speech-status {
        font-size: 14px;
        color: #6c757d;
        margin-left: 10px;
    }
    
    .speech-transcript {
        flex-grow: 1;
        padding: 8px 12px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        background-color: white;
        min-height: 20px;
        font-style: italic;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# Speech-to-text.
def speech_to_text_component():
    
    speech_component = """
    <div class="speech-controls">
        <button id="speechButton" class="speech-button" onclick="toggleSpeech()">
            🎤
        </button>
        <div id="transcript" class="speech-transcript">Click the microphone to start speaking...</div>
        <div id="status" class="speech-status">Ready</div>
        <textarea id="speechOutput" style="width: 100%; height: 60px; margin-top: 10px; padding: 5px; border: 1px solid #ccc; border-radius: 4px;" placeholder="Your speech will appear here..."></textarea>
        <button onclick="copyToClipboard()" style="margin-top: 5px; padding: 5px 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">📋 Copy Text</button>
    </div>

    <script>
    let recognition = null;
    let isRecording = false;
    let currentTranscript = '';
    
    // Check if browser supports speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 1;
        
        recognition.onstart = function() {
            document.getElementById('status').textContent = 'Listening...';
            document.getElementById('speechButton').classList.add('recording');
            document.getElementById('transcript').textContent = 'Speak now...';
        };
        
        recognition.onresult = function(event) {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            currentTranscript = finalTranscript || interimTranscript;
            const transcriptDiv = document.getElementById('transcript');
            transcriptDiv.textContent = currentTranscript;
            
            // Put the transcript in the textarea immediately
            if (finalTranscript.trim()) {
                document.getElementById('speechOutput').value = finalTranscript.trim();
                document.getElementById('status').textContent = 'Speech captured! Copy the text below and paste it in the question box.';
            }
        };
        
        recognition.onerror = function(event) {
            document.getElementById('status').textContent = 'Error: ' + event.error;
            document.getElementById('speechButton').classList.remove('recording');
            isRecording = false;
        };
        
        recognition.onend = function() {
            document.getElementById('speechButton').classList.remove('recording');
            isRecording = false;
            if (currentTranscript.trim()) {
                document.getElementById('status').textContent = 'Done! Copy the text and paste it below.';
            } else {
                document.getElementById('status').textContent = 'Ready';
            }
        };
    } else {
        document.getElementById('status').textContent = 'Speech recognition not supported in this browser';
        document.getElementById('speechButton').disabled = true;
    }
    
    function toggleSpeech() {
        if (!recognition) {
            alert('Speech recognition is not supported in your browser. Try using Chrome, Edge, or Safari.');
            return;
        }
        
        if (isRecording) {
            recognition.stop();
            isRecording = false;
        } else {
            // Clear previous transcript
            document.getElementById('speechOutput').value = '';
            currentTranscript = '';
            recognition.start();
            isRecording = true;
        }
    }
    
    function copyToClipboard() {
        const textarea = document.getElementById('speechOutput');
        if (textarea.value.trim()) {
            textarea.select();
            textarea.setSelectionRange(0, 99999); // For mobile devices
            
            try {
                document.execCommand('copy');
                document.getElementById('status').textContent = '✓ Copied! Now paste it in the question box below.';
            } catch (err) {
                document.getElementById('status').textContent = 'Please manually copy the text from the box above.';
            }
        } else {
            document.getElementById('status').textContent = 'No text to copy. Please speak first.';
        }
    }
    
    // Auto-select text when clicked
    document.addEventListener('DOMContentLoaded', function() {
        const textarea = document.getElementById('speechOutput');
        if (textarea) {
            textarea.addEventListener('click', function() {
                this.select();
            });
        }
    });
    </script>
    """
    
    return components.html(speech_component, height=200)

# RAG.
class Ragsystem:
    
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.openai_client = None
        self.retriever = None
        self.is_loaded = False
        
        # OpenAI Embedding Model
        self.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    def setup_openai(self, api_key: str):
        """Initialize OpenAI client and embeddings"""
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.embedding_model = OpenAIEmbeddings(model=self.OPENAI_EMBEDDING_MODEL, api_key=api_key)
            # Test connection
            response = self.openai_client.models.list()
            st.success("✓ OpenAI API connected.")
            return True
        except Exception as e:
            st.error(f"OpenAI setup failed: {e}.")
            return False

    def connect_zilliz(self):
        """Connect to existing Zilliz Cloud collection using environment variables"""
        try:
            if not self.embedding_model:
                st.error("OpenAI embeddings not initialized. Setup OpenAI first.")
                return False
            
            # Get Zilliz credentials from environment variables
            uri = os.getenv('ZILLIZ_URI')
            token = os.getenv('ZILLIZ_TOKEN')
            collection_name = os.getenv('ZILLIZ_COLLECTION_NAME')
            
            if not uri or not token or not collection_name:
                st.error("Missing Zilliz environment variables. Please set ZILLIZ_URI, ZILLIZ_TOKEN, and ZILLIZ_COLLECTION_NAME.")
                return False
                
            with st.spinner("Connecting to Zilliz Cloud..."):
                self.vector_store = Milvus(
                    embedding_function=self.embedding_model,
                    connection_args={
                        "uri": uri,
                        "token": token,
                        "secure": True,
                    },
                    collection_name=collection_name
                )
                
                # Create retriever
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 8}
                )
                
                self.is_loaded = True
                st.success("✓ Connected to Zilliz Cloud vector store.")
                return True
                
        except Exception as e:
            st.error(f"Zilliz connection failed: {e}")
            return False

    def apply_keyword_boosting(self, query, results):
        """Apply keyword boosting to search results"""
        query_lower = query.lower()
        
        boost_categories = {
            'deadline_application': {
                'keywords': ['deadline', 'date', 'apply', 'application', 'due', 'portal', 'september', 'cohort', 'filled', 'open', '2025', '2026', 'events', 'deadlines'],
                'boost': 2.5
            },
            'transcript_address': {
                'keywords': ['transcript', 'mail', 'address', 'send', 'official', 'university of chicago', 'cityfront', '455', 'suite', 'chicago', 'illinois', '60611', 'graham school'],
                'boost': 2.5
            },
            'mba_joint': {
                'keywords': ['mba', 'booth', 'joint', 'dual', 'centralized', 'full-time mba', 'application process', 'chicago booth'],
                'boost': 2.5
            },
            'visa_sponsorship': {
                'keywords': ['visa', 'sponsorship', 'f-1', 'international', 'in-person', 'full-time', 'eligible', 'only the', 'program provides'],
                'boost': 2.5
            },
            'tuition_cost': {
                'keywords': ['tuition', 'cost', 'fee', 'price', 'dollar', 'per course', 'total', 'financial', 'payment', 'expense'],
                'boost': 2.0
            },
            'scholarship_aid': {
                'keywords': ['scholarship', 'financial aid', 'funding', 'grant', 'data science institute', 'alumni', 'merit', 'need-based'],
                'boost': 2.0
            },
            'requirements': {
                'keywords': ['requirement', 'toefl', 'ielts', 'english', 'language', 'minimum', 'gpa', 'prerequisite'],
                'boost': 1.8
            },
            'program_structure': {
                'keywords': ['courses', 'credits', 'degree', 'complete', 'graduation', 'curriculum', 'stem', 'opt'],
                'boost': 1.8
            },
            'contact_advising': {
                'keywords': ['contact', 'appointment', 'advisor', 'schedule', 'jose', 'patrick', 'alvarado', 'vonesh', 'advising'],
                'boost': 1.5
            }
        }
        
        for result in results:
            text_lower = result['text'].lower()
            final_score = result.get('semantic_score', 1.0)
            
            # Boost for micro chunks and key facts
            if 'KEY ' in result['text']:
                final_score *= 3.0
            elif result.get('chunk_type') == 'important':
                final_score *= 2.0
            
            # Apply category boosting
            for category, config in boost_categories.items():
                query_matches = sum(1 for keyword in config['keywords'] if keyword in query_lower)
                if query_matches > 0:
                    text_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
                    if text_matches > 0:
                        boost_factor = config['boost'] * (1 + 0.2 * text_matches) * (1 + 0.1 * query_matches)
                        final_score *= boost_factor
            
            # Exact phrase matching
            exact_phrases = {
                'application portal': 2.0,
                'events & deadlines': 2.0,
                'data science institute scholarship': 2.5,
                'ms in applied data science alumni scholarship': 2.5,
                'university of chicago': 1.5,
                'chicago booth': 2.0,
                'only the in-person': 2.5,
                'full-time program is visa eligible': 2.5,
                '455 n cityfront plaza': 2.5,
                'graham school': 1.8,
                'per course': 2.0,
                'total cost': 2.0
            }
            
            for phrase, boost in exact_phrases.items():
                if phrase in query_lower and phrase in text_lower:
                    final_score *= boost
            
            # Length penalty for very short chunks
            text_length = len(result['text'])
            if text_length < 50:
                final_score *= 0.7
            elif text_length > 200:
                final_score *= 1.1
            
            result['final_score'] = final_score
        
        return results

    def search_chunks(self, query: str, k: int = 8):
        """Search chunks using Zilliz Cloud vector store"""
        if not self.retriever:
            return []
        
        try:
            # Use LangChain retriever
            langchain_docs = self.retriever.invoke(query)
            
            # Convert to dictionary format
            results = []
            for doc in langchain_docs:
                result_dict = {
                    'text': doc.page_content,
                    'source_url': doc.metadata.get('source_url', ''),
                    'title': doc.metadata.get('title', ''),
                    'length': doc.metadata.get('length', 0),
                    'chunk_type': doc.metadata.get('chunk_type', 'regular'),
                    'semantic_score': 1.0
                }
                results.append(result_dict)
            
            # Apply boosting
            enhanced_results = self.apply_keyword_boosting(query, results)
            enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return enhanced_results[:k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

    def generate_answer(self, query: str, chunks: List[Dict]):
        """Generate answer using OpenAI with retrieved chunks"""
        if not self.openai_client:
            return "OpenAI client not configured."
        
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        # Build context with priority
        context_parts = []
        key_facts = [c for c in chunks if c.get('chunk_type') in ['key_fact', 'micro'] or 'KEY ' in c['text']]
        important_chunks = [c for c in chunks if c.get('chunk_type') == 'important']
        regular_chunks = [c for c in chunks if c.get('chunk_type') not in ['key_fact', 'micro', 'important'] and 'KEY ' not in c['text']]
        
        # Prioritize key facts and important information
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
                return "OpenAI API quota exceeded. Please add credits to your account or check your billing."
            elif "401" in error_msg or "authentication" in error_msg:
                return "Invalid OpenAI API key. Please check your API key."
            elif "rate_limit" in error_msg:
                return "Rate limit exceeded. Please wait a moment and try again."
            else:
                return f"Error generating answer: {str(e)}"

    def ask_question(self, query: str):
        """Pipeline for questions and answers"""
        if not self.is_loaded:
            return "System not loaded", []
        
        # Search for chunks
        relevant_chunks = self.search_chunks(query, 8)
        
        # Generate answer
        if relevant_chunks:
            answer = self.generate_answer(query, relevant_chunks)
        else:
            answer = f"I couldn't find specific information about '{query}' in the MS in Applied Data Science knowledge base. This might be because:\n\n1. The information isn't available in the current data\n2. Try rephrasing your question\n3. Contact the program directly for the most current information\n\nProgram contacts:\n- In-Person Program: Jose Alvarado, Associate Director\n- Online Program: Patrick Vonesh, Senior Assistant Director."
            relevant_chunks = []
        
        return answer, relevant_chunks

# Session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'speech_transcript' not in st.session_state:
    st.session_state.speech_transcript = ""

def main():
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
        
        # OpenAI setup
        st.markdown("#### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key and st.button("🔗 Setup OpenAI"):
            st.session_state.rag_system = Ragsystem()
            if st.session_state.rag_system.setup_openai(api_key):
                # Automatically connect to Zilliz after OpenAI setup
                if st.session_state.rag_system.connect_zilliz():
                    st.session_state.system_ready = "ready"
                else:
                    st.session_state.system_ready = "openai_ready"
        
        # Zilliz connection (automatic via environment variables)
        if st.session_state.system_ready == "openai_ready":
            st.markdown("#### ☁️ Zilliz Cloud Status")
            st.warning("Zilliz connection failed. Check environment variables.")
            st.info("Required environment variables:")
            st.code("""
ZILLIZ_URI=your_zilliz_uri
ZILLIZ_TOKEN=your_zilliz_token  
ZILLIZ_COLLECTION_NAME=your_collection_name
            """)
            
            if st.button("🔄 Retry Zilliz Connection"):
                if st.session_state.rag_system.connect_zilliz():
                    st.session_state.system_ready = "ready"

        # System status
        if st.session_state.system_ready == "ready":
            st.success("✓ System Ready")
            st.info("Zilliz Cloud + OpenAI Mode")
        elif st.session_state.system_ready == "openai_ready":
            st.warning("Zilliz Cloud connection failed")
        else:
            st.warning("Setup OpenAI API key first")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Speech-to-Text Component
        st.markdown("#### 🎤 Voice Question")
        speech_to_text_component()
        
        st.info("**Simple Speech Workflow:** 1) Click the microphone and speak, 2) Click 'Copy Text' button, 3) Paste in the question box below.")
        
        # Question input
        user_question = st.text_area(
            "Your Question:", 
            value=st.session_state.get('speech_transcript', ''),
            placeholder="Ask anything about the MS in Applied Data Science program or speak above and paste the text.",
            height=100,
            key="question_input"
        )
        
        # Update session state
        if user_question != st.session_state.get('speech_transcript', ''):
            st.session_state.speech_transcript = user_question
        
        # Control buttons
        col_clear, col_ask = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear"):
                st.session_state.speech_transcript = ""
                st.rerun()
        
        with col_ask:
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
                            st.info("No specific sources found. The answer was generated based on available knowledge base.")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': answer,
                            'sources': len(sources) if sources else 0,
                            'timestamp': time.strftime('%H:%M:%S'),
                            'input_method': 'voice' if st.session_state.speech_transcript else 'text'
                        })
                        
                        # Clear transcript
                        st.session_state.speech_transcript = ""

        # Voice instructions
        with st.expander("🎤 How to use Voice Input", expanded=False):
            st.markdown("""
            1. Click the 🎤 microphone button to start recording.
            2. Your speech will appear in the text box below the microphone.
            3. Click the "📋 Copy Text" button to copy the question.
            4. Paste the text into the "Your Question" box below.
            5. Click "Get Answer" to process your question.
            """)

        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Recent Questions")
            for chat in reversed(st.session_state.chat_history[-3:]):
                input_icon = "🎤" if chat.get('input_method') == 'voice' else "⌨️"
                with st.expander(f"{input_icon} Q: {chat['question'][:60]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'][:500] + ("..." if len(chat['answer']) > 500 else ""))

    with col2:
        st.markdown("### System Status")
        
        if st.session_state.system_ready == "ready":
            st.metric("Status", "🟢 Online")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            
            # Voice input status
            if st.session_state.chat_history:
                voice_questions = sum(1 for chat in st.session_state.chat_history if chat.get('input_method') == 'voice')
                st.metric("Voice Questions", f"{voice_questions} 🎤")
        else:
            st.metric("Status", "🔴 Setup Required")

        # Quick actions
        st.markdown("### Actions")
        if st.button("🔄 Clear History"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.button("🎤 Clear Voice"):
            st.session_state.speech_transcript = ""
            st.rerun()
            
        # End session
        st.markdown("---")
        st.markdown("### Session Management")
        
        if st.button("🔚 End Session", type="secondary", help="Clear all data and reset the application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Confirmation message
            st.success("✓ Session ended.")
            st.info("🔄 Refresh the page to start a new session.")
            
            cleanup_js = """
            <script>
            // Clear any browser storage
            localStorage.clear();
            sessionStorage.clear();
            
            // Clear speech recognition if active
            if (window.recognition) {
                try {
                    window.recognition.stop();
                } catch (e) {
                    // Recognition wasn't active
                }
            }
            
            // Show confirmation
            document.body.innerHTML = '<div style="background: #d4edda; color: #155724; padding: 20px; border-radius: 10px; text-align: center; font-size: 18px; margin: 20px;"><h3>🔚 Session Ended</h3><p>All data has been cleared from this browser.</p><p><strong>Refresh the page to start a new session.</strong></p><button onclick="location.reload()" style="background: #800000; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">🔄 Refresh Page</button></div>';
            </script>
            """
            
            components.html(cleanup_js, height=200)
            
            st.stop()

    # UChicago branding
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background-color: #800000; color: white; text-align: center; border-radius: 10px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/uchicago_logo.png" 
                 alt="UChicago" style="height: 60px; background-color: white; padding: 8px; border-radius: 6px;">
            <span style="font-weight: bold;">University of Chicago</span>
        </div>
        <p style="margin: 5px 0;">Data Science Institute | MS in Applied Data Science</p>
        <p style="margin: 0; font-size: 0.9rem; color: #D6D6CE;">Official Program Information Assistant. </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
