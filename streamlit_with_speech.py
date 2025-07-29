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
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import speech_recognition as sr
import queue
import threading
import io
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Config
st.set_page_config(
    page_title="UChicago MS-ADS Q&A Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with WebRTC styling
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
    
    .webrtc-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border: 2px solid #800000;
        margin: 10px 0;
    }
    
    .audio-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
    }
    
    .recording-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #dc3545;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    
    .transcript-box {
        background-color: white;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        min-height: 100px;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    .audio-controls {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)

# Global variables for audio processing
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()
    st.session_state.recognizer.energy_threshold = 4000
    st.session_state.recognizer.dynamic_energy_threshold = True
    st.session_state.recognizer.pause_threshold = 0.8
    st.session_state.recognizer.operation_timeout = None
    st.session_state.recognizer.phrase_threshold = 0.3

# Audio processing callback
def audio_frame_callback(frame):
    """Process audio frames from WebRTC"""
    audio = frame.to_ndarray()
    
    # Convert to int16 format for speech recognition
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Add to queue for processing
    if st.session_state.is_recording:
        st.session_state.audio_queue.put(audio_int16.tobytes())
    
    return frame

# Speech recognition worker
def speech_recognition_worker():
    """Background worker for speech recognition"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    while st.session_state.is_recording:
        try:
            # Collect audio data
            audio_data = b""
            timeout = time.time() + 3  # 3 second timeout
            
            while time.time() < timeout and st.session_state.is_recording:
                try:
                    chunk = st.session_state.audio_queue.get(timeout=0.1)
                    audio_data += chunk
                except queue.Empty:
                    continue
            
            if audio_data and len(audio_data) > 1000:  # Minimum audio length
                try:
                    # Convert bytes to AudioData
                    audio_segment = AudioSegment(
                        data=audio_data,
                        sample_width=2,  # 16-bit
                        frame_rate=48000,  # WebRTC default
                        channels=1
                    )
                    
                    # Export to wav format in memory
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    
                    # Create AudioData object
                    with sr.AudioFile(wav_io) as source:
                        audio = recognizer.record(source)
                    
                    # Recognize speech
                    try:
                        text = recognizer.recognize_google(audio, language='en-US')
                        if text:
                            st.session_state.transcript_text = text
                            st.rerun()
                    except sr.UnknownValueError:
                        pass  # No speech detected
                    except sr.RequestError as e:
                        st.error(f"Speech recognition service error: {e}")
                        
                except Exception as e:
                    st.error(f"Audio processing error: {e}")
                    
        except Exception as e:
            st.error(f"Recognition worker error: {e}")
            break

# Enhanced WebRTC Speech Component
def webrtc_speech_component():
    """Enhanced speech-to-text component using WebRTC"""
    
    st.markdown("### 🎤 Professional Voice Input (WebRTC)")
    
    with st.container():
        st.markdown('<div class="webrtc-container">', unsafe_allow_html=True)
        
        # WebRTC Configuration
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Audio settings
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # WebRTC Audio Streamer
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                audio_frame_callback=audio_frame_callback,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": False,
                    "audio": {
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": True,
                        "sampleRate": 48000,
                        "channelCount": 1
                    }
                }
            )
        
        with col2:
            # Recording controls
            if webrtc_ctx.state.playing:
                if not st.session_state.is_recording:
                    if st.button("🎙️ Start Recording", type="primary"):
                        st.session_state.is_recording = True
                        st.session_state.transcript_text = ""
                        # Start recognition worker in background
                        threading.Thread(target=speech_recognition_worker, daemon=True).start()
                        st.rerun()
                else:
                    if st.button("⏹️ Stop Recording", type="secondary"):
                        st.session_state.is_recording = False
                        st.rerun()
            
            # Clear transcript
            if st.button("🗑️ Clear", help="Clear transcript"):
                st.session_state.transcript_text = ""
                st.rerun()
        
        # Status indicator
        if webrtc_ctx.state.playing:
            if st.session_state.is_recording:
                st.markdown("""
                <div class="audio-status">
                    <div class="recording-indicator"></div>
                    <span style="color: #dc3545; font-weight: bold;">🔴 Recording... Speak now!</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="audio-status">
                    <span style="color: #28a745;">🟢 Ready to record</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("📢 Click 'START' to enable microphone access")
        
        # Transcript display
        st.markdown("**Live Transcript:**")
        transcript_display = st.session_state.transcript_text if st.session_state.transcript_text else "Your speech will appear here..."
        
        st.markdown(f'<div class="transcript-box">{transcript_display}</div>', unsafe_allow_html=True)
        
        # Action buttons
        if st.session_state.transcript_text:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    # JavaScript to copy to clipboard
                    copy_js = f"""
                    <script>
                    navigator.clipboard.writeText(`{st.session_state.transcript_text}`).then(function() {{
                        alert('✓ Copied to clipboard!');
                    }}).catch(function(err) {{
                        console.error('Could not copy text: ', err);
                    }});
                    </script>
                    """
                    components.html(copy_js, height=0)
            
            with col2:
                if st.button("📤 Use as Question"):
                    return st.session_state.transcript_text
            
            with col3:
                if st.button("🔄 Try Again"):
                    st.session_state.transcript_text = ""
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Audio quality settings
        with st.expander("🔧 Audio Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                energy_threshold = st.slider(
                    "Sensitivity", 
                    min_value=1000, 
                    max_value=8000, 
                    value=4000,
                    help="Lower = more sensitive to quiet speech"
                )
                st.session_state.recognizer.energy_threshold = energy_threshold
            
            with col2:
                pause_threshold = st.slider(
                    "Pause Detection (seconds)", 
                    min_value=0.3, 
                    max_value=2.0, 
                    value=0.8, 
                    step=0.1,
                    help="How long to wait before processing speech"
                )
                st.session_state.recognizer.pause_threshold = pause_threshold
        
        # Instructions
        with st.expander("📖 How to Use WebRTC Voice Input", expanded=False):
            st.markdown("""
            **Enhanced WebRTC Method:**
            1. **Click 'START'** to enable microphone access (browser will ask for permission)
            2. **Click '🎙️ Start Recording'** to begin voice capture
            3. **Speak clearly** - your speech will be transcribed in real-time
            4. **Click '⏹️ Stop Recording'** when finished
            5. **Use '📤 Use as Question'** to automatically populate the question box
            
            **Features:**
            - ✅ **Real-time transcription** with Google Speech Recognition
            - ✅ **Noise cancellation** and echo suppression
            - ✅ **Adjustable sensitivity** for different environments
            - ✅ **Cross-browser compatibility** (Chrome, Firefox, Safari, Edge)
            - ✅ **Professional audio processing** with WebRTC
            
            **Tips:**
            - Speak at normal pace and volume
            - Ensure good internet connection for speech recognition
            - Use headphones to reduce feedback
            - Adjust sensitivity if having issues with quiet/loud environments
            """)
    
    return None

# RAG System class (unchanged)
class Ragsystem:
    def __init__(self):
        self.chunks = []
        self.openai_client = None
        self.is_loaded = False
        self.metadata = {}
        self.vectorizer = None
        self.chunk_vectors = None
    
    def load_system(self, save_dir="rag_system_export"):
        try:
            if not os.path.exists(save_dir):
                st.error(f"Directory '{save_dir}' not found.")
                return False
            
            with st.spinner("Loading the RAG system..."):
                # Load metadata
                metadata_path = os.path.join(save_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    st.success(f"✓ Loaded metadata: {self.metadata.get('total_chunks', 0)} chunks expected.")
                
                # Load chunks
                chunks_path = os.path.join(save_dir, "chunks.pkl")
                if os.path.exists(chunks_path):
                    with open(chunks_path, 'rb') as f:
                        self.chunks = pickle.load(f)
                    st.success(f"✓ Loaded {len(self.chunks)} text chunks.")
                    
                    # Chunk type distribution
                    chunk_types = {}
                    for chunk in self.chunks:
                        chunk_type = chunk.get('chunk_type', 'unknown')
                        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    st.info(f"✓ Chunk types: {dict(chunk_types)}.")
                else:
                    st.error("chunks.pkl not found.")
                    return False
                
                # Create TF-IDF vectors
                with st.spinner("Creating search index..."):
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
                    st.success("✓ Created TF-IDF search index.")
                
                self.is_loaded = True
                st.success("✓ RAG system loaded successfully.")
                return True
                
        except Exception as e:
            st.error(f"Error loading system: {e}")
            return False

    def setup_openai(self, api_key: str):
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            # Test connection
            response = self.openai_client.models.list()
            st.success("✓ OpenAI API connected successfully.")
            return True
        except Exception as e:
            st.error(f"OpenAI setup failed: {e}")
            return False

    def search_chunks(self, query: str, k: int = 8):
        if not self.is_loaded or not self.vectorizer:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top candidates
            top_indices = similarities.argsort()[-k*3:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    result = self.chunks[idx].copy()
                    result['semantic_score'] = float(similarities[idx])
                    results.append(result)
            
            # Apply boosting
            enhanced_results = self.apply_boosting(query, results)
            enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return enhanced_results[:k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def apply_boosting(self, query, results):
        query_lower = query.lower()
        
        # Keyword categories with boosting
        boost_categories = {
            'deadline_application': {
                'keywords': ['deadline', 'date', 'apply', 'application', 'due', 'portal', 'september', 'cohort'],
                'boost': 2.5
            },
            'transcript_address': {
                'keywords': ['transcript', 'mail', 'address', 'send', 'official', 'university of chicago'],
                'boost': 2.5
            },
            'mba_joint': {
                'keywords': ['mba', 'booth', 'joint', 'dual', 'centralized', 'full-time mba'],
                'boost': 2.5
            },
            'visa_sponsorship': {
                'keywords': ['visa', 'sponsorship', 'f-1', 'international', 'in-person', 'full-time'],
                'boost': 2.5
            },
            'tuition_cost': {
                'keywords': ['tuition', 'cost', 'fee', 'price', 'dollar', 'per course', 'total'],
                'boost': 2.0
            }
        }
        
        for result in results:
            text_lower = result['text'].lower()
            final_score = result['semantic_score']
            
            # Boost for key facts
            chunk_type = result.get('chunk_type', 'regular')
            if chunk_type in ['key_fact', 'micro']:
                final_score *= 3.0
            elif chunk_type == 'important':
                final_score *= 2.0
            
            # Apply category boosting
            for category, config in boost_categories.items():
                query_matches = sum(1 for keyword in config['keywords'] if keyword in query_lower)
                if query_matches > 0:
                    text_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
                    if text_matches > 0:
                        boost_factor = config['boost'] * (1 + 0.2 * text_matches)
                        final_score *= boost_factor
            
            result['final_score'] = final_score
        
        return results

    def generate_answer(self, query: str, chunks: List[Dict]):
        if not self.openai_client:
            return "OpenAI client not configured."
        
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        # Build context
        context_parts = []
        key_facts = [c for c in chunks if c.get('chunk_type') in ['key_fact', 'micro']]
        important_chunks = [c for c in chunks if c.get('chunk_type') == 'important']
        regular_chunks = [c for c in chunks if c.get('chunk_type') not in ['key_fact', 'micro', 'important']]
        
        if key_facts:
            context_parts.append("CRITICAL INFORMATION:")
            for i, chunk in enumerate(key_facts[:6]):
                context_parts.append(f"{i+1}. {chunk['text']}")
            context_parts.append("")
        
        if important_chunks:
            context_parts.append("IMPORTANT DETAILS:")
            for chunk in important_chunks[:4]:
                context_parts.append(f"• {chunk['text']}")
            context_parts.append("")
        
        if regular_chunks:
            context_parts.append("ADDITIONAL CONTEXT:")
            for i, chunk in enumerate(regular_chunks[:4]):
                context_parts.append(f"Source {i+1}: {chunk['text']}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert assistant for the MS in Applied Data Science program at the University of Chicago.

OFFICIAL PROGRAM INFORMATION:
{context}

STUDENT QUESTION: {query}

Provide a complete, accurate, and helpful answer based on the official information provided. Include all specific details like addresses, dates, costs, and requirements when relevant."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert assistant for UChicago's MS in Applied Data Science program."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def ask_question(self, query: str):
        if not self.is_loaded:
            return "System not loaded", []
        
        relevant_chunks = self.search_chunks(query, 8)
        
        if relevant_chunks:
            answer = self.generate_answer(query, relevant_chunks)
        else:
            answer = f"I couldn't find specific information about '{query}' in the knowledge base. Please try rephrasing your question or contact the program directly."
            relevant_chunks = []
        
        return answer, relevant_chunks

# Session state initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Main function
def main():
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 30px;">
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/uchicago_logo.png" 
                 alt="University of Chicago" style="height: 120px; background-color: white; padding: 15px; border-radius: 10px;">
            <div style="text-align: center;">
                <h1 style="margin: 0; color: white;">MS in Applied Data Science</h1>
                <p style="margin: 5px 0; color: #D6D6CE;">Enhanced Voice Q&A Assistant</p>
                <p style="margin: 0; color: #D6D6CE; font-size: 0.9rem;">WebRTC-Powered Speech Recognition</p>
            </div>
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/dsi_logo.png" 
                 alt="Data Science Institute" style="height: 100px; background-color: white; padding: 15px; border-radius: 10px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for system setup
    with st.sidebar:
        st.markdown("### 🔧 System Setup")
        
        # Load RAG System
        st.markdown("#### 📂 Load RAG System")
        save_dir = st.text_input(
            "RAG System Directory", 
            value="rag_system_export",
            help="Path to your saved RAG system folder"
        )
        
        if st.button("📥 Load System", type="primary"):
            st.session_state.rag_system = Ragsystem()
            success = st.session_state.rag_system.load_system(save_dir)
            if success:
                st.session_state.system_ready = "loaded"
        
        # OpenAI API Key
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
            st.success("✅ System Ready")
            
            if st.session_state.rag_system:
                rag = st.session_state.rag_system
                st.metric("Text Chunks", len(rag.chunks))
                if rag.metadata:
                    st.metric("Pages Scraped", rag.metadata.get('total_pages', 'N/A'))
                st.info("🤖 RAG + LLM Mode Active")
                
        elif st.session_state.system_ready == "loaded":
            st.warning("⚠️ Add OpenAI API key to enable Q&A")
        else:
            st.warning("⚠️ Load RAG system first")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Enhanced WebRTC Speech Component
        voice_question = webrtc_speech_component()
        
        # Question input with voice integration
        if voice_question:
            user_question = st.text_area(
                "Your Question:", 
                value=voice_question,
                placeholder="Your voice input or type your question here...",
                height=100,
                key="question_input"
            )
        else:
            user_question = st.text_area(
                "Your Question:", 
                placeholder="Ask anything about the MS in Applied Data Science program or use voice input above...",
                height=100,
                key="question_input"
            )
        
        # Action buttons
        col_clear, col_ask = st.columns([1, 3])
        
        with col_clear:
            if st.button("🗑️ Clear All"):
                st.session_state.transcript_text = ""
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
                        
                        # Display sources
                        if sources:
                            st.info(f"📊 Found {len(sources)} relevant sources (response: {response_time:.1f}s)")
                            
                            with st.expander("📚 View Sources", expanded=False):
                                for i, source in enumerate(sources[:5]):
                                    st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                    st.markdown(f"**Source {i+1}** (Score: {source.get('final_score', 0):.4f})")
                                    st.markdown(f"**Type:** {source.get('chunk_type', 'regular')}")
                                    st.markdown(f"**Text:** {source['text'][:400]}...")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to chat history
                        input_method = 'voice' if voice_question else 'text'
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': answer,
                            'sources': len(sources) if sources else 0,
                            'timestamp': time.strftime('%H:%M:%S'),
                            'input_method': input_method
                        })

        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Recent Questions")
            for chat in reversed(st.session_state.chat_history[-3:]):
                input_icon = "🎤" if chat.get('input_method') == 'voice' else "⌨️"
                with st.expander(f"{input_icon} {chat['question'][:60]}... ({chat['timestamp']})"):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer'][:300]}...")

    with col2:
        st.markdown("### 🎛️ WebRTC Status")
        
        if st.session_state.system_ready == "ready":
            st.metric("System", "🟢 Online")
            st.metric("Questions", len(st.session_state.chat_history))
            
            if st.session_state.chat_history:
                voice_questions = sum(1 for chat in st.session_state.chat_history if chat.get('input_method') == 'voice')
                st.metric("Voice Questions", f"{voice_questions} 🎤")
        else:
            st.metric("System", "🔴 Setup Required")

        # WebRTC Audio Status
        st.markdown("### 🎙️ Audio Status")
        if st.session_state.is_recording:
            st.success("🔴 Recording Active")
        else:
            st.info("⚪ Ready to Record")
        
        if st.session_state.transcript_text:
            st.text_area(
                "Current Transcript", 
                value=st.session_state.transcript_text[:100] + ("..." if len(st.session_state.transcript_text) > 100 else ""),
                height=60, 
                disabled=True
            )

        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Clear History"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.button("🎤 Clear Voice"):
            st.session_state.transcript_text = ""
            st.rerun()
            
        if st.button("📊 Audio Stats"):
            st.info(f"""
            **Audio Queue:** {st.session_state.audio_queue.qsize()} items
            **Recording:** {'Yes' if st.session_state.is_recording else 'No'}
            **Transcript Length:** {len(st.session_state.transcript_text)} chars
            """)

        # System information
        st.markdown("### 🔧 WebRTC Features")
        st.info("""
        **Enhanced Audio:**
        - ✅ Real-time processing
        - ✅ Noise suppression
        - ✅ Echo cancellation
        - ✅ Auto gain control
        - ✅ Cross-browser support
        - ✅ Professional quality
        """)
            
        # Session management
        st.markdown("---")
        st.markdown("### 🗂️ Session Management")
        
        if st.button("🔚 End Session", type="secondary"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['audio_queue', 'recognizer']:  # Keep audio components
                    del st.session_state[key]
            
            # Reset audio state
            st.session_state.transcript_text = ""
            st.session_state.is_recording = False
            
            st.success("✓ Session ended. Refresh to start new session.")
            st.stop()

    # Requirements installation note
    st.markdown("---")
    st.info("""
    **📦 Required Dependencies for WebRTC Speech:**
    ```bash
    pip install streamlit-webrtc speechrecognition pydub pyaudio
    # For audio processing
    pip install av numpy
    ```
    """)

    # Footer
    st.markdown("""
    <div style="margin-top: 2rem; padding: 2rem; background-color: #800000; color: white; text-align: center; border-radius: 10px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;">
            <img src="https://raw.githubusercontent.com/CassandraMaldonado/rag-qa-tuner/main/assets/uchicago_logo.png" 
                 alt="UChicago" style="height: 60px; background-color: white; padding: 8px; border-radius: 6px;">
            <span style="font-weight: bold;">University of Chicago</span>
        </div>
        <p style="margin: 5px 0;">Data Science Institute | MS in Applied Data Science</p>
        <p style="margin: 0; font-size: 0.9rem; color: #D6D6CE;">WebRTC-Enhanced Voice Q&A Assistant 🎤🌐</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
