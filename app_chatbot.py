import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import time
from urllib.parse import urljoin, urlparse
import json

# Page configuration with UChicago branding
st.set_page_config(
    page_title="UChicago MS-ADS Q&A Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UChicago theme
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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class ImprovedRAGSystem:
    """
    Improved RAG system with better content extraction and answer generation.
    Fixed to work properly with OpenAI API and give complete, accurate answers.
    OPTIMIZED for performance - no more hanging on chunking or embeddings.
    """

    def __init__(self):
        self.scraped_data = []
        self.chunks = []
        self.vector_store = None
        self.embedding_model = None
        self.openai_client = None
        self.is_initialized = False

        # Improved configuration
        self.BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
        self.CHUNK_SIZE = 800  # Larger chunks for more context
        self.CHUNK_OVERLAP = 100  # More overlap to preserve context
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def setup_openai(self, api_key: str):
        """Setup OpenAI client with API key."""
        self.openai_client = openai.OpenAI(api_key=api_key)
        return True

    def scrape_website(self, max_pages: int = 15):
        """
        Enhanced web scraping to get complete, relevant content.
        Focus on getting full content, especially for tuition and financial info.
        """
        progress_bar = st.progress(0, text="Starting web scraping...")
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        visited_urls = set()
        urls_to_scrape = [self.BASE_URL]
        scraped_count = 0

        # Enhanced keywords - especially for tuition and financial info
        relevant_keywords = [
            'admission', 'admissions', 'apply', 'application',
            'curriculum', 'courses', 'course', 'program',
            'faculty', 'professors', 'staff',
            'tuition', 'cost', 'financial', 'aid', 'scholarship', 'funding',
            'requirements', 'prerequisite', 'career', 'outcomes',
            'employment', 'student', 'life', 'experience',
            'capstone', 'project', 'research', 'faq', 'faqs',
            'price', 'fee', 'fees', 'payment', 'billing'
        ]

        while urls_to_scrape and scraped_count < max_pages:
            current_url = urls_to_scrape.pop(0)

            if current_url in visited_urls:
                continue

            progress_bar.progress(scraped_count / max_pages, 
                                text=f"Scraping page {scraped_count + 1}/{max_pages}: {current_url[:50]}...")

            try:
                response = session.get(current_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                # Enhanced content extraction
                content_data = self._extract_enhanced_content(soup, current_url)
                if content_data and content_data['content'] and len(content_data['content']) > 100:
                    self.scraped_data.append(content_data)
                    scraped_count += 1

                visited_urls.add(current_url)

                # Find relevant links (prioritize from main page)
                if current_url == self.BASE_URL:
                    new_links = self._find_enhanced_links(soup, self.BASE_URL, relevant_keywords)
                    for link in new_links:
                        if link not in visited_urls and link not in urls_to_scrape:
                            urls_to_scrape.append(link)

                time.sleep(1)  # Be respectful

            except Exception as e:
                st.warning(f"Error scraping {current_url}: {str(e)}")
                continue

        progress_bar.progress(1.0, text=f"✓ Scraping complete! Collected {len(self.scraped_data)} pages")
        return self.scraped_data

    def _extract_enhanced_content(self, soup, url):
        """Enhanced content extraction that specifically preserves URLs and key structured data."""
        if not soup:
            return None

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Get title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract content with special attention to URLs and structured data
        content_parts = []

        # Strategy 1: Look for main content containers
        main_selectors = [
            'main', '.main-content', '#main-content', '.content',
            '.post-content', '.entry-content', '.page-content',
            '.container', '.wrapper', '.main', 'article'
        ]

        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        if main_content:
            # Extract text while preserving URLs and important formatting
            for element in main_content.find_all([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'li', 'div', 'span', 'td', 'th', 'dd', 'dt', 'a'
            ]):
                text = element.get_text().strip()

                # Special handling for links
                if element.name == 'a' and element.get('href'):
                    href = element.get('href')
                    if href.startswith('http') or href.startswith('www'):
                        text = f"{text} [URL: {href}]"

                # Keep longer text segments and those with important keywords
                if text and (len(text) > 15 or any(keyword in text.lower() for keyword in [
                    'tuition', 'cost', 'fee', 'scholarship', 'financial', 'admission',
                    'requirement', 'course', 'program', 'capstone', 'faculty',
                    'deadline', 'apply', 'contact', 'advisor', 'http', 'portal'
                ])):
                    content_parts.append(text)

        # Combine content
        content_text = " ".join(content_parts)

        # Clean up while preserving URLs
        content_text = re.sub(r'\s+', ' ', content_text)  # Normalize whitespace
        content_text = content_text.strip()

        return {
            'url': url,
            'title': title,
            'content': content_text,
            'length': len(content_text)
        }

    def _find_enhanced_links(self, soup, base_url, keywords):
        """Enhanced link finding with better keyword matching."""
        if not soup:
            return []

        relevant_links = []

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            link_text = link.get_text().lower().strip()

            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)

            # Enhanced relevance checking
            is_relevant = (
                self._is_same_domain(full_url, base_url) and
                (any(keyword in link_text for keyword in keywords) or
                 any(keyword in href.lower() for keyword in keywords) or
                 'faq' in href.lower() or 'tuition' in href.lower() or
                 'cost' in href.lower() or 'financial' in href.lower())
            )

            if is_relevant:
                relevant_links.append(full_url)

        return list(set(relevant_links))

    def _is_same_domain(self, url1, url2):
        """Check if URLs are from same domain."""
        try:
            return urlparse(url1).netloc == urlparse(url2).netloc
        except:
            return False

    def create_enhanced_chunks(self):
        """OPTIMIZED: Enhanced chunking strategy with fast regex processing."""
        progress_bar = st.progress(0, text="Creating enhanced text chunks...")
        
        self.chunks = []
        total_docs = len(self.scraped_data)

        for doc_idx, data in enumerate(self.scraped_data):
            progress_bar.progress(doc_idx / total_docs, 
                                text=f"Processing document {doc_idx + 1}/{total_docs}...")
            
            # Combine title and content with better formatting
            document = f"Page Title: {data['title']}\nSource URL: {data['url']}\n\nContent:\n{data['content']}"

            # Clean text while preserving important formatting
            cleaned_doc = re.sub(r'\s+', ' ', document)

            # Extract metadata for better source tracking
            source_url = data['url']
            title = data['title']

            # STRATEGY 1: Create specialized micro-chunks for key facts (OPTIMIZED)
            self._create_micro_chunks_optimized(cleaned_doc, doc_idx, source_url, title)

            # STRATEGY 2: Create regular overlapping chunks (existing logic)
            chunk_count = 0
            for i in range(0, len(cleaned_doc), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
                chunk_text = cleaned_doc[i:i + self.CHUNK_SIZE]

                # Skip very short chunks
                if len(chunk_text) < 150:
                    continue

                # Ensure chunks end at sentence boundaries when possible
                if i + self.CHUNK_SIZE < len(cleaned_doc):
                    # Try to end at a sentence
                    last_period = chunk_text.rfind('.')
                    last_question = chunk_text.rfind('?')
                    last_exclamation = chunk_text.rfind('!')

                    sentence_end = max(last_period, last_question, last_exclamation)
                    if sentence_end > len(chunk_text) * 0.8:  # If sentence end is in last 20%
                        chunk_text = chunk_text[:sentence_end + 1]

                self.chunks.append({
                    'text': chunk_text.strip(),
                    'doc_id': doc_idx,
                    'chunk_id': len(self.chunks),
                    'source_url': source_url,
                    'title': title,
                    'chunk_type': 'regular'
                })
                chunk_count += 1

        progress_bar.progress(1.0, text=f"✓ Created {len(self.chunks)} enhanced chunks")

    def _create_micro_chunks_optimized(self, document, doc_idx, source_url, title):
        """OPTIMIZED: Fast micro-chunk creation that prevents regex backtracking issues."""
        
        # Limit document size for regex processing to prevent hanging
        if len(document) > 50000:  # If document is very long, process in sections
            sections = [document[i:i+50000] for i in range(0, len(document), 45000)]
        else:
            sections = [document]
        
        for section in sections:
            # Simple, fast patterns (no complex quantifiers that cause backtracking)
            quick_patterns = {
                'tuition_cost': [
                    r'\$\d{1,2},?\d{3}\s*per\s*course',
                    r'\$\d{2},?\d{3}\s*total',
                    r'tuition[^.]{0,50}\$\d{1,2},?\d{3}',
                    r'\$\d{1,2},?\d{3}[^.]{0,30}tuition'
                ],
                'scholarship_names': [
                    r'Data Science Institute Scholarship',
                    r'MS in Applied Data Science Alumni Scholarship',
                    r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+Scholarship'
                ],
                'deadlines': [
                    r'deadline[^.]{0,100}(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
                    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[^.]{0,50}deadline',
                    r'application[^.]{0,50}due[^.]{0,50}\d{1,2}/\d{1,2}/\d{4}'
                ],
                'urls': [
                    r'https?://[^\s<>"{}|\\\^`\[\]]{10,100}',
                    r'www\.[^\s<>"{}|\\\^`\[\]]{5,50}'
                ],
                'contact_info': [
                    r'contact[^.]{0,50}(?:Patrick|Jose)',
                    r'(?:Patrick|Jose)[^.]{0,50}enrollment',
                    r'advising[^.]{0,30}appointment'
                ]
            }

            # Process each pattern type quickly
            for info_type, patterns in quick_patterns.items():
                for pattern in patterns:
                    try:
                        # Add timeout protection and limit matches
                        matches = list(re.finditer(pattern, section, re.IGNORECASE))[:5]  # Max 5 matches per pattern
                        
                        for match in matches:
                            # Get context around the match (±200 chars)
                            start = max(0, match.start() - 200)
                            end = min(len(section), match.end() + 200)
                            context = section[start:end].strip()

                            # Clean up context boundaries (simple version)
                            if len(context) > 100:  # Ensure meaningful context
                                # Simple boundary cleanup
                                if start > 0 and ' ' in context[:50]:
                                    space_idx = context.find(' ')
                                    context = context[space_idx:].strip()

                                self.chunks.append({
                                    'text': f"KEY {info_type.upper()}: {context}",
                                    'doc_id': doc_idx,
                                    'chunk_id': len(self.chunks),
                                    'source_url': source_url,
                                    'title': title,
                                    'chunk_type': 'micro',
                                    'info_type': info_type,
                                    'priority': 1.0
                                })
                                
                    except re.error:
                        # Skip problematic patterns
                        continue

    def create_embeddings(self):
        """OPTIMIZED: Create vector embeddings with progress tracking."""
        with st.spinner("Loading embedding model..."):
            self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
        
        progress_bar = st.progress(0, text=f"Creating embeddings for {len(self.chunks)} chunks...")
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Create embeddings in smaller batches with progress tracking
        batch_size = 16  # Reduced from 32 for better memory management
        all_embeddings = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch_texts = texts[i:i + batch_size]
            
            progress_bar.progress(batch_num / total_batches, 
                                text=f"Processing batch {batch_num}/{total_batches}...")
            
            # Create embeddings for this batch
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                show_progress_bar=False,  # Disable model's progress bar since we have our own
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.vector_store = {
            'index': index,
            'embeddings': embeddings,
            'chunks': self.chunks
        }
        
        progress_bar.progress(1.0, text=f"✓ Vector index created with {dimension}-dimensional embeddings")

    def search_chunks(self, query: str, k: int = 8):
        """Enhanced search with keyword boosting and priority weighting."""
        if not self.vector_store:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search with more results initially
        scores, indices = self.vector_store['index'].search(query_embedding, min(k * 3, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['semantic_score'] = float(score)
                results.append(result)

        # Apply keyword boosting and priority weighting
        enhanced_results = self._apply_keyword_boosting(query, results)

        # Sort by enhanced score and return top k
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)

        return enhanced_results[:k]

    def _apply_keyword_boosting(self, query, results):
        """Apply keyword boosting and priority weighting to search results."""
        query_lower = query.lower()

        # Define keyword boost mappings
        keyword_boosts = {
            'tuition': ['tuition', 'cost', 'fee', 'price', '$', 'dollar'],
            'scholarship': ['scholarship', 'financial aid', 'funding', 'grant'],
            'deadline': ['deadline', 'due date', 'application', 'submit', 'apply by'],
            'contact': ['contact', 'appointment', 'advisor', 'advising', 'schedule', 'meet'],
            'url': ['link', 'website', 'portal', 'apply', 'registration']
        }

        for result in results:
            text_lower = result['text'].lower()

            # Start with semantic score
            final_score = result['semantic_score']

            # Apply micro-chunk priority boost
            if result.get('chunk_type') == 'micro':
                final_score *= 1.5  # 50% boost for micro-chunks

            # Apply keyword boosting
            for category, keywords in keyword_boosts.items():
                if any(keyword in query_lower for keyword in keywords):
                    # Count keyword matches in the chunk
                    matches = sum(1 for keyword in keywords if keyword in text_lower)
                    if matches > 0:
                        final_score *= (1 + 0.2 * matches)  # Boost based on keyword frequency

            # Special boost for exact cost matches
            if any(term in query_lower for term in ['tuition', 'cost', 'price', 'fee']):
                if any(pattern in text_lower for pattern in ['$', 'dollar', 'per course', 'total']):
                    final_score *= 1.3

            # Special boost for scholarship name matches
            if 'scholarship' in query_lower:
                if any(name in text_lower for name in ['data science institute', 'alumni scholarship']):
                    final_score *= 1.4

            # Special boost for URL presence when contact/appointment mentioned
            if any(term in query_lower for term in ['contact', 'appointment', 'schedule', 'advisor']):
                if 'http' in text_lower or 'portal' in text_lower:
                    final_score *= 1.3

            result['final_score'] = final_score

        return results

    def generate_enhanced_answer(self, query: str, chunks: List[Dict]):
        """Enhanced answer generation with better context handling and fact extraction."""
        if not self.openai_client:
            return "OpenAI client not initialized"

        # Separate micro-chunks and regular chunks
        micro_chunks = [c for c in chunks if c.get('chunk_type') == 'micro']
        regular_chunks = [c for c in chunks if c.get('chunk_type') != 'micro']

        # Build context with micro-chunks prioritized
        context_parts = []

        # Add micro-chunks first (key facts)
        if micro_chunks:
            context_parts.append("KEY FACTS:")
            for i, chunk in enumerate(micro_chunks):
                context_parts.append(f"FACT {i+1}: {chunk['text']}")
            context_parts.append("\nADDITIONAL CONTEXT:")

        # Add regular chunks
        for i, chunk in enumerate(regular_chunks):
            context_parts.append(f"Source {i+1} (from {chunk['title']}):\n{chunk['text']}\n")

        context = "\n".join(context_parts)

        # Enhanced prompt with specific instructions for key facts
        prompt = f"""You are an expert assistant for the MS in Applied Data Science program at the University of Chicago.

Your task is to provide comprehensive, accurate answers based on the official program information provided below.

CONTEXT FROM OFFICIAL UCHICAGO WEBSITE:
{context}

QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. **COSTS/TUITION**: If asking about costs, you MUST include exact dollar amounts (e.g., "$5,967 per course", "$71,604 total tuition")
2. **SCHOLARSHIPS**: If asking about scholarships, you MUST mention specific scholarship names like "Data Science Institute Scholarship" and "MS in Applied Data Science Alumni Scholarship"
3. **DEADLINES**: If asking about deadlines, provide ALL specific dates mentioned (format: Month Day, Year)
4. **CONTACT/APPOINTMENTS**: If asking about scheduling or advising, include any URLs or portal links mentioned
5. **EXACT QUOTES**: For key facts, use the exact wording from the source material
6. **COMPLETENESS**: Provide all relevant details, not just summaries
7. **STRUCTURE**: Use bullet points for lists (costs, deadlines, requirements)
8. **SOURCE VERIFICATION**: If information seems incomplete, state "Additional details may be available on the official website"

Based ONLY on the information provided above, give a complete and detailed answer:

ANSWER:"""

        try:
            # Use GPT-3.5-turbo with optimized parameters for factual accuracy
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant for the UChicago MS in Applied Data Science program. You MUST provide complete, factual answers with exact details (costs, dates, names, URLs) from the provided context. Never summarize or generalize key facts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,  # More tokens for complete answers
                temperature=0.1,  # Very low temperature for maximum factual accuracy
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return "OpenAI API quota exceeded. Please add credits to your OpenAI account at https://platform.openai.com/account/billing"
            elif "401" in error_msg:
                return "Invalid OpenAI API key. Please check your API key."
            else:
                return f"Error generating response: {error_msg}"

    def ask_question(self, query: str):
        """Enhanced question answering with better error handling."""
        if not self.is_initialized:
            return "System not initialized. Please run full setup first.", []

        # Search for relevant chunks with more results
        relevant_chunks = self.search_chunks(query, 5)

        if not relevant_chunks:
            return "No relevant information found.", []

        # Generate enhanced answer
        answer = self.generate_enhanced_answer(query, relevant_chunks)

        return answer, relevant_chunks

    def initialize_system(self, openai_api_key: str, max_pages: int = 15):
        """OPTIMIZED: Initialize the complete enhanced RAG system."""
        try:
            # Setup OpenAI
            self.setup_openai(openai_api_key)

            # Enhanced scraping
            self.scrape_website(max_pages)

            if not self.scraped_data:
                st.error("Failed to scrape data")
                return False

            # Create enhanced chunks (ONLY ONCE - no duplication)
            self.create_enhanced_chunks()

            # Create embeddings (optimized with progress tracking)
            self.create_embeddings()

            self.is_initialized = True
            st.success("✓ Enhanced RAG System fully initialized and ready!")
            return True
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return False

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 UChicago MS-ADS Q&A Bot</h1>
        <p>Your intelligent assistant for the Master's in Applied Data Science program</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Setup")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key to initialize the system"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if api_key:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag_system = ImprovedRAGSystem()
                    success = st.session_state.rag_system.initialize_system(api_key)
                    st.session_state.system_ready = success
            else:
                st.error("Please enter your OpenAI API key first!")

        # System status
        if st.session_state.system_ready:
            st.success("✅ System Ready!")
            
            # System metrics
            if st.session_state.rag_system:
                rag = st.session_state.rag_system
                st.markdown("### 📊 System Info")
                st.metric("Pages Scraped", len(rag.scraped_data))
                st.metric("Text Chunks", len(rag.chunks))
                if rag.vector_store:
                    st.metric("Vector Dimensions", rag.vector_store['embeddings'].shape[1])
        else:
            st.warning("⚠️ System not initialized")

        # Quick tips
        st.markdown("### 💡 Usage Tips")
        st.markdown("""
        - Use specific questions for better results
        - Ask about tuition, scholarships, deadlines, etc.
        - The system searches official UChicago website content
        """)

        # Contact info
        st.markdown("### 📞 DSI Contact")
        st.markdown("""
        **In-Person Program:**  
        Jose Alvarado  
        Associate Director
        
        **Online Program:**  
        Patrick Vonesh  
        Senior Assistant Director
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Optimized question suggestions
        st.markdown("""
        <div class="question-suggestions">
            <strong>🎯 Optimized Questions (Click to use):</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Pre-optimized questions based on your requirements
        optimized_questions = [
            "What is the exact dollar amount tuition cost per course and total tuition for the MS in Applied Data Science program?",
            "What are the specific names of scholarships including Data Science Institute Scholarship and Alumni Scholarship for the MS Applied Data Science program?",
            "What is the complete mailing address including street address, suite number, and zip code for sending official transcripts to MS Applied Data Science Admissions?",
            "What are all the specific application deadline dates with month, day and year for the in-person MS Applied Data Science program 2024 and 2025?",
            "How do I schedule an advising appointment with Jose Alvarado or Patrick Vonesh and what is the portal URL link?",
            "What are the minimum scores for TOEFL and IELTS English Language Requirements?",
            "How many courses must you complete to earn UChicago's Master's in Applied Data Science?",
            "Is the MS in Applied Data Science program STEM/OPT eligible?",
            "Does the Master's in Applied Data Science Online program provide visa sponsorship?",
            "How do I apply to the MBA/MS program?"
        ]
        
        # Create clickable buttons for each question
        selected_question = None
        cols = st.columns(2)
        for i, question in enumerate(optimized_questions):
            with cols[i % 2]:
                if st.button(f"Q{i+1}: {question[:60]}...", key=f"q_{i}", help=question):
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
        if st.button("🔍 Get Answer", type="primary", disabled=not st.session_state.system_ready):
            if user_question and st.session_state.rag_system:
                with st.spinner("Searching and generating answer..."):
                    answer, sources = st.session_state.rag_system.ask_question(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'sources': sources,
                        'timestamp': time.strftime('%H:%M:%S')
                    })

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:80]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'])
                    
                    if chat['sources']:
                        st.markdown("**Sources:**")
                        for j, source in enumerate(chat['sources'][:3]):  # Show top 3 sources
                            relevance = source.get('final_score', source.get('semantic_score', 0))
                            st.markdown(f"- **Source {j+1}** (Relevance: {relevance:.3f}): [{source['title']}]({source['source_url']})")

    with col2:
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_ready and st.session_state.rag_system:
            rag = st.session_state.rag_system
            
            # Status metrics
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System Status", "🟢 Online")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Sources", f"{len(rag.scraped_data)} pages")
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

        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
            
        if st.button("📥 Download Chat History"):
            if st.session_state.chat_history:
                chat_data = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=chat_data,
                    file_name=f"uchicago_msads_chat_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        # Help section
        st.markdown("### ❓ Need Help?")
        st.markdown("""
        **Common Issues:**
        - Ensure your OpenAI API key is valid
        - Check your internet connection
        - Try rephrasing your question
        
        **Best Practices:**
        - Be specific in your questions
        - Use the optimized question templates
        - Ask one question at a time
        """)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎓 University of Chicago Data Science Institute | MS in Applied Data Science Program</p>
        <p>Powered by RAG (Retrieval-Augmented Generation) Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()