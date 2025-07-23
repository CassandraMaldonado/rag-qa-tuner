# Q&A Bot

An intelligent chatbot for the University of Chicago Master's in Applied Data Science program, powered by RAG technology.

## Live Demo

**Streamlit Cloud URL:**

## Overview

This chatbot provides accurate information about the UChicago MSADS program by:

- **Web Scraping**: Automatically scraped the UChicago Data Science Institute website.
- **Smart Chunking**: Created optimized text chunks for important facts.
- **Semantic Search**: Used FAISS vector search with keyword boosting for information retrieval.
- **Enhanced Generation**: Used OpenAI GPT 3.5 for answers.

## Technical Architecture

### RAG Pipeline
1. **Data Collection**: Scraped over 15 pages from the official UChicago DSI website.
2. **Content Processing**: Advanced text extraction.
3. **Chunking Strategy**: 
   - Regular overlapping chunks (800 chars, 100 overlap).
   - Specialized micro-chunks for key facts.
4. **Embedding Generation**: Used `all-MiniLM-L6-v2` with batch processing.
5. **Vector Search**: FAISS IndexFlatIP with cosine similarity.
6. **Answer Generation**: OpenAI GPT 3.5 with specialized prompts.

### Performance Optimizations
- Fast regex processing.
- Batch embedding generation with progress tracking  
- Memory-efficient chunking for large documents
- Keyword boosting for enhanced relevance

## System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB for models and data
- **Internet**: Required for web scraping and OpenAI API

## Optimized Questions

The system includes pre-configured optimized questions for best results:

1. **Tuition**: What is the exact dollar amount tuition cost per course and total tuition for the MS in Applied Data Science program?
2. **Scholarships**: What are the specific names of scholarships including Data Science Institute Scholarship and Alumni Scholarship?
3. **Deadlines**: What are all the specific application deadline dates with month, day and year for the in-person MS Applied Data Science program?
4. **Appointments**: How do I schedule an advising appointment with Jose Alvarado or Patrick Vonesh and what is the portal URL link?
5. **Transcripts**: What is the complete mailing address including street address, suite number, and zip code for sending official transcripts?
