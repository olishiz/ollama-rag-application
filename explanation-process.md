
# Ollama RAG Application Explanation

## App.py Explanation

The `app.py` file implements a Retrieval-Augmented Generation (RAG) application using Streamlit as the frontend framework. This application allows users to:

1. Upload and process documents (PDF, PPT, PPTX)
2. Upload and analyze images
3. Chat directly with AI models

### Key Components and Architecture

#### 1. Framework and Libraries
- **Streamlit**: Provides the web interface
- **LangChain**: Handles the RAG pipeline, including document processing, text splitting, and retrieval
- **FAISS**: Vector database for storing and retrieving document embeddings
- **Ollama**: Local LLM integration for text and image processing
- **Google Generative AI**: Alternative cloud-based AI option

#### 2. Agent Modes
The application supports three different agent modes:
- **Local**: Uses Ollama running locally for embeddings and LLM capabilities
- **Online**: Connects to a remote Ollama instance via URL
- **Google**: Uses Google's Gemini models for processing

#### 3. Document Processing Pipeline
1. **Text Extraction**: 
   - PDF documents are processed using PyPDF2
   - PowerPoint files are processed using python-pptx
   
2. **Text Chunking**:
   - Documents are split into smaller chunks using RecursiveCharacterTextSplitter
   - Default chunk size is 1000 characters with 200 character overlap

3. **Embedding Generation**:
   - For local/online modes: Uses Ollama's "nomic-embed-text" model
   - For Google mode: Uses HuggingFace's "sentence-transformers/all-MiniLM-L6-v2"

4. **Vector Storage**:
   - Embeddings are stored in a FAISS vector database
   - Retriever is configured to fetch the top 4 most relevant chunks

5. **Question Answering**:
   - For local/online modes: Uses Ollama's "qwen2.5vl:7b" model with RetrievalQA chain
   - For Google mode: Uses Gemini models with retrieved context

#### 4. Image Processing Pipeline
1. **Image Loading**:
   - Images can be uploaded directly or loaded from a URL
   
2. **Image Analysis**:
   - For local/online modes: Images are converted to base64 and sent to Ollama's "qwen2.5vl:7b" model
   - For Google mode: Images are sent to Gemini's vision model
   
3. **Response Formatting**:
   - Responses are formatted as JSON with description, answer, elements, and confidence
   - Results are displayed in both JSON and text views

#### 5. Chatbot Functionality
- Simple chat interface for direct interaction with the AI
- Supports all three agent modes
- Maintains chat history in session state

#### 6. Session State Management
- Streamlit's session state is used to maintain application state
- Stores processed documents, vector stores, retrievers, QA chains, and chat history