
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

## Updated README.md

# Document & Image RAG Application

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload PDF/PPT documents or images and query their content using natural language questions. It supports both local AI processing via Ollama and cloud-based processing via Google's Gemini models.

## Features

- PDF and PowerPoint document upload and processing
- Image upload (from local files or URLs)
- Text extraction and chunking for documents
- Vector embeddings using either Ollama or Google models
- Question answering for documents using RAG techniques
- Image analysis with structured JSON responses
- Interactive chatbot for direct AI interaction
- Multiple agent modes (local Ollama, online Ollama, Google Gemini)
- Interactive UI with Streamlit
- JSON and text views for image analysis results
- Copy to clipboard functionality for JSON responses

## Prerequisites

- Python 3.8+
- For local mode: Ollama installed with "qwen2.5vl:7b" and "nomic-embed-text" models
- For Google mode: Google API key for Gemini models

## Installation

1. Clone this repository or download the files

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:

```bash
python -m streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. In the sidebar, select your preferred agent mode:
   - Local: Uses locally installed Ollama
   - Google: Uses Google's Gemini models (requires API key)

4. Choose your input type (Document, Image, or Chatbot)

5. For documents:
   - Upload a PDF or PowerPoint document
   - Wait for the document to be processed
   - Enter your questions and click "Get Answer"
   - View the answer and source documents

6. For images:
   - Upload an image or provide a URL
   - Enter your questions about the image
   - View the results in either JSON or Text format

7. For the chatbot:
   - Type your message and click "Send"
   - View the conversation history
   - Use "Clear Chat" to reset the conversation

## How It Works

### Document Processing
1. The application extracts text from the uploaded document
2. The text is split into smaller chunks (1000 chars with 200 char overlap)
3. Each chunk is embedded using either Ollama or HuggingFace embeddings
4. The embeddings are stored in a FAISS vector database
5. When a question is asked, the application:
   - Retrieves the most relevant document chunks
   - Sends the question and relevant chunks to the LLM
   - Returns the generated answer with source documents

### Image Processing
1. The application loads the image (from upload or URL)
2. When a question is asked about the image:
   - For Ollama: The image is converted to base64 format
   - For Google: The image is sent directly to the Gemini vision model
   - The model analyzes the image and answers the question
   - Results are displayed in both JSON and text formats

### Chatbot
1. User messages are sent directly to the selected AI model
2. Responses are displayed in a chat-like interface
3. Chat history is maintained during the session

## Architecture

The application follows a modular architecture:
- Frontend: Streamlit for UI components and user interaction
- Document Processing: PyPDF2 and python-pptx for text extraction
- Embedding: Ollama or HuggingFace for vector embeddings
- Vector Storage: FAISS for efficient similarity search
- LLM Integration: Ollama API or Google Generative AI API
- Session Management: Streamlit session state

## Troubleshooting

- For local mode, ensure Ollama is running and has the required models
- For Google mode, verify your API key is correct
- Check that all dependencies are installed correctly
- If you encounter errors with document processing, try a different document format

## License

MIT