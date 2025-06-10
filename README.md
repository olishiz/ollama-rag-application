# Document & Image RAG Application

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload PDF/PPT
documents or images and query their content using natural language questions. It supports both local AI processing via
Ollama and cloud-based processing via Google's Gemini models.

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

You can install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application using:

```bash
python3 -m streamlit run app.py --server.port 8502 --server.headless true
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8502)

3. Choose your input type (PDF Document, Image, or Chatbot)

4. For PDF documents:
    - Upload a PDF document using the file uploader
    - Wait for the document to be processed
    - Enter your questions in the text input field and click "Get Answer"

5. For images:
    - Choose to upload from local files or provide a URL
    - Wait for the image to be processed
    - Enter your questions about the image and click "Get Answer"
    - View the results in either JSON or Text format using the tabs
    - Use the "Copy to Clipboard" button to copy the JSON response

6. For the chatbot:
    - Select the "Chatbot" option from the input type selection
    - Type your message in the text input field
    - Click "Send" to chat with the AI
    - View the conversation history in the chat window
    - Use the "Clear Chat" button to reset the conversation


## Troubleshooting

- Verify that your internet connection is working (required for API calls to Google)
- Check that all dependencies are installed correctly
- If you encounter API rate limits, wait a few minutes and try again

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

## Flow

User uploads document → Extract text → Split into chunks →
Create embeddings → Store in vector DB →
User asks question → Find relevant chunks →
Send to AI with context → Return answer

## License

MIT
