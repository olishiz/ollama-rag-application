# Document & Image RAG Application

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload PDF documents or images and query their content using natural language questions. It uses Google's Gemini 2.5-Pro model for text and image processing.

## Features

- PDF document upload and processing
- Image upload (from local files or URLs)
- Text extraction and chunking for PDFs
- Vector embeddings using Google's embedding model
- Question answering for PDFs using Google's Gemini 2.5-Pro model
- Image analysis using Google's Gemini 2.5-Pro Vision model
- Interactive chatbot for direct AI interaction
- Interactive UI with Streamlit
- JSON and text views for image analysis results
- Copy to clipboard functionality for JSON responses
- Easy deployment with Makefile

## Prerequisites

- Python 3.8+
- Google API key for Gemini 2.5-Pro (already configured in the app)

## Installation

1. Clone this repository or download the files

2. Use the Makefile to set up and run the application:

```bash
# Set up the environment and install dependencies
make setup

# Run the application
make run
```

Alternatively, you can install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application using the Makefile:

```bash
make run
```

Or manually:

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

## Deployment

The included Makefile provides several targets for easy deployment:

- `make setup`: Creates a virtual environment and installs dependencies
- `make run`: Runs the application in production mode
- `make dev`: Runs the application in development mode
- `make clean`: Cleans up the environment
- `make deploy`: Deploys the application (customize this target for your specific deployment needs)
- `make help`: Shows all available targets

## How It Works

### PDF Processing
1. The application extracts text from the uploaded PDF
2. The text is split into smaller chunks for processing
3. Each chunk is embedded using Google's embedding model
4. The embeddings are stored in a FAISS vector database
5. When a question is asked, the application:
   - Embeds the question
   - Retrieves the most relevant document chunks
   - Sends the question and relevant chunks to Gemini 2.5-Pro
   - Returns the generated answer

### Image Processing
1. The application loads the image (from upload or URL)
2. When a question is asked about the image:
   - The image is converted to base64 format
   - The image and question are sent to Gemini 2.5-Pro Vision model
   - The model analyzes the image and answers the question
   - Results are displayed in both JSON and text formats

## Troubleshooting

- Verify that your internet connection is working (required for API calls to Google)
- Check that all dependencies are installed correctly
- If you encounter API rate limits, wait a few minutes and try again

## License

MIT
