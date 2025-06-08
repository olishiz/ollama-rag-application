import os
import tempfile
import streamlit as st
import PyPDF2
from PyPDF2.errors import PdfReadError
import requests
from PIL import Image
import io
import base64
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
import google.generativeai as genai
from google import genai as genai_client
import os # Added for os.unlink
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(page_title="RAG App", layout="wide")

# App title and description
st.title("📚 Document & Image Query Engine")
st.markdown("""
Upload a PDF document or an image and ask questions about its content.
This app uses Retrieval-Augmented Generation (RAG) to provide accurate answers.
""")

# Initialize session state variables
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'input_type' not in st.session_state:
    st.session_state.input_type = None
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
# Ensure agent_mode is initialized here if not already, to avoid issues before sidebar rendering
if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = 'local' # Default to local
if 'online_ollama_url' not in st.session_state:
    st.session_state.online_ollama_url = "http://localhost:11434"
if 'uploaded_pdf_file' not in st.session_state:
    st.session_state.uploaded_pdf_file = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        # Check if the PDF is encrypted, and if so, try to decrypt with an empty password
        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt('')
            except Exception as e:
                st.error(f"Could not decrypt PDF: {e}. If it's password protected, this app cannot open it.")
                return None

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # This can happen for scanned PDFs or pages with no text
                    st.warning(f"Page {page_num + 1} contained no extractable text. It might be an image-based PDF.")
            except Exception as e:
                st.error(f"Error extracting text from page {page_num + 1}: {e}")
                # Optionally, continue to next page or return None
                # return None
        if not text.strip():
            st.warning("No text could be extracted from the PDF. The document might be empty or consist entirely of images.")
            return None
        return text
    except PdfReadError as e:
        st.error(f"Error reading PDF: {e}. The file might be corrupted, not a valid PDF, or the EOF marker is missing.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the PDF: {e}")
        return None

# Function to process the document
def process_document(text):
    # Check if text is None or empty
    if text is None or not text.strip():
        st.error("Cannot process document: No extractable text found in the PDF.")
        return None, None, None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vectorstore
    embeddings = None
    llm = None
    if st.session_state.agent_mode == 'local':
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = Ollama(model="qwen2.5vl:7b")
    elif st.session_state.agent_mode == 'online':
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=st.session_state.online_ollama_url)
        llm = Ollama(model="qwen2.5vl:7b", base_url=st.session_state.online_ollama_url)
    elif st.session_state.agent_mode == 'google':
        if not st.session_state.google_api_key:
            st.error("Google API Key is not configured. Please set it in the sidebar.")
            return None, None, None
        try:
            # genai.configure already called in sidebar if key is present
            # For RAG, we need embeddings. Google's Palm/Vertex AI embeddings can be used, or a generic one.
            # Using HuggingFaceEmbeddings as a fallback if no direct Google embedding is set up.
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # LLM for RetrievalQA is not directly Gemini. We'll call Gemini API in the Q&A step.
            llm = None # Placeholder, will use Gemini API directly.
        except Exception as e:
            st.error(f"Failed to configure for Google agent during document processing: {e}")
            return None, None, None
    else:
        st.error(f"Unknown agent mode: {st.session_state.agent_mode}")
        return None, None, None

    if not embeddings:
        st.error("Embeddings could not be initialized.")
        return None, None, None

    vectorstore = FAISS.from_texts(chunks, embeddings)


    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create QA chain or set up for custom handling
    if st.session_state.agent_mode == 'google':
        # No traditional qa_chain for Google; we'll use retriever and call Gemini API directly.
        qa_chain = None
    elif llm: # For local and online Ollama
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    else:
        st.error("LLM not initialized for QA chain.")
        qa_chain = None

    return vectorstore, retriever, qa_chain


# Function to process image with Ollama
def process_image(image):
    # Convert image to base64
    buffered = io.BytesIO()
    # Convert RGBA to RGB mode if needed, as JPEG doesn't support transparency
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Set up LLM/Model for image processing
    model_instance = None
    if st.session_state.agent_mode == 'local':
        model_instance = Ollama(model="qwen2.5vl:7b")
    elif st.session_state.agent_mode == 'online':
        model_instance = Ollama(model="qwen2.5vl:7b", base_url=st.session_state.online_ollama_url)
    elif st.session_state.agent_mode == 'google':
        if not st.session_state.google_api_key:
            st.error("Google API Key is not configured for image processing.")
            return None, img_str
        try:
            # genai.configure should have been called in the sidebar
            model_instance = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # Use appropriate vision model
        except Exception as e:
            st.error(f"Failed to initialize Google Gemini for image processing: {e}")
            return None, img_str

    return model_instance, img_str # img_str is base64 for Ollama, PIL image might be needed for Gemini directly


# Function to load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# SpongeBob and Patrick image modal popup
def show_spongebob_patrick():
    # Use the SpongeBob and Patrick images
    spongebob_url = "https://upload.wikimedia.org/wikipedia/commons/7/7a/SpongeBob_SquarePants_character.png"
    patrick_url = "https://ih1.redbubble.net/image.4032169436.4488/bg,f8f8f8-flat,750x,075,f-pad,750x1000,f8f8f8.jpg"

    try:
        # Create a placeholder for the modal
        modal_placeholder = st.empty()

        with modal_placeholder.container():
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            .modal-content {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 80%;
                max-height: 80%;
                overflow: auto;
                position: relative;
            }
            .close-button {
                position: absolute;
                top: 10px;
                right: 10px;
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("<div class='modal-overlay'><div class='modal-content'>", unsafe_allow_html=True)

            # Add a close button at the top right
            if st.button("✖️", key="close_modal", help="Close", type="secondary"):
                st.session_state.show_spongebob = False
                st.experimental_rerun()

            st.markdown("## SpongeBob & Patrick Modal")

            # Create two columns for the images
            col1, col2 = st.columns(2)

            # Display SpongeBob in the first column
            with col1:
                st.markdown("### SpongeBob SquarePants!")
                st.image(spongebob_url, width=250)

            # Display Patrick in the second column
            with col2:
                st.markdown("### Patrick Star!")
                st.image(patrick_url, width=250)

            st.markdown("</div></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying SpongeBob & Patrick modal: {e}")

# Sidebar for document/image upload
with st.sidebar:
    st.header("Upload Content")

    # Input type selection
    input_type = st.radio("Select input type:", ["PDF Document", "Image", "Chatbot"])
    st.session_state.input_type = input_type

    st.markdown("--- Agent Configuration ---")
    agent_options = ['local', 'google']
    # Get current index, default to 0 if agent_mode is somehow not in options (should not happen)
    current_agent_mode_index = agent_options.index(st.session_state.agent_mode) if st.session_state.agent_mode in agent_options else 0

    selected_agent_mode = st.selectbox(
        "Select Agent Mode:",
        agent_options,
        index=current_agent_mode_index,
        key='agent_mode_selector'
    )

    if selected_agent_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = selected_agent_mode
        st.experimental_rerun() # Rerun if the selection changes

    st.write(f"Current Agent Mode: **{st.session_state.agent_mode.capitalize()}**")

    if st.session_state.agent_mode == 'online':
        st.info("Please ensure your online Ollama instance is running and accessible.")
        st.session_state.online_ollama_url = st.text_input("Online Ollama URL", st.session_state.online_ollama_url)
    elif st.session_state.agent_mode == 'google':
        st.info("Using Google Gemini. Please enter your API key.")
        st.session_state.google_api_key = st.text_input("Google API Key", type="password", value="AIzaSyBuUbwo94ZkR5L0t2HnM_1lEhAYvhxRRxA")
        if not st.session_state.google_api_key:
            st.warning("Google API Key is required to use the Google agent.")
        else:
            try:
                genai.configure(api_key=st.session_state.google_api_key)
                # Test configuration by listing models, for example
                # genai.list_models()
                st.success("Google API Key configured.")
            except Exception as e:
                st.error(f"Invalid Google API Key or configuration error: {e}")


    if input_type == "PDF Document":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Handle differently based on agent mode
                if st.session_state.agent_mode == 'google':
                    try:
                        # Upload the PDF file to Google's API
                        client = genai_client.Client(api_key=st.session_state.google_api_key)
                        uploaded_pdf = client.files.upload(file=tmp_file_path)
                        st.session_state.uploaded_pdf_file = uploaded_pdf

                        # Still extract text for RAG fallback if needed
                        text = extract_text_from_pdf(tmp_file_path)
                        st.session_state.processed_text = text

                        # Process document for RAG fallback only if text was successfully extracted
                        if text is not None:
                            vectorstore, retriever, qa_chain = process_document(text)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.retriever = retriever
                            st.session_state.qa_chain = qa_chain
                        else:
                            # Even if text extraction fails, we can still use the uploaded PDF file directly with Google API
                            st.warning("No extractable text found in the PDF, but the file was uploaded to Google API successfully. Using image-based PDF processing.")
                            st.session_state.vectorstore = None
                            st.session_state.retriever = None
                            st.session_state.qa_chain = None

                        st.success("Document uploaded to Google API and processed successfully!")
                    except Exception as e:
                        st.error(f"Error uploading PDF to Google API: {e}")
                        # Fall back to text extraction
                        text = extract_text_from_pdf(tmp_file_path)
                        st.session_state.processed_text = text

                        # Process document only if text was successfully extracted
                        if text is not None:
                            vectorstore, retriever, qa_chain = process_document(text)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.retriever = retriever
                            st.session_state.qa_chain = qa_chain
                        else:
                            st.error("Cannot process document: No extractable text found in the PDF.")
                            st.session_state.vectorstore = None
                            st.session_state.retriever = None
                            st.session_state.qa_chain = None

                        st.warning("Falling back to text extraction due to upload error.")
                else:
                    # Extract text from PDF for local/online modes
                    text = extract_text_from_pdf(tmp_file_path)
                    st.session_state.processed_text = text

                    # Process document only if text was successfully extracted
                    if text is not None:
                        vectorstore, retriever, qa_chain = process_document(text)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = retriever
                        st.session_state.qa_chain = qa_chain
                        st.success("Document processed successfully!")
                    else:
                        st.error("Cannot process document: No extractable text found in the PDF.")
                        st.session_state.vectorstore = None
                        st.session_state.retriever = None
                        st.session_state.qa_chain = None

                # Clean up the temporary file
                os.unlink(tmp_file_path)

                # Reset image state
                st.session_state.image = None

    else:  # Image input
        image_source = st.radio("Select image source:", ["Upload", "URL"])

        if image_source == "Upload":
            uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                with st.spinner("Processing image..."):
                    # Load the image
                    image = Image.open(uploaded_image)
                    st.session_state.image = image

                    # Reset document state
                    st.session_state.processed_text = None
                    st.session_state.vectorstore = None
                    st.session_state.retriever = None
                    st.session_state.qa_chain = None

                    st.success("Image processed successfully!")

        else:  # URL input
            image_url = st.text_input("Enter image URL:")

            if image_url and st.button("Load Image"):
                with st.spinner("Loading and processing image..."):
                    image = load_image_from_url(image_url)

                    if image is not None:
                        st.session_state.image = image

                        # Reset document state
                        st.session_state.processed_text = None
                        st.session_state.vectorstore = None
                        st.session_state.retriever = None
                        st.session_state.qa_chain = None

                        st.success("Image processed successfully!")


    st.markdown("---")
    st.markdown("""
    ## About

    This is a RAG (Retrieval Augmented Generation) application that allows you to:

    1. Upload and process PDF documents for question answering
    2. Upload images (from file or URL) for visual analysis
    3. Chat directly with the AI using the Chatbot feature

    ### Technologies Used:
    - LangChain for document processing and RAG pipeline
    - nomic-embed-text for embeddings
    - Qwen2.5VL:7b for text and image processing
    """)

# Main content area


# Display and query content based on input type
if st.session_state.qa_chain is not None or (st.session_state.agent_mode == 'google' and st.session_state.uploaded_pdf_file is not None):  # PDF document processing
    st.header("Ask Questions About Your Document")
    with st.form(key='document_query_form'):
        query = st.text_input("Enter your question:")
        submit_button = st.form_submit_button(label='Get Answer')

        if submit_button and query:
            with st.spinner("Generating answer..."):
                answer = "Could not generate an answer."
                source_docs = []
                if st.session_state.agent_mode == 'google':
                    if not st.session_state.google_api_key:
                        st.error("Google API Key is not configured.")
                    elif st.session_state.uploaded_pdf_file is not None:
                        # Use the uploaded PDF file with the new Google API
                        try:
                            client = genai_client.Client(api_key=st.session_state.google_api_key)
                            prompt = f"Answer this question based on the PDF document: {query}"
                            response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=[prompt, st.session_state.uploaded_pdf_file]
                            )
                            answer = response.text
                            # No source docs when using direct file upload
                            source_docs = []
                        except Exception as e:
                            st.error(f"Error querying Google Gemini with PDF file: {e}")
                            answer = f"Error: {e}"
                            # Fall back to RAG if available
                            if st.session_state.retriever:
                                try:
                                    st.warning("Falling back to text-based RAG due to file upload query error.")
                                    docs = st.session_state.retriever.get_relevant_documents(query)
                                    context = "\n".join([doc.page_content for doc in docs])
                                    prompt = f"Based on the following context, answer the question.\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
                                    model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20")
                                    response = model.generate_content(prompt)
                                    answer = response.text
                                    source_docs = docs
                                except Exception as e2:
                                    st.error(f"Error with fallback RAG: {e2}")
                    elif not st.session_state.retriever:
                        st.error("Document not processed yet or retriever not available.")
                    else:
                        # Fall back to RAG if file upload wasn't successful
                        try:
                            docs = st.session_state.retriever.get_relevant_documents(query)
                            context = "\n".join([doc.page_content for doc in docs])
                            prompt = f"Based on the following context, answer the question.\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
                            model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20") # Or gemini-2.0-flash as per user's latest info
                            response = model.generate_content(prompt)
                            answer = response.text
                            source_docs = docs
                        except Exception as e:
                            st.error(f"Error querying Google Gemini: {e}")
                            answer = f"Error: {e}"
                elif st.session_state.qa_chain: # For local/online Ollama
                    try:
                        result = st.session_state.qa_chain({"query": query})
                        answer = result["result"]
                        source_docs = result["source_documents"]
                    except Exception as e:
                        st.error(f"Error with QA chain: {e}")
                        answer = f"Error: {e}"

                # Display answer
                st.markdown("### Answer")
                st.text_area("Answer", answer, height=200, disabled=True)

                # Display source documents
                with st.expander("Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(f"```\n{doc.page_content}\n```")
                        st.markdown("---")

elif st.session_state.image is not None:  # Image processing
    st.header("Ask Questions About Your Image")

    # Display the image
    st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)

    # Get the image query
    with st.form(key='image_query_form'):
        image_query = st.text_input("Enter your question about the image:")
        submit_button = st.form_submit_button(label='Get Answer')

        if submit_button and image_query:
            with st.spinner("Analyzing image..."):
                model_or_llm, img_data_for_llm = process_image(st.session_state.image) # img_data_for_llm is base64
                response = "Could not analyze image."

                if st.session_state.agent_mode == 'google':
                    if model_or_llm: # This is the Gemini Model (e.g., gemini-pro-vision)
                        try:
                            # Gemini Vision typically takes the image (PIL) and text prompt separately or as parts of a list.
                            # The user provided: client.models.generate_content(model="gemini-2.0-flash", contents=["Explain...", image_object])
                            # Adapting to genai.GenerativeModel('gemini-pro-vision').generate_content([prompt, image])
                            pil_image = st.session_state.image # Original PIL image
                            json_instruction = "Please format your response as a valid JSON object with the structure: { \"description\": \"...\", \"answer\": \"...\", \"elements\": [...], \"confidence\": \"...\" }. Return ONLY the JSON object."
                            prompt_parts = [f"Analyze this image and answer the following question: {image_query}. {json_instruction}", pil_image]

                            gemini_response = model_or_llm.generate_content(prompt_parts)
                            response = gemini_response.text
                        except Exception as e:
                            st.error(f"Error analyzing image with Google Gemini: {e}")
                            response = f"Error: {e}"
                    else:
                        response = "Google Gemini model not available/configured for image analysis."
                elif model_or_llm: # For local/online Ollama (model_or_llm is an Ollama instance)
                    prompt = f"""Analyze this image and answer the following question: {image_query}

                    Please format your response as a valid JSON object with the following structure:
                    {{
                        "description": "A brief description of what you see in the image",
                        "answer": "Your detailed answer to the question",
                        "elements": ["List of key elements identified in the image"],
                        "confidence": "High/Medium/Low based on your certainty"
                    }}

                    Return ONLY the JSON object without any additional text."""
                    try:
                        response = model_or_llm.invoke(prompt, images=[img_data_for_llm]) # img_data_for_llm is base64
                    except Exception as e:
                        st.error(f"Error analyzing image with Ollama: {e}")
                        response = f"Error: {e}"
                else:
                    response = "Image analysis model not available for the current agent."


            # Display the response
            st.markdown("### Analysis")

            # Initialize session state for clipboard notification
            if 'json_copied' not in st.session_state:
                st.session_state.json_copied = False

            # Create tabs for different views
            json_tab, text_tab = st.tabs(["JSON View", "Text View"])

            # Try to parse the response as JSON
            try:
                # Clean the response to extract just the JSON part
                # This handles cases where the model might add text before or after the JSON
                json_str = response
                # Try to find JSON-like content between curly braces
                import re
                json_match = re.search(r'\{[\s\S]*\}', json_str)
                if json_match:
                    json_str = json_match.group(0)

                # Parse the JSON
                json_response = json.loads(json_str)

                # JSON View Tab
                with json_tab:
                    # Display the JSON in a formatted way
                    st.json(json_response)

                    # Add a copy button with notification
                    st.code(json_str, language="json")

                    # Function to handle copy button click
                    def copy_to_clipboard():
                        st.session_state.json_copied = True

                    # Copy button
                    st.button("Copy to Clipboard", key="copy_json",
                              on_click=copy_to_clipboard)

                    # Show success message if copied
                    if st.session_state.json_copied:
                        st.success("JSON copied to clipboard!")
                        # Add the actual copy script
                        # Prepare the JavaScript string with proper escaping
                        js_str = json_str.replace("'", "\\'")
                        st.markdown(f"<script>navigator.clipboard.writeText('{js_str}');</script>",
                                   unsafe_allow_html=True)
                        # Reset after 3 seconds
                        import time
                        time.sleep(0.1)  # Small delay to ensure the message is shown
                        st.session_state.json_copied = False

                # Text View Tab
                with text_tab:
                    # Display a more readable text version
                    st.subheader("Description")
                    st.write(json_response.get("description", "No description available"))

                    st.subheader("Answer")
                    st.write(json_response.get("answer", "No answer available"))

                    st.subheader("Elements Identified")
                    elements = json_response.get("elements", [])
                    if elements:
                        for i, element in enumerate(elements, 1):
                            st.write(f"{i}. {element}")
                    else:
                        st.write("No elements identified")

                    st.subheader("Confidence")
                    st.write(json_response.get("confidence", "Not specified"))

            except Exception as e:
                # If JSON parsing fails, show the raw response in both tabs
                with json_tab:
                    st.error("Could not parse response as JSON. Raw response:")
                    st.write(response)

                with text_tab:
                    st.write("Raw Text Response:")
                    st.write(response)

elif st.session_state.input_type == "Chatbot":
    st.header("Chat with AI")

    # Add a brief explanation
    st.markdown("""
    This is a simple chatbot powered by Qwen2.5VL:7b. Type your message below and click 'Send' to chat with the AI.
    """)

    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'role': 'assistant', 'content': 'Hello! I\'m your AI assistant. How can I help you today?'}
        ]

    # Add a clear button for chat history
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat"):
            # Reset chat history but keep the welcome message
            st.session_state.chat_history = [
                {'role': 'assistant', 'content': 'Hello! I\'m your AI assistant. How can I help you today?'}
            ]
            st.experimental_rerun()

    # Create a container for chat messages with scrolling
    chat_container = st.container()

    # Chat input form - placing it before displaying messages for better UX
    with st.form(key='chat_form'):
        chat_input = st.text_input("Type your message:", key="chat_input")
        chat_button = st.form_submit_button(label='Send')

        if chat_button and chat_input:
            # Add user message to chat history
            st.session_state.chat_history.append({'role': 'user', 'content': chat_input})

            with st.spinner("AI is thinking..."):
                response_content = "Could not get a response."
                if st.session_state.agent_mode == 'local':
                    llm = Ollama(model="qwen2.5vl:7b")
                    try:
                        response_content = llm.invoke(chat_input)
                    except Exception as e:
                        response_content = f"Error with local Ollama: {e}"
                elif st.session_state.agent_mode == 'online':
                    llm = Ollama(model="qwen2.5vl:7b", base_url=st.session_state.online_ollama_url)
                    try:
                        response_content = llm.invoke(chat_input)
                    except Exception as e:
                        response_content = f"Error with online Ollama: {e}"
                elif st.session_state.agent_mode == 'google':
                    if not st.session_state.google_api_key:
                        response_content = "Google API Key is not configured. Please set it in the sidebar."
                    else:
                        try:
                            # genai.configure should have been called
                            model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20") # Or gemini-2.0-flash
                            # For a chatbot, you might want to manage history. For simplicity, just sending the input.
                            gemini_response = model.generate_content(chat_input)
                            response_content = gemini_response.text
                        except Exception as e:
                            response_content = f"Error with Google Gemini: {e}"
                else:
                    response_content = f"Unknown agent mode: {st.session_state.agent_mode}"

                # Add AI response to chat history
                st.session_state.chat_history.append({'role': 'assistant', 'content': response_content})

            # Rerun to update the chat display
            st.experimental_rerun()

    # Display chat history in the container with custom styling
    with chat_container:
        # Add some CSS for better message styling
        st.markdown("""
        <style>
        .user-message {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .ai-message {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display messages with custom styling
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-message'><strong>AI:</strong> {message['content']}</div>", unsafe_allow_html=True)

else:
    st.info("Please upload a PDF document or an image to get started, or select Chatbot to chat with AI.")
