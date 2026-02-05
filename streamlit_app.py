"""
Streamlit UI for OmniRAG - Multimodal RAG System
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image as PILImage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.retrieval.retriever import HybridRetriever
from omnirag.models.llm_generator import LLMGenerator
from omnirag.models.image_processor import ImageProcessor
from omnirag.db.vector_store import VectorStore
from omnirag.utils.security import (
    validate_filename, sanitize_filename, validate_file_size, 
    sanitize_query, MAX_FILE_SIZE
)
from omnirag.utils.logger import logger


# Page configuration
st.set_page_config(
    page_title="OmniRAG - Multimodal RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = None
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None


def initialize_components():
    """Initialize vector store, retriever, and generator."""
    if st.session_state.image_processor is None:
        with st.spinner("Loading image processing models (this may take a minute)..."):
            st.session_state.image_processor = ImageProcessor(device="cpu")
    
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store..."):
            st.session_state.vector_store = VectorStore(
                persist_directory="./chroma_db",
                collection_name="omnirag",
                image_processor=st.session_state.image_processor
            )
    
    if st.session_state.retriever is None:
        st.session_state.retriever = HybridRetriever(st.session_state.vector_store)
    
    if st.session_state.generator is None:
        # Use OpenAI API (no token limits)
        # Load from .env file or environment variable
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini
        # Allow environment variables to set limits, but default to None (no limits)
        max_context = os.getenv("MAX_CONTEXT_TOKENS")
        max_response = os.getenv("MAX_RESPONSE_TOKENS")
        max_context = int(max_context) if max_context else None
        max_response = int(max_response) if max_response else None
        
        st.session_state.generator = LLMGenerator(
            model_type="openai",
            model_name=openai_model,
            openai_api_key=openai_key,
            max_context_tokens=max_context,  # None = no limit
            max_response_tokens=max_response  # None = model default
        )


def main():
    """Main Streamlit app."""
    st.title("📚 OmniRAG - Multimodal RAG System")
    st.markdown("Upload a PDF and ask questions. Get answers with images and citations!")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigation", ["Upload Document", "Chat"])
    
    if page == "Upload Document":
        upload_page()
    elif page == "Chat":
        chat_page()


def upload_page():
    """Upload and process PDF documents with security validation."""
    st.header("📤 Upload Document")
    
    initialize_components()
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help=f"Upload a PDF document (max {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"
    )
    
    if uploaded_file is not None:
        # Security validation
        # 1. Validate filename
        if not validate_filename(uploaded_file.name):
            st.error("❌ Invalid filename. Please use a safe filename with alphanumeric characters only.")
            logger.warning(f"Invalid filename attempted: {uploaded_file.name}")
            return
        
        # 2. Validate file size
        file_size = len(uploaded_file.getbuffer())
        is_valid, error_msg = validate_file_size(file_size)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            logger.warning(f"File size validation failed: {file_size} bytes")
            return
        
        # 3. Sanitize filename
        safe_filename = sanitize_filename(uploaded_file.name)
        
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / safe_filename
        
        if not file_path.exists() or st.button("Re-process Document"):
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                logger.info(f"File uploaded: {safe_filename} ({file_size / 1024 / 1024:.2f} MB)")
                
                st.session_state.uploaded_file = str(file_path)
                st.session_state.ingestion_complete = False
                
                # Process document
                with st.spinner("Processing document... This may take several minutes for large PDFs."):
                    try:
                        pipeline = IngestionPipeline(
                            st.session_state.vector_store,
                            st.session_state.image_processor
                        )
                        
                        summary = pipeline.ingest_pdf(str(file_path))
                        
                        st.session_state.ingestion_complete = True
                        
                        # Display summary
                        st.success("✅ Document processed successfully!")
                        st.json(summary)
                        logger.info(f"Document processed: {summary}")
                        
                    except Exception as e:
                        error_msg = f"Error processing document: {e}"
                        st.error(error_msg)
                        st.exception(e)
                        logger.error(error_msg, exc_info=True)
            except Exception as e:
                error_msg = f"Error saving file: {e}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
        else:
            st.info(f"Document '{safe_filename}' already uploaded. Click 'Re-process Document' to process again.")
            st.session_state.uploaded_file = str(file_path)
            st.session_state.ingestion_complete = True
    
    # Show status
    if st.session_state.ingestion_complete:
        st.success("✅ Ready to chat! Go to the Chat page.")
    else:
        st.info("👆 Upload a PDF document to get started.")


def chat_page():
    """Chat interface for asking questions."""
    st.header("💬 Chat with Document")
    
    if not st.session_state.ingestion_complete:
        st.warning("⚠️ Please upload and process a document first on the Upload page.")
        return
    
    initialize_components()
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display images if present
            if "images" in message and len(message["images"]) > 0:
                # Filter to only valid, existing images
                valid_images = []
                for idx, img_path in enumerate(message["images"]):
                    # Normalize path (handle both absolute and relative)
                    if img_path:
                        img_path_obj = Path(img_path)
                        if not img_path_obj.is_absolute():
                            img_path_obj = Path.cwd() / img_path_obj
                        if not img_path_obj.exists():
                            alt_path = Path("./chroma_db/metadata/images") / Path(img_path).name
                            if alt_path.exists():
                                img_path_obj = alt_path
                        img_path = str(img_path_obj)
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            # Verify image can be opened
                            test_img = PILImage.open(img_path)
                            test_img.verify()
                            valid_images.append({
                                'path': img_path,
                                'page': message.get('image_pages', [])[idx] if 'image_pages' in message and idx < len(message.get('image_pages', [])) else 0
                            })
                        except Exception as e:
                            logger.warning(f"Error loading image from history: {img_path}: {e}")
                            continue
                
                # Only display if we have valid images
                if valid_images:
                    num_cols = min(len(valid_images), 3)
                    cols = st.columns(num_cols)
                    for idx, img_data in enumerate(valid_images):
                        with cols[idx % num_cols]:
                            try:
                                img = PILImage.open(img_data['path'])
                                caption = f"Page {img_data['page']}" if img_data['page'] > 0 else ""
                                st.image(img, caption=caption, use_container_width=True)
                            except Exception as e:
                                logger.warning(f"Error displaying image in history: {e}")
            
            # Display citations
            if "citations" in message:
                with st.expander("📄 Citations"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f"**Citation {i}** (Page {citation['page']}):")
                        st.text(citation['text'][:300] + "...")
    
    # User input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Sanitize user query
        sanitized_prompt = sanitize_query(prompt)
        
        if not sanitized_prompt:
            st.warning("⚠️ Query was blocked for security reasons. Please rephrase your question.")
            logger.warning(f"Query blocked: {prompt[:50]}...")
            return
        
        # Add user message (use sanitized version)
        st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
        with st.chat_message("user"):
            st.markdown(sanitized_prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    # Retrieve (no token restrictions - get more context)
                    retrieval_results = st.session_state.retriever.retrieve(
                        sanitized_prompt,
                        text_top_k=10,  # Increased for better context
                        image_top_k=5  # Increased for better image coverage
                    )
                    
                    # Link vector store to generator for figure reference extraction
                    st.session_state.generator.vector_store = st.session_state.vector_store
                    
                    # Generate
                    response = st.session_state.generator.generate(
                        sanitized_prompt,
                        retrieval_results['text_results'],
                        retrieval_results['image_results']
                    )
                    
                    logger.info(f"Query processed: {sanitized_prompt[:50]}...")
                    
                    # Display answer
                    st.markdown(response['answer'])
                    
                    # Display images
                    image_paths = []
                    image_pages = []
                    if response['image_references']:
                        # First, collect all valid image paths
                        valid_images = []
                        for img_ref in response['image_references']:
                            img_path = img_ref.get('image_path', '')
                            
                            # Normalize path (handle both absolute and relative)
                            if img_path:
                                # Convert to Path object for better handling
                                img_path_obj = Path(img_path)
                                
                                # Try absolute path first
                                if not img_path_obj.is_absolute():
                                    # Try relative to current working directory
                                    img_path_obj = Path.cwd() / img_path_obj
                                
                                # Also try relative to chroma_db metadata directory
                                if not img_path_obj.exists():
                                    alt_path = Path("./chroma_db/metadata/images") / Path(img_path).name
                                    if alt_path.exists():
                                        img_path_obj = alt_path
                                
                                img_path = str(img_path_obj)
                            
                            if img_path and os.path.exists(img_path):
                                try:
                                    # Verify image can be opened
                                    test_img = PILImage.open(img_path)
                                    test_img.verify()  # Verify it's a valid image
                                    valid_images.append({
                                        'path': img_path,
                                        'page': img_ref.get('page', 0),
                                        'caption': img_ref.get('caption', '')
                                    })
                                except Exception as e:
                                    logger.warning(f"Invalid image file {img_path}: {e}")
                                    continue
                            else:
                                # Log missing images for debugging
                                logger.warning(f"Image path not found: {img_path} (from image_id: {img_ref.get('image_id', 'N/A')})")
                        
                        # Display images or message
                        if valid_images:
                            st.markdown("### 📷 Relevant Figures")
                            num_cols = min(len(valid_images), 3)
                            cols = st.columns(num_cols)
                            
                            for idx, img_data in enumerate(valid_images):
                                image_paths.append(img_data['path'])
                                image_pages.append(img_data['page'])
                                
                                with cols[idx % num_cols]:
                                    try:
                                        img = PILImage.open(img_data['path'])
                                        caption = img_data['caption'] if img_data['caption'] else f"Page {img_data['page']}"
                                        st.image(img, caption=caption, use_container_width=True)
                                    except Exception as e:
                                        logger.error(f"Error displaying image {img_data['path']}: {e}")
                                        st.error(f"Error loading image from page {img_data['page']}")
                        else:
                            # Image references exist but can't be loaded
                            st.warning(f"⚠️ Found {len(response['image_references'])} image reference(s) related to your question, but couldn't load the image files. The images may have been moved or deleted.")
                            logger.warning(f"Could not load any images from {len(response['image_references'])} image references")
                    else:
                        # No image references at all - tell user explicitly
                        st.info("ℹ️ No images found related to your question.")
                    
                    # Display citations
                    if response['citations']:
                        with st.expander("📄 Citations"):
                            for i, citation in enumerate(response['citations'], 1):
                                citation_type = "Table" if citation.get('is_table') else "Text"
                                st.markdown(f"**Citation {i}** ({citation_type}, Page {citation['page']}):")
                                
                                # Display table if it's a table citation
                                if citation.get('is_table') and citation.get('table_metadata'):
                                    table_meta = citation['table_metadata']
                                    if table_meta.get('table_html'):
                                        st.markdown("**Table:**")
                                        st.markdown(table_meta['table_html'], unsafe_allow_html=True)
                                    elif table_meta.get('table_markdown'):
                                        st.code(table_meta['table_markdown'], language='markdown')
                                else:
                                    st.text(citation['text'][:300] + "...")
                    
                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "images": image_paths,
                        "image_pages": image_pages,
                        "citations": response['citations']
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.exception(e)
                    logger.error(error_msg, exc_info=True)


if __name__ == "__main__":
    main()

