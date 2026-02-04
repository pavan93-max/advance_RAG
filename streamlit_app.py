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
        # Use OpenAI API (with token optimization)
        # Load from .env file or environment variable
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default to cheaper model
        max_context = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        max_response = int(os.getenv("MAX_RESPONSE_TOKENS", "500"))
        
        st.session_state.generator = LLMGenerator(
            model_type="openai",
            model_name=openai_model,
            openai_api_key=openai_key,
            max_context_tokens=max_context,
            max_response_tokens=max_response
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
    """Upload and process PDF documents."""
    st.header("📤 Upload Document")
    
    initialize_components()
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document (book, research paper, etc.)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        
        if not file_path.exists() or st.button("Re-process Document"):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
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
                    
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    st.exception(e)
        else:
            st.info(f"Document '{uploaded_file.name}' already uploaded. Click 'Re-process Document' to process again.")
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
                num_cols = min(len(message["images"]), 3)
                cols = st.columns(num_cols)
                for idx, img_path in enumerate(message["images"]):
                    if os.path.exists(img_path):
                        with cols[idx % len(cols)]:
                            img = PILImage.open(img_path)
                            st.image(img, caption=f"Page {message.get('image_pages', [])[idx] if 'image_pages' in message else ''}")
            
            # Display citations
            if "citations" in message:
                with st.expander("📄 Citations"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f"**Citation {i}** (Page {citation['page']}):")
                        st.text(citation['text'][:300] + "...")
    
    # User input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    # Retrieve (optimized for token usage - fewer chunks)
                    retrieval_results = st.session_state.retriever.retrieve(
                        prompt,
                        text_top_k=3,  # Reduced from 5 to save tokens
                        image_top_k=2  # Reduced from 3 to save tokens
                    )
                    
                    # Generate
                    response = st.session_state.generator.generate(
                        prompt,
                        retrieval_results['text_results'],
                        retrieval_results['image_results']
                    )
                    
                    # Display answer
                    st.markdown(response['answer'])
                    
                    # Display images
                    image_paths = []
                    image_pages = []
                    if response['image_references']:
                        st.markdown("### 📷 Relevant Figures")
                        cols = st.columns(min(len(response['image_references']), 3))
                        
                        for idx, img_ref in enumerate(response['image_references']):
                            img_path = img_ref.get('image_path', '')
                            if img_path and os.path.exists(img_path):
                                image_paths.append(img_path)
                                image_pages.append(img_ref.get('page', 0))
                                
                                with cols[idx % len(cols)]:
                                    img = PILImage.open(img_path)
                                    caption = img_ref.get('caption', '')
                                    if not caption:
                                        caption = f"Page {img_ref.get('page', 0)}"
                                    st.image(img, caption=caption)
                    
                    # Display citations
                    if response['citations']:
                        with st.expander("📄 Citations"):
                            for i, citation in enumerate(response['citations'], 1):
                                st.markdown(f"**Citation {i}** (Page {citation['page']}):")
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
                    st.error(f"Error generating response: {e}")
                    st.exception(e)


if __name__ == "__main__":
    main()

