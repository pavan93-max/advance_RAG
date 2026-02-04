# OmniRAG - Multimodal RAG System

A local, multimodal RAG (Retrieval-Augmented Generation) system that processes PDF documents and answers questions with text, images, and citations.

## Features

- 📄 **PDF Processing**: Extracts text, images, and tables from PDFs
- 🖼️ **Image Understanding**: OCR, VLM captioning, and CLIP embeddings for images without captions
- 🔍 **Hybrid Retrieval**: Semantic + keyword search for text, multi-strategy image search
- 💬 **Chat Interface**: Ask questions and get answers with citations and relevant figures
- 🏠 **Fully Local**: Runs entirely on your laptop, no cloud dependencies

## Architecture

```
PDF → Parser → Text Blocks + Images + Tables
                ↓
        Image Processor (OCR + VLM + CLIP)
                ↓
        Vector Store (Chroma)
                ↓
        Hybrid Retriever
                ↓
        LLM Generator → Answer + Citations + Images
```

## Installation

**📖 For complete installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Quick Install

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR** (required, not a Python package):
   - **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux:** `sudo apt-get install tesseract-ocr`
   - **macOS:** `brew install tesseract`

3. **Run setup check:**
   ```bash
   python setup.py
   ```

4. **Start the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

### What Gets Installed

- ✅ **Python packages** (~2-3 GB): From `requirements.txt`
- ✅ **Tesseract OCR** (~50-100 MB): Manual install (see above)
- ✅ **ML Models** (~1.2 GB): Auto-downloaded on first run
  - CLIP: ~150 MB
  - BLIP: ~990 MB
  - Sentence Transformers: ~80 MB
- ⚠️ **Ollama** (optional, ~2-4 GB): For better LLM answers

**Total minimum:** ~4-5 GB disk space required

### Setup OpenAI API Key

The system uses OpenAI API for generating answers. You need an API key:

**Option 1: Using .env file (Recommended)**
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=your-key-here
   ```

**Option 2: Environment variable**
1. Get API key from: https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"  # Linux/Mac
   set OPENAI_API_KEY=your-key-here        # Windows
   ```

**Token Optimization:**
- Uses `gpt-3.5-turbo` (cheaper than gpt-4)
- Limits context to ~3000 tokens
- Limits response to 500 tokens
- Estimated cost: ~$0.002-0.005 per query

If API key is not set, the system will use a template-based fallback.

## Usage

### 1. Start the Streamlit App

```bash
streamlit run streamlit_app.py
```

### 2. Upload a PDF

1. Go to the **Upload Document** page
2. Upload a PDF file (book, research paper, etc.)
3. Wait for processing to complete (may take several minutes for large PDFs)

### 3. Ask Questions

1. Go to the **Chat** page
2. Type your question
3. Get answers with:
   - Text response
   - Relevant images/figures
   - Page number citations

## How It Works

### Ingestion Pipeline

1. **PDF Parsing**: Extracts text blocks, images, and tables with page numbers
2. **Image Processing**:
   - If caption exists → use it
   - If no caption:
     - Extract surrounding text context
     - Run OCR to extract text from image
     - Generate synthetic caption using BLIP VLM
     - Create CLIP embedding for similarity search
3. **Storage**: Stores everything in Chroma vector database

### Retrieval

1. **Text Search**:
   - Semantic search using sentence transformers
   - Keyword matching (BM25-like)
   - Re-ranking

2. **Image Search**:
   - CLIP similarity (text-to-image)
   - Caption/OCR text matching
   - Images linked to retrieved text chunks

### Generation

- Uses retrieved text and images
- Generates answer with citations
- References figures explicitly
- Displays images in UI

## Project Structure

```
omnirag/
├── ingestion/
│   ├── pdf_parser.py      # PDF parsing (text, images, tables)
│   └── pipeline.py         # Complete ingestion pipeline
├── retrieval/
│   └── retriever.py        # Hybrid retrieval logic
├── models/
│   ├── image_processor.py  # OCR, VLM, CLIP
│   └── llm_generator.py    # LLM generation with citations
├── db/
│   └── vector_store.py     # Chroma vector database
└── utils/
streamlit_app.py            # Streamlit UI
requirements.txt
README.md
```

## Configuration

### Model Selection

Edit `streamlit_app.py` to change the LLM:

```python
st.session_state.generator = LLMGenerator(
    model_type="ollama",  # or "transformers"
    model_name="llama3.2"  # or "mistral", etc.
)
```

### Device Selection

For GPU acceleration, modify `image_processor.py`:

```python
image_processor = ImageProcessor(device="cuda")  # if GPU available
```

## Troubleshooting

### Tesseract OCR Not Found

Set the path to Tesseract:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### Out of Memory

- Process smaller PDFs
- Reduce batch sizes in image processing
- Use CPU instead of GPU

### Slow Processing

- Large PDFs (1000+ pages) may take 30+ minutes
- First-time model loading takes time
- Consider processing in batches

## Limitations

- Processing large PDFs can be slow
- OCR quality depends on image quality
- VLM captions may not always be perfect
- Requires significant RAM for large documents

## Example Use Cases

- 📚 Research paper Q&A
- 📖 Textbook question answering
- 📊 Technical document analysis
- 🖼️ Figure and diagram retrieval

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Areas for improvement:
- Better chunking strategies
- More VLM options
- Improved reranking
- Batch processing
- Multi-document support

