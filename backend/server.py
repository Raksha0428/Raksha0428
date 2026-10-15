from io import BytesIO

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="PDF Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development. Restrict this before public deployment.
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The uploaded document's vector store stays in memory while this server runs.
vector_store = None
uploaded_filename = None
uploaded_pages = 0

# Load the embedding model once when the server starts.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@app.get("/")
def home():
    return {"message": "PDF Chatbot API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, uploaded_filename, uploaded_pages

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        reader = PdfReader(BytesIO(content))
        documents = []

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_index + 1,
                            "source": file.filename,
                        },
                    )
                )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this PDF: {exc}",
        ) from exc

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. This may be a scanned PDF and need OCR.",
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not split PDF text.")

    try:
        # Replace the previous document so questions use the newly uploaded PDF.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not index PDF text: {exc}",
        ) from exc

    uploaded_filename = file.filename
    uploaded_pages = len(reader.pages)

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": uploaded_filename,
        "pages": uploaded_pages,
        "text_chunks": len(chunks),
    }


@app.post("/chat")
async def chat(data: dict):
    global vector_store

    question = str(data.get("question", "")).strip()

    if vector_store is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question.")

    try:
        # Retrieve more than one relevant chunk to improve context coverage.
        relevant_docs = vector_store.similarity_search(question, k=5)
        if not relevant_docs:
            return {
                "answer": "I could not find relevant information in the uploaded PDF.",
                "source_pages": [],
            }

        context_parts = []
        for doc in relevant_docs:
            page = doc.metadata.get("page", "unknown")
            context_parts.append(f"[Page {page}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a careful assistant answering questions about one uploaded PDF.

Rules:
- Answer the user's question using only the PDF excerpts provided below.
- Be direct, specific, and complete. Include important names, dates, figures, and steps when they appear in the excerpts.
- Do not use outside knowledge or invent missing details.
- If the excerpts do not contain enough information, clearly say what could not be found.
- If the question asks for a summary, summarize the relevant information in clear bullet points.
- Cite the page number(s) in your answer like [Page 2].

PDF EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The local AI model returned an empty answer.",
            )

        source_pages = sorted({
            int(doc.metadata["page"])
            for doc in relevant_docs
            if str(doc.metadata.get("page", "")).isdigit()
        })

        return {
            "answer": answer,
            "source_pages": source_pages,
            "filename": uploaded_filename,
        }

    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Start Ollama and make sure llama3.2:3b is installed.",
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="The AI model took too long to respond. Try a shorter question or a smaller PDF.",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        ) from exc

