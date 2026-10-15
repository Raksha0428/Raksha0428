
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def main():
    pdf_path = input(
        "Enter PDF path (or press Enter for ril.pdf): "
    ).strip()

    if not pdf_path:
        pdf_path = "ril.pdf"

    # 1. Load the PDF
    print("\nLoading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        print("No text found in the PDF.")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. Split the PDF into smaller chunks
    print("Splitting PDF text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create local embeddings and searchable vector store
    print("Loading embedding model (first run may take time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("\nPDF chatbot is ready!")
    print("Type 'exit' to stop.\n")

    # 4. Ask questions
    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            relevant_docs = vector_store.similarity_search(
                question,
                k=3
            )

            context_parts = []
            for doc in relevant_docs:
                page_number = doc.metadata.get("page", 0) + 1
                context_parts.append(
                    f"Page {page_number}:\n{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
Answer the question using only the PDF context below.
If the answer is not present, say you could not find it in the PDF.
Do not invent facts.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            answer = response.json()["response"].strip()

            print("\nAnswer:")
            print(answer)

            pages = sorted({
                doc.metadata.get("page", 0) + 1
                for doc in relevant_docs
            })
            print("\nSource pages:", ", ".join(map(str, pages)))
            print()

        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Make sure Ollama is running.")
        except requests.exceptions.RequestException as error:
            print(f"Ollama request failed: {error}")
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
