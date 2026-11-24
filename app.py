import time

import streamlit as st

from rag_engine import (
    MODEL,
    extract_text,
    create_chunks,
    create_vector_database,
    retrieve_documents,
    generate_answer,
)

st.set_page_config(
    page_title="Document AI - RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document AI Assistant")
st.caption("Retrieval-Augmented Generation using Gemini API")

st.markdown(
    "Upload your documents and ask questions. "
    "Answers are generated using retrieved document passages."
)

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved passages",
        min_value=1,
        max_value=8,
        value=4
    )

    st.divider()

    st.subheader("Project Information")
    st.write("Retrieval: TF-IDF + cosine similarity")
    st.write("Vector store: NumPy matrix")
    st.write(f"LLM: Google Gemini ({MODEL})")

    if st.session_state.sources:
        st.divider()
        st.subheader("Loaded documents")
        for name in st.session_state.sources:
            st.write(f"• {name}")

    if st.button("Clear knowledge base"):
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.sources = []
        st.rerun()

# Document upload
st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Process Documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        try:
            with st.spinner("Extracting and processing documents..."):

                all_documents = []

                for uploaded_file in uploaded_files:
                    documents = extract_text(uploaded_file)

                    if not documents:
                        st.warning(
                            f"No readable text in '{uploaded_file.name}' "
                            "(scanned PDFs need OCR)."
                        )

                    all_documents.extend(documents)

                if not all_documents:
                    st.error("No readable text found in documents.")

                else:
                    chunks = create_chunks(all_documents)

                    index, chunks = create_vector_database(chunks)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.sources = [
                        file.name for file in uploaded_files
                    ]

                    st.success("Documents processed successfully!")

                    col1, col2 = st.columns(2)
                    col1.metric("Documents", len(uploaded_files))
                    col2.metric("Text chunks", len(chunks))

        except Exception as e:
            st.error(f"Processing error: {e}")

# Question answering
st.divider()

st.header("2. Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="What does the uploaded document explain?"
)

if st.button("Generate Answer", type="primary"):

    if st.session_state.index is None:
        st.warning("Please process documents first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents and generating answer..."):

                t0 = time.time()

                retrieved = retrieve_documents(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=top_k
                )

                # Generation is wrapped separately so the retrieved
                # passages still show if Gemini is overloaded.
                try:
                    answer = generate_answer(question, retrieved)
                    answer_failed = False
                except Exception as e:
                    answer = f"Could not generate an answer: {e}"
                    answer_failed = True

                elapsed = time.time() - t0

            st.subheader("🤖 AI Answer")
            st.caption(f"Answered in {elapsed:.1f}s")

            if answer_failed:
                st.warning(answer)
            else:
                st.markdown(answer)

            st.divider()

            st.subheader("📖 Retrieved Passages")

            if retrieved:
                for i, item in enumerate(retrieved, start=1):

                    page = item["page"]
                    location = (
                        f"Page {page}" if page else "Text document"
                    )

                    with st.expander(
                        f"Source {i}: {item['source']} | "
                        f"{location} | "
                        f"Similarity: {item['score']:.3f}",
                        expanded=(i == 1)
                    ):
                        st.write(item["text"])

                        st.caption(
                            f"Chunk ID: {item['chunk_id']}"
                        )

            else:
                st.info("No passages were retrieved.")

        except Exception as e:
            st.error(f"Question answering error: {e}")

st.divider()

st.caption(
    "Document AI Assistant | Task 03 | "
    "Retrieval-Augmented Generation"
)