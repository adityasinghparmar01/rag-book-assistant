import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant by AdiTech")
st.write("Upload a PDF and ask questions from the document")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


# Create Vector Database
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            try:
                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Check if PDF has readable text
                if not docs:
                    st.error("No content found in PDF.")
                    st.stop()

                # Remove empty pages
                docs = [
                    doc for doc in docs
                    if doc.page_content.strip()
                ]

                if not docs:
                    st.error(
                        "PDF contains no readable text. "
                        "It may be scanned/image-based."
                    )
                    st.stop()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                # Check chunks
                if not chunks:
                    st.error("No text chunks created from PDF.")
                    st.stop()

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Create vector database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

                vectorstore.persist()

                st.success("Vector database created successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

            finally:
                # Remove temp file
                if os.path.exists(file_path):
                    os.remove(file_path)


# Load Existing Vector DB
if os.path.exists("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        try:
            docs = retriever.invoke(query)

            if not docs:
                st.warning("No relevant content found.")
                st.stop()

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {str(e)}")