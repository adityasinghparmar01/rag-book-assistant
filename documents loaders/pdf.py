from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("documents loaders/GRU.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # taking hundreds tokens at a time because of token limit and to maintain the context of the text.
    chunk_overlap=10
    )
docs = data.load()

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)