# load pdf 
# split into chunks 
# create the embeddings 
# store into chroma 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

# load PDF
data = PyPDFLoader("documents loaders/deeplearning.pdf")
docs = data.load()

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"   # fast & lightweight
)

# store into chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)