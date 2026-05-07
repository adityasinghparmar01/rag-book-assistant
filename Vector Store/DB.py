from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"})
]
# chroma khud se hi embedding generate kar deta hai bas usse batana padhta hai ki kiske through embeeding generate kar rahe ho
embedding_model = HuggingFaceEmbeddings()  # jyada kuch likhne ki jarurat nahi hai kyoki huggingface ka ek hi embedding model hai

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"    #ye chota hai to ham isse locally save kar sakte hai ,  yaha pe humne persist directory diya hai jaha pe chroma apna database create karega aur usme apne vector store ko save karega taki hum future me usse access kar sake. agar hum persist directory nahi dete hai to chroma sirf memory me hi vector store create karega aur jab hum program ko band kar denge to wo vector store bhi chala jayega. isliye persist directory dena jaruri hai taki humara vector store safe rahe aur future me use kar sake.
)

# sqlite me data store hoga like embedding vector, page content , documentid ,  metadata in sequence me store hoga aur jab bhi hum query karenge to chroma us sqlite database me se data fetch karega aur uske basis pe answer dega. isliye persist directory dena jaruri hai taki hamara vector store safe rahe aur future me use kar sake.
# we know ki vector store is not like a normal database jisme sequence me database ho jisme hum sql queries ke through data ko access kar sakte hai, in vector store data  instead data is stored in a way that it can be easily accessed and retrieved based on the similarity of the vectors. isliye jab bhi hum query karenge to chroma us sqlite database me se data fetch karega aur uske basis pe answer dega. isliye persist directory dena jaruri hai taki hamara vector store safe rahe aur future me use kar sake.
# vector store -> dta accesed by vector embedding -> similarity search 
# normal dta -> by sql queries like by name or id se ham data fetch kar sakte hai
# chroma db => sqlite database/ vector store => embedding save + creating a idx stucture for fast retrieval of data based on similarity search.

result = vectorstore.similarity_search("what is used for data analysis?" , k=2)
# vector store is not responsible for answering your question, it is responsible for fetching the relevant documents based on the similarity of the vectors. so when you ask a question like "what is used for data analysis?" , chroma will fetch the relevant documents from the vector store based on the similarity of the vectors and then you can use those documents to answer your question. in this case, it will fetch the document "Pandas is used for data analysis in Python." because it is similar to the question you asked.
# k => number of similar documents to fetch, in this case it will fetch 2 similar documents based on the similarity of the vectors.
# llm is responsible for answering your question based on the relevant documents fetched by the vector store. so when you ask a question like "what is used for data analysis?" , llm will use the relevant documents fetched by the vector store to answer your question. in this case, it will answer "Pandas is used for data analysis in Python." because it is similar to the question you asked.

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
 print(d.page_content) 