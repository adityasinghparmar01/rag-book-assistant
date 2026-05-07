from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="", # if we write \n then it will split the text at every new line and if we write "" then it will split the text at every character. so we are using "" to split the text at every character and then we will use chunk_size to create chunks of 10 characters.
    chunk_size=10, 
    chunk_overlap=1
    # whwen seperator is "\n" than it creates and seperates when line changes and form one line as a one chunk nd no use of chunk size and chunk overlap but when seperator is "" then it creates chunks of 10 characters and overlap of 1 character so that we can maintain the context of the text.
    )
data = TextLoader("documents loaders/notes.txt")

docs = data.load()
# there are 
chunks = splitter.split_documents(docs)
 
for i in chunks:
    print(i.page_content)
    print()