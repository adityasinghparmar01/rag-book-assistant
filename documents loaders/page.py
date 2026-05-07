from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"
# web length is also 1 because we are loading only one webpage 
data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content)