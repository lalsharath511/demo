    texts=get_text_chunks()
    # embeddings = OpenAIEmbeddings()
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectordb= FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectordb=Chroma.from_documents(texts,embedding=embedding_function,persist_directory=persist_directory )
    vectordb.persist()
    vectordb=None
    
    
    
    
import pymongo
import sys
import os
import getpass
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GooglePalm
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter



api_key="AIzaSyAZwZy6l2GvsNH0hmoMr-WsCMGxVSGzhOs"
data=[]
class Document:
    def __init__(self, page_content,metadata):
        self.page_content = page_content
        self.metadata = metadata
       

client = pymongo.MongoClient("mongodb+srv://lalsharath511:Sharathbhagavan15192142@legal.mosm3f4.mongodb.net/")
database = client["lex_learn"]
collection = database["all_acts_data"]

# Specify the fields you want to retrieve
projection = {"title": 1, "html_content": 1, "_id": 0}  # 1 means include, 0 means exclude

# Retrieve documents with specified fields
result = collection.find({}, projection)
data=[]
for document in result:
    data.append(Document(page_content=f"{document['title']}{document['html_content']}", metadata={"source":document['title']}))
    # item={"page_content":f"{document['title']}{document['html_content']}"}
   
    

# print(doc.page_content)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(data)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb= FAISS.from_texts(texts=text_chunks, embedding=embeddings)
vectordb=Chroma.from_documents(texts,embedding=embedding_function,persist_directory="./data" )
vectordb.persist()
vectordb=None

# vectordb = Chroma.from_documents(chunks, embedding=GooglePalmEmbeddings(google_api_key=api_key), persist_directory="./data")

# # persist the vector database
# vectordb.persist()

# create a ConversationalRetrievalChain object for PDF question answering
pdf_qa = ConversationalRetrievalChain.from_llm(
    GooglePalm(google_api_key=api_key),  # use GooglePalm for language modeling
    vectordb.as_retriever(search_kwargs={'k': 6}),  # use the Chroma vector store for document retrieval
    return_source_documents=True,  # return the source documents along with the answers
    verbose=False  # do not print verbose output
)
chat_history = []
# print a welcome message to the user
print("---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
# create an interactive loop to prompt the user for questions
while True:
    # prompt the user for a question
    query = input("Prompt: ")
    
    # check if the user wants to exit the loop
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    
    # check if the user entered an empty query
    if query == '':
        continue
    
    # use the ConversationalRetrievalChain to find the answer to the user's question
    result = pdf_qa({"question": query, "chat_history": chat_history})
    
    # print the answer to the user
    print("Answer: " + result["answer"])
    
    # add the user's question and the resulting answer to the chat history
    chat_history.append((query, result["answer"]))