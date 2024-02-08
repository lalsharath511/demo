import pymongo
import sys
import os
import getpass
from langchain_community.embeddings import GooglePalmEmbeddings,HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GooglePalm
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
import json
from langchain_google_vertexai import VertexAI
# from langchain.prompts.chat import PromptTemplat



api_key="AIzaSyAZwZy6l2GvsNH0hmoMr-WsCMGxVSGzhOs"
# data=[]
# class Document:
#     def __init__(self, page_content,metadata):
#         self.page_content = page_content
#         self.metadata = metadata
       

# client = pymongo.MongoClient("mongodb+srv://lalsharath511:Sharathbhagavan15192142@legal.mosm3f4.mongodb.net/")
# database = client["lex_learn"]
# collection = database["all_acts_data"]

# # Specify the fields you want to retrieve
# projection = {"title": 1, "html_content": 1, "_id": 0}  # 1 means include, 0 means exclude

# # Retrieve documents with specified fields
# result = collection.find({}, projection)
# data=[]
# with open("output.json", 'r') as file:
#     loaded_data = json.load(file)

# # Create a list of Document objects
# documents = []
# for document in loaded_data:
#     documents.append(Document(page_content=f"{document['title']}{document['html_content']}", metadata={"source": document['title']}))
#     # item={"page_content":f"{document['title']}{document['html_content']}"}
   
    

# # print(doc.page_content)
# text_splitter = CharacterTextSplitter(
#     separator=" ",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )
# chunks = text_splitter.split_documents(documents)
# # import json
# # with open("as.txt", 'r',encoding='utf-8') as file:
# #     content = file.read()
# embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# # vectordb= FAISS.from_texts(texts=text_chunks, embedding=embeddings)
# vectordb=Chroma.from_documents(chunks,embedding=embedding_function,persist_directory="./data" )
# vectordb.persist()
# vectordb=None
    
# embedding_function = GooglePalmEmbeddings(google_api_key=api_key)
# vectordb=Chroma(persist_directory="./data" ,embedding_function=embedding_function)
# vectordb = Chroma.from_documents(chunks, embedding=GooglePalmEmbeddings(google_api_key=api_key), persist_directory="./data")

# # # persist the vector database
# vectordb.persist()
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb=Chroma(persist_directory="./data" ,embedding_function=embedding_function)
# create a ConversationalRetrievalChain object for PDF question answering
llm = VertexAI(
    api_key=api_key,
    model_name="text-bison@001",
    max_output_tokens=1000,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

template = """SYSTEM: You are an intelligent assistant helping the users with their question based on insident and provide.

Insident: {question}
Find Indian act related to the Insident from the context and give a Professional Opinion on income tax useing the context.
Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
 - If the context is empty, just say "I do not know the answer to that."

=============
{context}
=============


Insident: {question}
Helpful Answer example:    **Professional Opinion on Income Tax**

        1. **Introduction:**
           Provide an introduction outlining the purpose of the opinion and a brief summary of the issues to be addressed.

        2. **Factual Background:**
           Describe the relevant facts and circumstances surrounding the issue at hand. This is crucial for understanding the context in which the opinion is being sought.

        3. **Legal Analysis:**
           Perform a detailed analysis of the relevant provisions of the Income Tax Act, 1961, along with any applicable case law or judicial precedents. Tailor the analysis to the specific situation to provide a clear understanding of how the law applies.

        4. **Conclusion:**
           Conclude with a clear and concise summary of the legal position and the recommended course of action based on the analysis. This should be the key takeaway from the opinion.

        5. **Recommendations:**
           If applicable, provide recommendations for action or further steps that should be taken based on the analysis and conclusion.

       """
pdf_qa = RetrievalQA.from_chain_type(
    llm=llm,  # use GooglePalm for language modeling
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={'k':6}),  # use the Chroma vector store for document retrieval
    return_source_documents=True,  # return the source documents along with the answers
    verbose=False,  # do not print verbose output
        chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    },
)
chat_history = []
# result = pdf_qa({"query":"Powers regarding discovery and production of evidence"})
# print(result)
# print a welcome message to the user
print("---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
# create an interactive loop to prompt the user for questions
# chat_history=""
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
    result = pdf_qa({"query": query})
    
    # print the answer to the user
    print("Answer: " + result['result'])
    
    # add the user's question and the resulting answer to the chat history
