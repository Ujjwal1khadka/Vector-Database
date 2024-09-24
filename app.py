from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages.base import BaseMessage
from fastapi import UploadFile, File, HTTPException
from langchain_openai import OpenAIEmbeddings
import uuid
from fastapi import Form
from fastapi import Query
import io  
from PyPDF2 import PdfReader
import docx
import shutil
import uvicorn
from typing import List, Dict
import numpy as np
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import (
PineconeHybridSearchRetriever)
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_core.runnables import RunnableParallel
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, validator, ValidationError
from tqdm.auto import tqdm
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse, JSONResponse
import pinecone
import glob
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from uuid import uuid4
import time



class MyModel(BaseModel):
    message: BaseMessage  


class Config:
    arbitrary_types_allowed = True


app = FastAPI(
    title="LangChain Server",
    version="o1",
    description="",
)
class Config:
    arbitrary_types_allowed = True


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')


if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
if pinecone_api_key:
    os.environ['PINECONE_API_KEY'] = pinecone_api_key


pinecone.Pinecone(api_key=pinecone_api_key, environment="us-east-1") 

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

directory = '/Users/user/Downloads/colab/data'
def load_docs(directory):
    loader = DirectoryLoader(directory)
    docs = loader.load()
    return docs

docs = load_docs(directory)

index_name = "test-2"  
pc = Pinecone(api_key=pinecone_api_key)

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pinecone.Index(index_name, host="https://test-2-5vwf04k.svc.aped-4627-b74a.pinecone.io")

chunk_size = 2000 
chunk_overlap = 50 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
split_docs = text_splitter.split_documents(docs)

index_name = "test-2"
vectorstore = PineconeVectorStore.from_documents(split_docs, embeddings, index_name=index_name)
keyword = """ """
retriever = vectorstore.as_retriever(search_kwargs = {"k":3})
retriever.get_relevant_documents(keyword)
retriever

llm = ChatOpenAI(model="gpt-4o", temperature=1.0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True  # This will return source documents in the response
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieved_docs = retriever.invoke(""" How to create a membership?""")
print(format_docs(retrieved_docs))


template = """You are an expert LLM assistant specialized in answering questions based solely on the information provided in the uploaded documents (PDF, DOCX, or TXT formats). Use only the information from the documents to respond accurately and clearly to each question.

Guidelines:
1. Provide concise and informative answers.
2. If the answer is not found in the uploaded documents, state, "The answer is not specifically mentioned in the provided documents."
3. Avoid using outside knowledge or assumptions. Stick strictly to the content in the documents.
4. Maintain a professional and helpful tone thinking you are giving service to the customer for their documents 
5. Answer for normal conversation question like "Hi", "Hey", "Hello", "How are you", and many others questions with answer "Hello, How can I assist you?".
6. If question is on "summarize" or "summerization", then summarize the documents and (1/4)th the size of documents.

Question: {question}

Context: {context}

Answer:
"""
prompt = template.format(question = keyword, context =  format_docs(retrieved_docs))

custom_rag_template = PromptTemplate.from_template(template)

# Create the parallel chain
My_rag_chain = RunnableParallel(
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
) | custom_rag_template | llm | StrOutputParser()


## My chain : Retriever(Pinecone) | custom_rag_template(prompt) | llm | StrOutputParser()

#$$$$$$$$$$ DOCS $$$$$$$$$
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


class Tenant_ID(BaseModel):
    tenant_ID: str
#$$$$$$$$$ UPLOAD $$$$$$$$$
@app.post("/upload")
async def upload_files(
    tenant_ID: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    """Upload multiple PDF, DOCX, and TXT files"""
    
    # Create a directory for the tenant if it doesn't exist
    base_directory = 'data'
    tenant_directory = os.path.join(base_directory, tenant_ID)
    os.makedirs(tenant_directory, exist_ok=True)

    # Allowed file extensions
    allowed_extensions = {".pdf", ".docx", ".txt"}

    # Track file names to check for duplicates
    file_names = set()

    # Save each uploaded file
    for file in files:
        if file.filename in file_names:
            raise HTTPException(status_code=400, detail=f"Duplicate file detected: {file.filename}")
        
        file_names.add(file.filename)
        
        # Check file extension
        _, extension = os.path.splitext(file.filename)
        if extension.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only PDF, DOCX, and TXT files are allowed.")

        # Define the destination path for the uploaded file
        destination = os.path.join(tenant_directory, file.filename)
        
        # Save the uploaded file to the tenant's directory
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Load documents from the tenant's directory after all files are uploaded
    docs = load_docs(tenant_directory)

    # Prepare texts and metadata for each document
    texts = [doc.page_content for doc in docs]  # Assuming docs have a page_content attribute
    metadata = [{"source": os.path.basename(doc.metadata['source']), "text": doc.page_content} for doc in docs]

    # Generate unique IDs for each document using uuid4()
    ids = [str(uuid4()) for _ in range(len(texts))]
    
    # Create a structured list of documents with IDs and names
    documents = [
        {
            "id": doc_id,
            "docNumber": f"doc{i + 1}",
            "fileName": meta["source"]
        }
        for i, (doc_id, meta) in enumerate(zip(ids, metadata))
    ]

    print("Generated Document IDs and their corresponding texts:")
    for doc in documents:
        print(f"ID: {doc['id']}, Source: {doc['fileName']}, Text: {texts[documents.index(doc)][:30]}...")  # Print first 30 characters of text

    # Initialize the Pinecone Vector Store using LangChain's Pinecone wrapper
    vector_store = Pinecone(index=index, embedding=embeddings, text_key=texts)

    return JSONResponse(content={"message": "Files uploaded successfully.", "documents": documents}, status_code=200)


#$$$$$$$$$$$ PROMPTS $$$$$$$$$$$
@app.get("/prompts")
async def prompts_keyword(keyword):
    try:
        response = qa.invoke(keyword)
        result = response.get('result', 'No result found')
        source_documents = response.get('source_documents', 'No authentic source documents available')
    except Exception as e:
        return {"error": str(e)}
    
    return {"result": result, "source_documents": source_documents}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)














