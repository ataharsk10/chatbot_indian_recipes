# Push vector to vectorDB
from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#Load data
extracted_data = load_pdf("data/")

#Chunk Data
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

#Embedding
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

#Creating Embeddings for Each of The Text Chunks & STORING
index_name="langchain-chatbot"
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

