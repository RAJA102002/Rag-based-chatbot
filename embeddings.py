from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from utils.llm_utils import initialize_llm

def initialize_embeddings():
    """Initialize the embedding model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def create_vector_store(chunks, embeddings):
    """Create a FAISS vector store from document chunks"""
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_retriever(vector_store):
    """Create a retriever from the vector store"""
    # Basic retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Optional: Add contextual compression
    # llm = initialize_llm()
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=retriever
    # )
    # return compression_retriever
    