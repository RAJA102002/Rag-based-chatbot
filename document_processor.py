from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_file):
    """Extract text from a text file"""
    return txt_file.read().decode("utf-8")

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF or text files and return chunks"""
    documents = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = extract_text_from_txt(uploaded_file)
        else:
            continue
        
        documents.append(text)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.create_documents(documents)