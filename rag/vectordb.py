#!/usr/bin/env python3
"""
PDF to Vector Database Converter

This script converts a collection of PDF files into a searchable vector database
using LangChain and Chroma. It processes PDFs efficiently with memory management
and progress tracking.
"""

import os
import gc
import sys
import logging
import argparse
from typing import List, Generator
from pathlib import Path
from tqdm import tqdm

# Fix for SQLite version - must be done before other imports
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader            # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter     # type: ignore
from langchain_community.vectorstores import Chroma                     # type: ignore 
from langchain_core.documents import Document                           # type: ignore   
from langchain_community.embeddings import HuggingFaceEmbeddings        # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vectordb_creation.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert PDFs to a vector database")
    parser.add_argument(
        "--pdf_dir", 
        type=str,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="vector_stores",
        help="Output directory for the vector store"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=5000,
        help="Size of text chunks"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=500,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for adding documents to the vector store"
    )
    return parser.parse_args()

def load_pdf_generator(
    pdf_files: List[str], 
    chunk_size: int = 5000, 
    chunk_overlap: int = 500
) -> Generator[List[Document], None, None]:
    """
    Generator function that loads PDFs one at a time, chunks them, and yields the chunks.
    
    Args:
        pdf_files: List of PDF file paths
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Yields:
        A list of document chunks from each PDF
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Load the PDF
            loader = PyPDFLoader(file_path=pdf_path)
            data = loader.load()
            
            if not data:
                logger.warning(f"No content extracted from {pdf_path}")
                continue
                
            # Split into chunks
            chunks = text_splitter.split_documents(data)
            
            # Add source filename metadata if not already present
            for chunk in chunks:
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = os.path.basename(pdf_path)
                # Add page numbers if available
                if 'page' in chunk.metadata:
                    chunk.metadata['page_num'] = chunk.metadata['page'] + 1
            
            logger.info(f"Generated {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            
            # Yield the chunks for this PDF
            yield chunks
            
            # Clear memory
            del data, chunks
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            continue

def create_vectordb(args):
    """Main function to create the vector database."""
    # Ensure directories exist
    pdf_directory = Path(args.pdf_dir)
    persist_directory = Path(args.output_dir)
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding model
    logger.info(f"Initializing embedding model: {args.model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)
    
    # List all PDF files
    pdf_files = list(pdf_directory.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        return
    
    # Create or load the vector store
    if os.path.exists(persist_directory / "chroma.sqlite3"):
        logger.info(f"Loading existing vector store from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings
        )
    else:
        logger.info(f"Creating new vector store at {persist_directory}")
        vectorstore = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings
        )
    
    # Process PDFs in batches using the generator
    pdf_generator = load_pdf_generator(pdf_files, args.chunk_size, args.chunk_overlap)
    
    total_chunks = 0
    total_processed = 0
    batch_documents = []
    
    for chunks in pdf_generator:
        if not chunks:
            continue
            
        batch_documents.extend(chunks)
        total_processed += 1
        
        # Process in batches to manage memory
        if len(batch_documents) >= args.batch_size or total_processed == len(pdf_files):
            try:
                logger.info(f"Adding batch of {len(batch_documents)} chunks to vector store")
                vectorstore.add_documents(batch_documents)
                vectorstore.persist()
                
                total_chunks += len(batch_documents)
                batch_documents = []
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
                # Continue with next batch
                batch_documents = []
    
    # Final persist
    vectorstore.persist()
    logger.info(f"Vector database creation complete")
    logger.info(f"Processed {total_processed}/{len(pdf_files)} PDFs")
    logger.info(f"Added {total_chunks} total chunks to the vector store")
    
    return total_chunks

def main():
    """Main entry point."""
    args = parse_arguments()
    logger.info("Starting PDF to Vector Database conversion")
    
    try:
        total_chunks = create_vectordb(args)
        logger.info(f"Successfully created vector database with {total_chunks} chunks")
    except Exception as e:
        logger.critical(f"Fatal error during vector database creation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
