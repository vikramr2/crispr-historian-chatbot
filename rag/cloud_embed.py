#!/usr/bin/env python3
"""
PDF to Pinecone Vector Database Converter

This script converts a collection of PDF files into a searchable Pinecone vector database
using LangChain and Pinecone. It processes PDFs efficiently with memory management
and progress tracking.
"""

# pylint: disable=logging-fstring-interpolation,broad-except,no-name-in-module

import os
import gc
import sys
import logging
import argparse
import hashlib
from typing import List, Generator, Dict, Any
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec   # type: ignore pylint: disable=no-name-in-module
from dotenv import load_dotenv                  # type: ignore
from tqdm import tqdm

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader            # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings        # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter     # type: ignore
from langchain_pinecone import PineconeVectorStore                      # type: ignore
from langchain_core.documents import Document                           # type: ignore   

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

# Load environment variables
load_dotenv()

# Ensure PINECONE_API_KEY is set
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert PDFs to a Pinecone vector database")
    parser.add_argument(
        "--pdf_dir", 
        type=str,
        required=True,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        required=True,
        help="Name of the Pinecone index"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000 for better Pinecone performance)"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int,
        default=200,
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
        default=100,
        help="Batch size for adding documents to Pinecone"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Vector dimension"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dotproduct"],
        help="Distance metric for Pinecone index"
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default="aws",
        choices=["aws", "gcp", "azure"],
        help="Cloud provider for Pinecone serverless"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Cloud region for Pinecone serverless"
    )
    parser.add_argument(
        "--namespace", 
        type=str,
        default="",
        help="Pinecone namespace (optional)"
    )
    return parser.parse_args()

def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """Generate a unique ID for a document chunk."""
    # Create a hash based on content and key metadata
    source = metadata.get('source', '')
    page = metadata.get('page_num', 0)
    hash_input = f"{source}_{page}_{content[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def load_pdf_generator(
    pdf_files: List[str], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
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
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
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
            
            # Add enhanced metadata
            for i, chunk in enumerate(chunks):
                # Ensure source filename is present
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = os.path.basename(pdf_path)
                
                # Add page numbers if available
                if 'page' in chunk.metadata:
                    chunk.metadata['page_num'] = chunk.metadata['page'] + 1
                else:
                    chunk.metadata['page_num'] = 1
                
                # Add chunk index within the document
                chunk.metadata['chunk_index'] = i
                
                # Add document length for context
                chunk.metadata['total_chunks'] = len(chunks)
                
                # Generate unique ID for this chunk
                chunk.metadata['id'] = generate_document_id(chunk.page_content, chunk.metadata)
            
            logger.info(f"Generated {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            
            # Yield the chunks for this PDF
            yield chunks
            
            # Clear memory
            del data, chunks
            gc.collect()
            
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            continue

def setup_pinecone_index(args, embeddings):
    """Set up Pinecone index and return PineconeVectorStore."""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if args.index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {args.index_name}")
        pc.create_index(
            name=args.index_name,
            dimension=args.dimension,
            metric=args.metric,
            spec=ServerlessSpec(
                cloud=args.cloud,
                region=args.region
            )
        )
        logger.info(f"Index '{args.index_name}' created successfully")
    else:
        logger.info(f"Using existing Pinecone index: {args.index_name}")
    
    # Get index stats
    index = pc.Index(args.index_name)
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")
    
    # Create PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=args.namespace
    )
    
    return vectorstore, index

def create_vectordb(args):
    """Main function to create the vector database."""
    # Ensure PDF directory exists
    pdf_directory = Path(args.pdf_dir)
    if not pdf_directory.exists():
        logger.error(f"PDF directory {pdf_directory} does not exist")
        return 0
    
    # Initialize embedding model
    logger.info(f"Initializing embedding model: {args.model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.model_name,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Test embedding dimension
    test_embedding = embeddings.embed_query("test")
    actual_dimension = len(test_embedding)
    
    if actual_dimension != args.dimension:
        logger.warning(f"Embedding dimension mismatch: expected {args.dimension}, got {actual_dimension}")
        logger.info(f"Using actual dimension: {actual_dimension}")
        args.dimension = actual_dimension
    
    # Set up Pinecone
    vectorstore, index = setup_pinecone_index(args, embeddings)
    
    # List all PDF files
    pdf_files = list(pdf_directory.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        return 0
    
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
        
        # Process in batches to manage memory and API limits
        if len(batch_documents) >= args.batch_size or total_processed == len(pdf_files):
            try:
                logger.info(f"Adding batch of {len(batch_documents)} chunks to Pinecone")
                
                # Add documents to Pinecone
                vectorstore.add_documents(
                    documents=batch_documents,
                    ids=[doc.metadata['id'] for doc in batch_documents]
                )
                
                total_chunks += len(batch_documents)
                logger.info(f"Successfully added {len(batch_documents)} chunks. Total: {total_chunks}")
                
                batch_documents = []
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to add documents to Pinecone: {e}", exc_info=True)
                # Continue with next batch
                batch_documents = []
    
    # Get final index stats
    final_stats = index.describe_index_stats()
    logger.info(f"Final index stats: {final_stats}")
    logger.info(f"Vector database creation complete")
    logger.info(f"Processed {total_processed}/{len(pdf_files)} PDFs")
    logger.info(f"Added {total_chunks} total chunks to Pinecone")
    
    return total_chunks

def main():
    """Main entry point."""
    args = parse_arguments()
    logger.info("Starting PDF to Pinecone Vector Database conversion")
    logger.info(f"Target index: {args.index_name}")
    logger.info(f"Namespace: {args.namespace if args.namespace else 'default'}")
    
    try:
        total_chunks = create_vectordb(args)
        logger.info(f"Successfully created Pinecone vector database with {total_chunks} chunks")
    except Exception as e:
        logger.critical(f"Fatal error during vector database creation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
