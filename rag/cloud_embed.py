from pinecone import Pinecone, ServerlessSpec   # type: ignore
from dotenv import load_dotenv                  # type: ignore

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

# Load PINECONE_API_KEY from dotenv
load_dotenv()

# Ensure PINECONE_API_KEY is set
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

# TODO
