"""
Main chat application for CRISPR Historian using LangChain, Ollama, and Pinecone.
Enhanced with document source display and performance optimizations.
"""

import os
import time
import ollama                                                       # type: ignore
import streamlit as st                                              # type: ignore    
from dotenv import load_dotenv                                      # type: ignore
from pinecone import Pinecone                                       # type: ignore
from langchain.prompts import ChatPromptTemplate, PromptTemplate    # type: ignore
from langchain_community.chat_models import ChatOllama              # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings    # type: ignore
from langchain_pinecone import PineconeVectorStore                  # type: ignore
from langchain_core.documents import Document                       # type: ignore
from typing import List, Dict, Any
from utilities.icon import page_icon

# Load environment variables
load_dotenv()

# Hardcoded index configuration
PINECONE_INDEX_NAME = "llm-crispr"
PINECONE_NAMESPACE = ""  # Use default namespace

st.set_page_config(
    page_title="CRISPR Historian",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

class DocumentRetriever:
    """Custom retriever that captures retrieved documents for display"""
    
    def __init__(self, vectorstore, llm, k: int = 6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.last_retrieved_docs = []
        
        # Improved query prompt
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert at generating search queries for CRISPR research documents. 
            Generate 3 different search queries to find relevant information about the user's question.
            Focus on key scientific terms, researcher names, techniques, and applications.
            Make queries specific and technical when appropriate.
            
            Original question: {question}
            
            Search queries:"""
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents and store them for display"""
        try:
            # Generate multiple queries for better recall
            query_variations = self.llm.invoke(self.query_prompt.format(question=query))
            queries = [line.strip() for line in query_variations.content.split('\n') if line.strip()]
            
            # Add original query if not enough variations
            if len(queries) < 2:
                queries = [query] + queries
            else:
                queries = queries[:3]  # Limit to 3 queries
            
            all_docs = []
            seen_content = set()
            
            # Search with each query variation
            for search_query in queries:
                docs = self.vectorstore.similarity_search(
                    search_query, 
                    k=self.k,
                    filter={}  # Add any metadata filters here if needed
                )
                
                # Deduplicate based on content
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            
            # Sort by relevance (you could add scoring here)
            # For now, just take the first k unique documents
            self.last_retrieved_docs = all_docs[:self.k * 2]  # Store more for display
            
            return self.last_retrieved_docs[:self.k]  # Return top k for context
            
        except Exception as e:
            st.error(f"Error in document retrieval: {e}")
            self.last_retrieved_docs = []
            return []

# Initialize the complete RAG chain with Pinecone
@st.cache_resource
def initialize_rag_chain():
    """Initialize the complete RAG chain using Pinecone vector database"""
    
    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("🔑 PINECONE_API_KEY not found in environment variables!")
        st.error("Please add your Pinecone API key to your .env file")
        st.stop()

    try:
        # Initialize embeddings with optimized settings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Process embeddings in batches
            }
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        available_indexes = pc.list_indexes().names()
        if PINECONE_INDEX_NAME not in available_indexes:
            st.error(f"❌ Pinecone index '{PINECONE_INDEX_NAME}' not found!")
            st.error(f"Available indexes: {available_indexes}")
            st.error("Please create the index first using the PDF ingestion script.")
            st.stop()
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Get index stats for sidebar display
        stats = index.describe_index_stats()
        
        # Create PineconeVectorStore with optimized settings
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=PINECONE_NAMESPACE
        )

        # Initialize local LLM
        local_model = "gemma3:latest"
        llm = ChatOllama(
            model=local_model,
            temperature=0.1,  # Lower temperature for more focused responses
            num_ctx=4096      # Larger context window
        )

        # Create custom retriever
        retriever = DocumentRetriever(vectorstore, llm, k=6)

        # Enhanced context prompt template
        template = """You are a CRISPR technology expert. Use the provided context to answer the question comprehensively.

        Context from CRISPR research documents:
        {context}

        Question: {question}

        Instructions:
        1. Base your answer ONLY on the provided context
        2. Be specific and cite key details from the research
        3. If multiple sources provide different perspectives, mention them
        4. If the context lacks sufficient information, state this clearly
        5. Focus on scientific accuracy and technical details

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create a custom chain that captures retrieved documents
        def rag_chain_with_docs(question: str) -> Dict[str, Any]:
            # Get documents
            docs = retriever.get_relevant_documents(question)
            
            # Format context
            context = "\n\n".join([
                f"Source {i+1} (from {doc.metadata.get('source', 'unknown')} - Page {doc.metadata.get('page_num', 'unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            # Generate response
            response = llm.invoke(prompt.format_messages(context=context, question=question))
            
            return {
                'answer': response.content,
                'source_documents': retriever.last_retrieved_docs,
                'context_used': docs
            }
        
        return rag_chain_with_docs, stats, retriever
        
    except Exception as e:
        st.error(f"❌ Error initializing Pinecone RAG chain: {str(e)}")
        st.error("Please check your Pinecone configuration and try again.")
        st.stop()

@st.cache_resource
def initialize_regular_chat():
    """Initialize a regular chat without RAG for comparison"""
    from openai import OpenAI   # type: ignore
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def extract_model_names(models_info: list) -> tuple:
    """Extract model names from models information."""
    return tuple(model["model"] for model in models_info["models"])

def format_timing(elapsed_time: float) -> str:
    """Format elapsed time for display"""
    if elapsed_time < 1:
        return f"⏱️ {elapsed_time*1000:.0f}ms"
    elif elapsed_time < 60:
        return f"⏱️ {elapsed_time:.1f}s"
    else:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        return f"⏱️ {minutes}m {seconds:.1f}s"

def display_timing(elapsed_time: float):
    """Display timing information in gray text"""
    timing_text = format_timing(elapsed_time)
    st.markdown(f'<p style="color: gray; font-size: 0.8em; margin-top: 5px;">{timing_text}</p>', 
                unsafe_allow_html=True)

def display_source_documents(docs: List[Document], message_id: str = ""):
    """Display retrieved documents in an expandable format"""
    if not docs:
        return
    
    with st.expander(f"📚 View Source Documents ({len(docs)} found)", expanded=False):
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page_num', 'Unknown page')
            chunk_index = doc.metadata.get('chunk_index', '')
            
            # Create a header for each document
            st.markdown(f"**Document {i+1}:** {source} (Page {page})")
            
            # Show metadata if available
            if chunk_index:
                st.caption(f"Chunk {chunk_index}")
            
            # Show content in a code block for better formatting
            with st.container():
                st.markdown("**Content:**")
                content_preview = doc.page_content
                
                # Truncate very long content
                if len(content_preview) > 1000:
                    content_preview = content_preview[:1000] + "..."
                
                st.text_area(
                    f"Content from Document {i+1}",
                    content_preview,
                    height=150,
                    key=f"doc_content_{message_id}_{i}",
                    label_visibility="collapsed"
                )
            
            st.divider()

def main():
    """The main function that runs the application."""
    page_icon("🧬")
    st.subheader("CRISPR Historian Chatbot", divider="red", anchor=False)

    # Sidebar configuration
    st.sidebar.header("🧬 Configuration")
    
    # Display hardcoded index info
    st.sidebar.info(f"Using Index: **{PINECONE_INDEX_NAME}**")

    # Toggle for RAG vs regular chat
    use_rag = st.sidebar.toggle("Use CRISPR Knowledge Retrieval", value=True)
    
    if use_rag:
        st.sidebar.info("🧬 Using RAG with CRISPR knowledge base (Pinecone)")
        
        # Initialize RAG and get stats
        try:
            chain, stats, retriever = initialize_rag_chain()
            
            # Display Pinecone stats
            with st.sidebar.expander("📊 Knowledge Base Stats", expanded=True):
                st.metric("Total Documents", stats.get('total_vector_count', 'Unknown'))
                st.info("Using default namespace")
                
                # Show namespaces if available
                if 'namespaces' in stats and stats['namespaces']:
                    st.write("Available namespaces:")
                    for ns, info in stats['namespaces'].items():
                        ns_name = ns if ns else "default"
                        vector_count = info.get('vector_count', 0)
                        st.write(f"- {ns_name}: {vector_count} vectors")
                        
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Pinecone: {str(e)}")
            use_rag = False
    else:
        st.sidebar.info("💬 Using regular chat mode")

    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

    # Model selection
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if not available_models:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/settings.py") # pylint: disable=no-member
        return

    selected_model = st.selectbox(
        "Pick a model available locally on your system ↓", available_models
    )

    # Chat interface
    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            # Display timing for assistant messages if available
            if message["role"] == "assistant" and "timing" in message:
                display_timing(message["timing"])
            # Display source documents if available
            if message["role"] == "assistant" and "source_docs" in message:
                display_source_documents(message["source_docs"], f"msg_{idx}")

    # Handle new user input
    if prompt := st.chat_input("Ask a question about CRISPR..."):
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                start_time = time.time()
                
                if use_rag:
                    # Use RAG chain with Pinecone
                    with st.spinner("🔍 Searching CRISPR knowledge base..."):
                        try:
                            chain, _, _ = initialize_rag_chain()
                            result = chain(prompt)
                            
                            response = result['answer']
                            source_docs = result['source_documents']
                            
                            elapsed_time = time.time() - start_time
                            
                            st.markdown(response)
                            display_timing(elapsed_time)
                            
                            # Display source documents
                            current_message_id = f"current_{len(st.session_state.messages)}"
                            display_source_documents(source_docs, current_message_id)
                            
                            # Store response with source documents
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "timing": elapsed_time,
                                "source_docs": source_docs
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error retrieving from knowledge base: {str(e)}")
                            response = "Sorry, I encountered an error while searching the CRISPR knowledge base. Please try again or switch to regular chat mode."
                            elapsed_time = time.time() - start_time
                            display_timing(elapsed_time)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "timing": elapsed_time
                            })
                else:
                    # Regular chat without retrieval
                    with st.spinner("Generating response..."):
                        client = initialize_regular_chat()
                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                        response = st.write_stream(stream)
                        elapsed_time = time.time() - start_time

                        display_timing(elapsed_time)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timing": elapsed_time
                        })

        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"Error: {e}", icon="⛔️")
            if use_rag:
                st.error("Make sure your Pinecone index exists and contains CRISPR data.")
            else:
                st.error("Make sure Ollama is running and the selected model is available.")

if __name__ == "__main__":
    main()
