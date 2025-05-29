"""
Main application for the CRISPR Historian chatbot.

This application uses an enhanced RAG (Retrieval-Augmented Generation) approach
to provide precise answers about CRISPR technology, its history, and related topics.
"""

import os
import time
from typing import List, Dict, Any
import ollama
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from openai import OpenAI
from utilities.icon import page_icon

# Load environment variables
load_dotenv()

# Hardcoded index configuration
PINECONE_INDEX_NAME = "llm-crispr"
PINECONE_NAMESPACE = ""

st.set_page_config(
    page_title="CRISPR Historian",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

class EnhancedDocumentRetriever:
    """Enhanced retriever with better context handling and source attribution"""

    def __init__(self, vectorstore, llm, k: int = 6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.last_retrieved_docs = []

        # Improved query prompt
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 precise search queries for CRISPR research documents.
            Focus on exact terms, names, and concepts from the question.
            Avoid combining unrelated concepts in a single query.
            
            Question: {question}
            
            Search queries (one per line):"""
        )

    def clean_and_annotate_chunks(self, docs: List[Document]) -> List[Document]:
        """Clean chunks and add clear source attribution"""
        cleaned_docs = []
        for i, doc in enumerate(docs):
            # Create enhanced metadata
            source = doc.metadata.get('source', f'Unknown_source_{i}')
            page = doc.metadata.get('page_num', 'Unknown')

            # Clean the content
            content = doc.page_content.strip()

            # Add source annotation directly to content
            annotated_content = f"[SOURCE: {source}, Page {page}]\n{content}\n[END SOURCE]"

            # Create new document with annotated content
            cleaned_doc = Document(
                page_content=annotated_content,
                metadata={
                    **doc.metadata,
                    'source_id': f"{source}_p{page}",
                    'chunk_id': i
                }
            )
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve and clean documents"""
        try:
            # Generate focused queries
            query_variations = self.llm.invoke(self.query_prompt.format(question=query))
            queries = [
                line.strip() for line in query_variations.content.split('\n') if line.strip()
            ]

            # Ensure we have good queries
            if len(queries) < 2:
                queries = [query] + queries
            queries = queries[:3]

            all_docs = []
            seen_content = set()

            # Search with each query
            for search_query in queries:
                docs = self.vectorstore.similarity_search(
                    search_query, 
                    k=self.k,
                    filter={}
                )

                # Deduplicate
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)

            # Clean and annotate
            cleaned_docs = self.clean_and_annotate_chunks(all_docs[:self.k * 2])
            self.last_retrieved_docs = cleaned_docs

            return cleaned_docs[:self.k]

        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Error in document retrieval: {e}")
            self.last_retrieved_docs = []
            return []

@st.cache_resource
def initialize_regular_chat():
    """Initialize a regular chat without RAG for comparison"""
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def format_timing(elapsed_time: float) -> str:
    """Format elapsed time for display"""
    if elapsed_time < 1:
        return f"⏱️ {elapsed_time*1000:.0f}ms"
    if elapsed_time < 60:
        return f"⏱️ {elapsed_time:.1f}s"
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    return f"⏱️ {minutes}m {seconds:.1f}s"

def display_timing(elapsed_time: float):
    """Display timing information in gray text"""
    timing_text = format_timing(elapsed_time)
    st.markdown(f'<p style="color: gray; font-size: 0.8em; margin-top: 5px;">{timing_text}</p>', 
                unsafe_allow_html=True)

def extract_model_names(models_info: list) -> tuple:
    """Extract model names from models information."""
    return tuple(model["model"] for model in models_info["models"])

def create_enhanced_prompt() -> ChatPromptTemplate:
    """Create an enhanced prompt that prevents context bleeding"""

    template = """You are a CRISPR technology expert assistant. Your task is to answer questions using ONLY the provided source documents with extreme precision.

    CRITICAL INSTRUCTIONS:
    1. ONLY state facts that are EXPLICITLY stated in the sources
    2. NEVER combine information from different sources unless they explicitly connect
    3. If a person is mentioned near other information, do NOT assume they are connected unless explicitly stated
    4. When unsure about a connection, say "The sources do not explicitly state this connection"
    5. Provide a clean, readable answer WITHOUT source annotations in the text
    6. You can see source boundaries in the context - use them to avoid mixing information

    CONTEXT FROM SOURCES:
    {context}

    QUESTION: {question}

    ANALYSIS PROCESS:
    1. First, identify which sources contain information relevant to the question
    2. For each relevant fact, verify it is explicitly stated (not inferred)
    3. Check if multiple sources confirm the same information
    4. Note any contradictions or gaps

    ANSWER FORMAT:
    - Provide a clear, natural answer without citation clutter
    - Only state facts that are explicitly mentioned in the sources
    - If information is missing or unclear, explicitly state this
    - Be precise about what the sources do and do not say

    ANSWER:"""

    return ChatPromptTemplate.from_template(template)

def create_fact_verification_chain(llm) -> callable:
    """Create a secondary verification chain to double-check claims"""

    verification_template = """Review this answer for factual accuracy based on the provided sources.

    ORIGINAL ANSWER:
    {answer}

    SOURCES:
    {context}

    VERIFICATION TASK:
    1. Check each factual claim in the answer against the source boundaries
    2. Verify claims are explicitly stated (not inferred from proximity)
    3. Flag any potential errors or unsupported inferences
    4. Pay special attention to claims about achievements, awards, or connections between people

    VERIFICATION RESULT:
    - Confirmed facts: [list facts clearly supported by sources]
    - Questionable claims: [list claims that might be incorrectly inferred]
    - Source mixing issues: [note if information from different sources was incorrectly combined]
    - Overall assessment: [ACCURATE/NEEDS_REVIEW/INCORRECT]

    If issues found, provide CORRECTED VERSION:"""

    verification_prompt = ChatPromptTemplate.from_template(verification_template)

    def verify_answer(answer: str, context: str) -> Dict[str, str]:
        verification = llm.invoke(verification_prompt.format_messages(
            answer=answer, 
            context=context
        ))

        verification_text = verification.content.lower()
        needs_correction = any(phrase in verification_text for phrase in [
            'questionable claims:', 'source mixing issues:', 'needs_review', 'incorrect',
            'corrected version:', 'not explicitly stated'
        ])

        return {
            'verification': verification.content,
            'needs_correction': needs_correction
        }

    return verify_answer

@st.cache_resource
def initialize_enhanced_rag_chain():
    """Initialize enhanced RAG chain with better context handling"""

    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("🔑 PINECONE_API_KEY not found in environment variables!")
        st.stop()

    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)

        # Check index
        available_indexes = pc.list_indexes().names()
        if PINECONE_INDEX_NAME not in available_indexes:
            st.error(f"❌ Pinecone index '{PINECONE_INDEX_NAME}' not found!")
            st.stop()

        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        # Create vectorstore
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=PINECONE_NAMESPACE
        )

        # Initialize LLM with better settings for precision
        llm = ChatOllama(
            model="gemma3:latest",
            temperature=0.05,  # Very low temperature for precision
            num_ctx=6144,      # Larger context
            top_p=0.9,         # Focused sampling
            repeat_penalty=1.1  # Avoid repetition
        )

        # Create enhanced retriever
        retriever = EnhancedDocumentRetriever(vectorstore, llm, k=5)

        # Create enhanced prompt
        prompt = create_enhanced_prompt()

        # Create verification chain
        verifier = create_fact_verification_chain(llm)

        def enhanced_rag_chain(question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
            """Enhanced RAG chain with verification"""

            # Create context-aware query if we have conversation history
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
                context_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
                enhanced_query = f"Context from previous conversation:\n{context_summary}\n\nCurrent question: {question}"
            else:
                enhanced_query = question

            # Get documents
            docs = retriever.get_relevant_documents(enhanced_query)

            # Format context with clear source separation for LLM processing
            # (but this won't appear in the final answer)
            context_parts = []
            for i, doc in enumerate(docs):
                context_parts.append(f"DOCUMENT {i+1}:\n{doc.page_content}\n")

            context = "\n" + "="*50 + "\n".join(context_parts)

            # Generate initial response
            initial_response = llm.invoke(prompt.format_messages(
                context=context, 
                question=enhanced_query
            ))

            # Verify the response
            verification_result = verifier(initial_response.content, context)

            # Return both original and verification
            return {
                'answer': initial_response.content,
                'verification': verification_result,
                'source_documents': retriever.last_retrieved_docs,
                'context_used': docs,
                'needs_review': verification_result.get('needs_correction', False)
            }

        return enhanced_rag_chain, stats, retriever

    except Exception as e: # pylint: disable=broad-except
        st.error(f"❌ Error initializing enhanced RAG: {str(e)}")
        st.stop()

def display_enhanced_response(result: Dict[str, Any], message_id: str):
    """Display response with verification information"""

    # Main answer
    st.markdown(result['answer'])

    # Show verification if there are concerns
    if result.get('needs_review', False):
        with st.expander("⚠️ Fact Verification Check", expanded=True):
            st.warning("Please review carefully.")
            st.markdown(result['verification']['verification'])

    # Show sources
    display_source_documents(result['source_documents'], message_id)

def display_source_documents(docs: List[Document], message_id: str = ""):
    """Display source documents with better formatting"""
    if not docs:
        return

    with st.expander(f"📚 Source Documents ({len(docs)} found)", expanded=False):
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page_num', 'Unknown page')

            st.markdown(f"**Source {i+1}:** {source} (Page {page})")

            # Clean content for display (remove annotations)
            content = doc.page_content
            if content.startswith('[SOURCE:'):
                # Remove source annotations for display
                lines = content.split('\n')
                clean_lines = [line for line in lines if not line.startswith('[SOURCE:') and not line.startswith('[END SOURCE]')]
                content = '\n'.join(clean_lines)

            if len(content) > 800:
                content = content[:800] + "..."

            st.text_area(
                f"Content from Source {i+1}",
                content,
                height=120,
                key=f"doc_content_{message_id}_{i}",
                label_visibility="collapsed"
            )

            st.divider()

# [Keep the existing helper functions: extract_model_names, format_timing, display_timing]

def main():
    """Enhanced main function with better context handling"""
    page_icon("🧬")
    st.subheader("CRISPR Historian Chatbot", divider="red", anchor=False)

    # Sidebar
    st.sidebar.header("🧬 Configuration")
    st.sidebar.info(f"Using Index: **{PINECONE_INDEX_NAME}**")
    
    use_enhanced_rag = st.sidebar.toggle("Use Enhanced CRISPR Knowledge Retrieval", value=True)
    
    if use_enhanced_rag:
        st.sidebar.success("🧬 Using RAG with fact verification")
        
        try:
            chain, stats, _ = initialize_enhanced_rag_chain()
            
            with st.sidebar.expander("📊 Knowledge Base Stats", expanded=True):
                st.metric("Total Documents", stats.get('total_vector_count', 'Unknown'))
                st.info("Context isolation enabled")
                
        except Exception as e:  # pylint: disable=broad-except
            st.sidebar.error(f"Failed to connect: {str(e)}")
            use_enhanced_rag = False

    # Model selection
    models_info = ollama.list()
    available_models = tuple(model["model"] for model in models_info["models"])

    if not available_models:
        st.warning("No Ollama models available!")
        return

    selected_model = st.selectbox("Select model:", available_models)

    # Chat interface
    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant" and "result" in message:
                # Enhanced response display
                display_enhanced_response(message["result"], f"msg_{idx}")
            else:
                st.markdown(message["content"])

    # Handle new input
    if prompt := st.chat_input("Ask a precise question about CRISPR..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        message_container.chat_message("user", avatar="😎").markdown(prompt)

        with message_container.chat_message("assistant", avatar="🤖"):
            start_time = time.time()
            
            if use_enhanced_rag:
                with st.spinner("🔍 Searching with fact checking..."):
                    try:
                        # Get conversation history (excluding current prompt)
                        conversation_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                        result = chain(prompt, conversation_history)
                        elapsed_time = time.time() - start_time
                        
                        display_enhanced_response(result, f"current_{len(st.session_state.messages)}")

                        display_timing(elapsed_time)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['answer'],
                            "result": result,
                            "timing": elapsed_time
                        })
                        
                    except Exception as e:  # pylint: disable=broad-except
                        st.error(f"❌ Error: {str(e)}")
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


if __name__ == "__main__":
    main()
