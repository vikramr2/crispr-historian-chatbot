import os
from typing import Any, Dict, List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from rag.document_retriever import DocumentRetriever, create_evolutionary_answer_prompt, create_temporal_enhanced_prompt
from rag.temporal_query_classifier import TemporalQueryClassifier

from .constants import PINECONE_INDEX_NAME, PINECONE_NAMESPACE

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
        retriever = DocumentRetriever(vectorstore, llm, k=5)

        # Create verification chain
        verifier = create_fact_verification_chain(llm)

        def enhanced_rag_chain(question: str, conversation_history: List[Dict] = None, use_conversation_context: bool = True) -> Dict[str, Any]:
            """Enhanced RAG chain with verification and optional conversation context"""
            # Add classifier initialization
            classifier_llm = ChatOllama(
                model="gemma3:latest",
                temperature=0.1,
                num_ctx=2048
            )
            classifier = TemporalQueryClassifier(classifier_llm)
            
            # Classify the query
            classification_result = classifier.classify_query(question)
            classification = classification_result["classification"]

            # Create context-aware query only if use_conversation_context is True
            if use_conversation_context and conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
                context_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
                enhanced_query = f"Context from previous conversation:\n{context_summary}\n\nCurrent question: {question}"
            else:
                enhanced_query = question

            # Get documents
            docs = retriever.get_relevant_documents_with_classification(enhanced_query, classification)

            # Choose prompt and context formatting based on classification
            if classification == "STANDARD":
                # Create enhanced prompt
                prompt = create_enhanced_prompt()

                # Format context with clear source separation for LLM processing
                # (but this won't appear in the final answer)
                context_parts = []
                for i, doc in enumerate(docs):
                    context_parts.append(f"DOCUMENT {i+1}:\n{doc.page_content}\n")

                context = "\n" + "="*50 + "\n".join(context_parts)
            elif classification == "EVOLUTIONARY":
                # Use evolutionary answer prompt for this classification
                prompt = create_evolutionary_answer_prompt()
                context = retriever.create_evolutionary_context(docs)
            else:
                # Use temporal enhanced prompt for this classification
                prompt = create_temporal_enhanced_prompt()

                # Format context with clear source separation
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
                'needs_review': verification_result.get('needs_correction', False),
                'used_conversation_context': use_conversation_context and conversation_history and len(conversation_history) > 0,
                'temporal_classification': classification_result,
                'retrieval_strategy': classification
            }

        return enhanced_rag_chain, stats, retriever

    except Exception as e: # pylint: disable=broad-except
        st.error(f"❌ Error initializing enhanced RAG: {str(e)}")
        st.stop()
