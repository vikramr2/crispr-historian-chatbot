import os
import json
import re
from typing import Any, Dict, List

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
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

    verification_template = """Review this answer for factual accuracy and temporal consistency based on the provided sources.

    ORIGINAL ANSWER:
    {answer}

    SOURCES:
    {context}

    VERIFICATION TASK:
    1. FACTUAL ACCURACY:
    - Check each factual claim against the source boundaries
    - Verify claims are explicitly stated (not inferred from proximity)
    - Confirm author attributions match the sources exactly
    - Verify year citations are correct

    2. TEMPORAL CONSISTENCY:
    - Extract all years mentioned in the answer
    - Check if temporal language matches chronological order
    - Flag cases where "later" refers to earlier years
    - Verify evolutionary narrative follows proper chronological sequence

    3. SOURCE ATTRIBUTION:
    - Confirm each author citation exists in the sources
    - Check that findings are attributed to the correct papers
    - Verify that author names and years match exactly

    4. LOGICAL FLOW:
    - Ensure the narrative progression makes chronological sense
    - Check that temporal connectors ("Initially", "Later", "Subsequently") are used correctly
    - Verify that the evolutionary story follows a logical timeline

    VERIFICATION RESULT:

    FACTUAL ACCURACY:
    - Confirmed facts: [list facts clearly supported by sources]
    - Questionable claims: [list claims that might be incorrectly inferred]
    - Attribution errors: [note any misattributed authors or years]

    TEMPORAL CONSISTENCY:
    - Timeline issues: [list any chronological inconsistencies]
    - Language problems: [note where temporal language doesn't match chronology]
    - Suggested fixes: [propose corrections for temporal flow]

    SOURCE MIXING ISSUES:
    - [note if information from different sources was incorrectly combined]

    OVERALL ASSESSMENT: [ACCURATE/NEEDS_TEMPORAL_CORRECTION/NEEDS_FACTUAL_CORRECTION/INCORRECT]

    CORRECTED VERSION (if needed):
    [Provide a corrected version that maintains proper chronological flow and accurate facts]

    SPECIFIC CORRECTIONS MADE:
    - [list specific changes made to fix temporal/factual issues]
    """

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

def create_timeline_extraction_chain(llm) -> callable:
    """Create an LLM chain specifically for extracting timeline events"""
    
    extraction_template = """You are an expert at extracting chronological events from scientific text about CRISPR technology.

Your task is to identify key discoveries, developments, or milestones mentioned in the text and format them as timeline events.

INSTRUCTIONS:
1. Extract events that have specific years (1990-2030)
2. Focus on scientific discoveries, publications, developments, or breakthroughs
3. Include the author/researcher name when mentioned
4. Provide a concise but meaningful description of what happened
5. Ignore vague statements or general narrative text
6. Only include events explicitly mentioned in the text

TEXT TO ANALYZE:
{text}

FORMAT YOUR RESPONSE AS A JSON LIST:
[
  {
    "year": 2012,
    "author": "Jinek et al.",
    "description": "demonstrated programmable DNA cleavage using guide RNAs",
    "confidence": "high"
  },
  {
    "year": 2013,
    "author": "Zhang",
    "description": "adapted CRISPR-Cas9 for use in mammalian cells",
    "confidence": "medium"
  }
]

CONFIDENCE LEVELS:
- "high": Author and specific discovery clearly stated
- "medium": Year and discovery clear, author may be implied
- "low": Year mentioned but discovery details are vague

RESPONSE (JSON only, no other text):"""

    extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
    
    def extract_timeline_events(text: str) -> List[Dict[str, Any]]:
        """Extract timeline events from text using LLM"""
        try:
            response = llm.invoke(extraction_prompt.format_messages(text=text))
            events = json.loads(response.content)
            
            # Validate and clean the events
            cleaned_events = []
            for event in events:
                year = event.get('year')
                if year and isinstance(year, int) and 1990 <= year <= 2030:
                    cleaned_events.append({
                        'year': year,
                        'author': event.get('author', 'Unknown'),
                        'description': event.get('description', ''),
                        'type': 'discovery',
                        'confidence': event.get('confidence', 'medium')
                    })
            
            return cleaned_events
            
        except (json.JSONDecodeError, Exception) as e:  # pylint: disable=broad-exception-caught
            # Fallback to empty list if extraction fails
            print(f"Timeline extraction failed: {e}")
            return []
    
    return extract_timeline_events

def create_source_event_extraction_chain(llm) -> callable:
    """Create an LLM chain for extracting events from source documents"""
    
    source_template = """Extract the key contribution or discovery from this research paper.

PAPER METADATA:
Author: {author}
Year: {year}
Title: {title}

PAPER CONTENT (excerpt):
{content}

Your task: Provide a 1-2 sentence description of the main contribution or discovery from this paper.
Focus on what was discovered, developed, or demonstrated.

RESPONSE FORMAT:
[author, year] 'Brief description of the main contribution/discovery'

RESPONSE:"""

    source_prompt = ChatPromptTemplate.from_template(source_template)
    
    def extract_source_event(author: str, year: int, title: str, content: str) -> str:
        """Extract main contribution from a source document"""
        try:
            response = llm.invoke(source_prompt.format_messages(
                author=author,
                year=year,
                title=title,
                content=content
            ))
            return response.content.strip()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Source event extraction failed: {e}")
            return f"{title[:50]}..."
    
    return extract_source_event

def extract_timeline_events_with_llm(answer_text: str, source_docs: List[Document], 
                                   timeline_extractor: callable, source_extractor: callable) -> List[Dict[str, Any]]:
    """Extract timeline events using LLM for better accuracy and meaningful descriptions"""
    
    events = []
    seen_events = set()  # Track (year, author) pairs to avoid duplicates
    
    # Extract events from the main answer text
    if answer_text.strip():
        extracted_events = timeline_extractor(answer_text)
        
        for event in extracted_events:
            year = event.get('year')
            author = event.get('author', 'Unknown')
            
            if year and 1990 <= year <= 2030:
                event_key = (year, author.split()[0] if author != 'Unknown' else 'Unknown')
                
                if event_key not in seen_events:
                    events.append(event)
                    seen_events.add(event_key)
    
    # Extract events from source documents
    for doc in source_docs:
        try:
            doc_year = doc.metadata.get('year')
            if doc_year and str(doc_year).isdigit():
                year = int(doc_year)
                if 1990 <= year <= 2030:
                    
                    author = doc.metadata.get('first_author', doc.metadata.get('author', 'Unknown author'))
                    title = doc.metadata.get('title', 'research')
                    
                    # Clean author name
                    if author and author != 'Unknown author':
                        # Format as "FirstAuthor et al." if it's a full name
                        author_parts = author.strip().split()
                        if len(author_parts) > 1:
                            author = f"{author_parts[0]} et al."
                    
                    event_key = (year, author.split()[0] if author != 'Unknown author' else 'Unknown')
                    
                    if event_key not in seen_events:
                        # Use LLM to extract meaningful description from the document
                        content_excerpt = doc.page_content[:500]  # First 500 chars
                        description = source_extractor(author, year, title, content_excerpt)
                        
                        events.append({
                            'year': year,
                            'author': author,
                            'description': description,
                            'type': 'source',
                            'confidence': 'medium'
                        })
                        seen_events.add(event_key)
                        
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error processing source document: {e}")
            continue
    
    # Sort by year, then by confidence (high first)
    confidence_order = {'high': 0, 'medium': 1, 'low': 2}
    events.sort(key=lambda x: (x['year'], confidence_order.get(x.get('confidence', 'medium'), 1)))
    
    return events

def extract_timeline_events_regex_fallback(answer_text: str, source_docs: List[Document]) -> List[Dict[str, Any]]:
    """Fallback regex-based extraction for when LLM fails"""
    
    events = []
    
    # Regular expression to find year mentions with context
    year_patterns = [
        r'(?:Initially,?\s+in|In|By|During)\s+(\d{4}),?\s+([^.]*\.)',
        r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)\s+([^.]*\.)',
        r'(\d{4})[,:]?\s+([^.]*\.)',
    ]
    
    seen_years = set()
    
    for pattern in year_patterns:
        matches = re.finditer(pattern, answer_text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            
            if len(groups) == 2:
                year_str, description = groups
                if year_str.isdigit():
                    year = int(year_str)
                    if 1990 <= year <= 2030 and year not in seen_years:
                        events.append({
                            'year': year,
                            'author': 'Unknown',
                            'description': description.strip(),
                            'type': 'discovery',
                            'confidence': 'low'
                        })
                        seen_years.add(year)
                        
            elif len(groups) == 3:
                author_or_year, year_or_desc, desc_or_year = groups
                
                if author_or_year.isdigit():
                    year = int(author_or_year)
                    description = f"{year_or_desc} {desc_or_year}".strip()
                    author = 'Unknown'
                elif year_or_desc.isdigit():
                    year = int(year_or_desc)
                    description = desc_or_year.strip()
                    author = author_or_year
                else:
                    continue
                    
                if 1990 <= year <= 2030 and year not in seen_years:
                    events.append({
                        'year': year,
                        'author': author,
                        'description': description.strip(),
                        'type': 'discovery',
                        'confidence': 'medium'
                    })
                    seen_years.add(year)
    
    # Also extract years from source documents
    for doc in source_docs:
        doc_year = doc.metadata.get('year')
        if doc_year and str(doc_year).isdigit():
            year = int(doc_year)
            if 1990 <= year <= 2030 and year not in seen_years:
                first_author = doc.metadata.get('first_author', 'Unknown author')
                title = doc.metadata.get('title', 'research')
                description = f"{title[:50]}..."
                
                events.append({
                    'year': year,
                    'author': first_author,
                    'description': description,
                    'type': 'source',
                    'confidence': 'low'
                })
                seen_years.add(year)
    
    events.sort(key=lambda x: x['year'])
    return events

@st.cache_resource
def initialize_enhanced_rag_chain():
    """Initialize enhanced RAG chain with better context handling and timeline extraction"""

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

        # Initialize timeline extraction LLM (slightly higher temperature for creativity)
        timeline_llm = ChatOllama(
            model="gemma3:latest",
            temperature=0.1,   # Slightly higher for timeline extraction
            num_ctx=4096,
            top_p=0.9
        )

        # Create enhanced retriever
        retriever = DocumentRetriever(vectorstore, llm, k=5)

        # Create verification chain
        verifier = create_fact_verification_chain(llm)

        # Create timeline extraction chains
        timeline_extractor = create_timeline_extraction_chain(timeline_llm)
        source_extractor = create_source_event_extraction_chain(timeline_llm)

        def enhanced_rag_chain(question: str, conversation_history: List[Dict] = None, use_conversation_context: bool = True) -> Dict[str, Any]:
            """Enhanced RAG chain with verification, timeline extraction, and optional conversation context"""
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

            # Extract timeline events if this is an evolutionary query
            timeline_events = []
            if classification == "EVOLUTIONARY":
                try:
                    timeline_events = extract_timeline_events_with_llm(
                        initial_response.content, 
                        retriever.last_retrieved_docs,
                        timeline_extractor,
                        source_extractor
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Timeline extraction failed, using fallback: {e}")
                    timeline_events = extract_timeline_events_regex_fallback(
                        initial_response.content, 
                        retriever.last_retrieved_docs
                    )

            # Return both original and verification
            return {
                'answer': initial_response.content,
                'verification': verification_result,
                'source_documents': retriever.last_retrieved_docs,
                'context_used': docs,
                'needs_review': verification_result.get('needs_correction', False),
                'used_conversation_context': use_conversation_context and conversation_history and len(conversation_history) > 0,
                'temporal_classification': classification_result,
                'retrieval_strategy': classification,
                'timeline_events': timeline_events  # Add timeline events to the result
            }

        return enhanced_rag_chain, stats, retriever

    except Exception as e: # pylint: disable=broad-except
        st.error(f"❌ Error initializing enhanced RAG: {str(e)}")
        st.stop()
