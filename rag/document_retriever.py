from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import re


class DocumentRetriever:
    """Enhanced retriever with better context handling and source attribution"""

    def __init__(self, vectorstore, llm, k: int = 6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.last_retrieved_docs = []

        # Standard query prompt
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 precise search queries for CRISPR research documents.
            Focus on exact terms, names, and concepts from the question.
            If the question mentions specific researchers or authors, include their names.
            Avoid combining unrelated concepts in a single query.
            
            Question: {question}
            
            Search queries (one per line):"""
        )

        # Year extraction prompt for explicit temporal queries
        self.year_extraction_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Extract specific years, date ranges, or decades from this question.
            Return only the numerical years, one per line. If a range is given, list all years in that range.
            If a decade is mentioned (like "1990s"), list the start and end years of that decade.
            
            Question: {question}
            
            Years:"""
        )

        # Evolutionary story prompt
        self.evolutionary_prompt = PromptTemplate(
            input_variables=["question", "sorted_context"],
            template="""You are a science historian. Using the chronologically sorted sources below, 
            tell the story of how scientific understanding evolved over time to answer this question.

            Question: {question}

            Chronologically sorted sources (earliest to latest):
            {sorted_context}

            Structure your answer as a temporal narrative:
            1. Start with the earliest discoveries/observations
            2. Show how each subsequent finding built upon or refined previous understanding  
            3. Highlight key breakthroughs and when they occurred
            4. End with current understanding

            Use phrases like "Initially...", "Later research showed...", "By [year]...", "Subsequently...", "Today..."
            Always cite the year when discussing specific findings.

            Answer:"""
        )

    def create_evolutionary_context(self, docs: List[Document]) -> str:
        """Create chronologically sorted context for evolutionary stories"""
        # Sort documents by year
        def get_year(doc):
            year = doc.metadata.get('year', 9999)
            try:
                return int(year)
            except (ValueError, TypeError):
                return 9999

        sorted_docs = sorted(docs, key=get_year)
        
        # Create context with clear temporal markers
        context_parts = []
        for i, doc in enumerate(sorted_docs):
            year = doc.metadata.get('year', 'Unknown year')
            source = doc.metadata.get('source', 'Unknown source')
            
            context_parts.append(f"DOCUMENT {i+1} ({year}):\nSource: {source}\n{doc.page_content}\n")

        return "\n" + "="*50 + "\n".join(context_parts)

    def extract_years_from_query(self, query: str) -> List[int]:
        """Extract years from explicit temporal queries"""
        try:
            # First try direct regex extraction from the original query
            direct_years = []
            year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
            for year_str in year_matches:
                year = int(year_str)
                if 1950 <= year <= 2030:  # Reasonable range for CRISPR research
                    direct_years.append(year)
            
            # If we found years directly, use those
            if direct_years:
                return sorted(list(set(direct_years)))
            
            # Otherwise, use LLM extraction as fallback
            response = self.llm.invoke(self.year_extraction_prompt.format(question=query))
            year_text = response.content.strip()
            
            years = []
            for line in year_text.split('\n'):
                line = line.strip()
                # Extract 4-digit years - fixed regex to capture full year
                year_matches = re.findall(r'\b((?:19|20)\d{2})\b', line)
                for year_str in year_matches:
                    year = int(year_str)
                    if 1950 <= year <= 2030:  # Reasonable range for CRISPR research
                        years.append(year)
            
            return sorted(list(set(years)))  # Remove duplicates and sort

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error extracting years: {e}")
            return []
        
    def retrieve_with_year_filter(self, query: str, years: List[int]) -> List[Document]:
        """Retrieve documents filtered by specific years"""
        try:
            # Generate search queries
            query_variations = self.llm.invoke(self.query_prompt.format(question=query))
            queries = [
                line.strip() for line in query_variations.content.split('\n') if line.strip()
            ]
            
            if len(queries) < 2:
                queries = [query] + queries
            queries = queries[:3]

            all_docs = []
            seen_content = set()

            # Search with year filtering
            for search_query in queries:
                for year in years:
                    # Create year filter for Pinecone
                    year_filter = {"year": {"$eq": str(year)}}
                    
                    docs = self.vectorstore.similarity_search(
                        search_query,
                        k=self.k,
                        filter=year_filter
                    )
                    
                    # Deduplicate
                    for doc in docs:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_docs.append(doc)

            return all_docs[:self.k * 2]  # Return more docs for temporal queries
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in year-filtered retrieval: {e}")
            return []
        
    def retrieve_for_evolutionary_story(self, query: str) -> List[Document]:
        """Retrieve documents for evolutionary timeline, ensuring temporal spread"""
        try:
            # Generate search queries
            query_variations = self.llm.invoke(self.query_prompt.format(question=query))
            queries = [
                line.strip() for line in query_variations.content.split('\n') if line.strip()
            ]
            
            if len(queries) < 2:
                queries = [query] + queries
            queries = queries[:3]

            all_docs = []
            seen_content = set()

            # Standard similarity search (no year filtering)
            for search_query in queries:
                docs = self.vectorstore.similarity_search(
                    search_query,
                    k=self.k * 2,  # Get more docs for better temporal coverage
                    filter={}
                )
                
                # Deduplicate
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)

            # Sort by year for evolutionary narrative
            def get_year(doc):
                year = doc.metadata.get('year', 9999)  # Default to future for missing years
                try:
                    return int(year)
                except (ValueError, TypeError):
                    return 9999

            sorted_docs = sorted(all_docs, key=get_year)
            
            return sorted_docs[:self.k * 2]
            
        except Exception as e:    # pylint: disable=broad-except
            print(f"Error in evolutionary retrieval: {e}")
            return []
        
    def get_relevant_documents_with_classification(self, query: str, classification: str) -> List[Document]:
        """Main retrieval method that handles different classification types"""
        
        if classification == "EXPLICIT_TEMPORAL":
            # Extract years and filter
            years = self.extract_years_from_query(query)
            if years:
                docs = self.retrieve_with_year_filter(query, years)
                print(f"Temporal retrieval for years {years}: found {len(docs)} documents")
            else:
                # Fallback to standard retrieval if no years found
                docs = self.get_relevant_documents(query)
                print("No years extracted, falling back to standard retrieval")
                
        elif classification == "EVOLUTIONARY":
            # Get documents for evolutionary story
            docs = self.retrieve_for_evolutionary_story(query)
            print(f"Evolutionary retrieval: found {len(docs)} documents spanning multiple years")
            
        else:  # STANDARD
            # Use existing standard retrieval
            docs = self.get_relevant_documents(query)
            print(f"Standard retrieval: found {len(docs)} documents")

        # Clean and annotate
        cleaned_docs = self.clean_and_annotate_chunks(docs)
        self.last_retrieved_docs = cleaned_docs
        
        return cleaned_docs

    def clean_and_annotate_chunks(self, docs: List[Document]) -> List[Document]:
        """Clean chunks and add clear source attribution"""
        cleaned_docs = []
        for i, doc in enumerate(docs):
            # Enhanced metadata extraction
            source = doc.metadata.get('source', f'Unknown_source_{i}')
            title = doc.metadata.get('title', 'Unknown title')
            first_author = doc.metadata.get('first_author', 'Unknown author')
            year = doc.metadata.get('year', 'Unknown year')
            page = doc.metadata.get('page_num', 'Unknown')

            # Clean the content
            content = doc.page_content.strip()

            # Enhanced source annotation with author and title
            annotated_content = f"[SOURCE: {first_author} ({year}) - {title}, Page {page}, {source}]\n{content}\n[END SOURCE]"

            # Create new document with enhanced metadata
            cleaned_doc = Document(
                page_content=annotated_content,
                metadata={
                    **doc.metadata,
                    'source_id': f"{first_author}_{year}_p{page}",
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

def create_temporal_enhanced_prompt() -> ChatPromptTemplate:
    """Create enhanced prompt that handles different temporal contexts"""

    template = """You are a CRISPR technology expert assistant. Your task is to answer questions using ONLY the provided source documents with extreme precision.

    CRITICAL INSTRUCTIONS:
    1. ONLY state facts that are EXPLICITLY stated in the sources
    2. NEVER combine information from different sources unless they explicitly connect
    3. If a person is mentioned near other information, do NOT assume they are connected unless explicitly stated
    4. When unsure about a connection, say "The sources do not explicitly state this connection"
    5. Provide a clean, readable answer WITHOUT source annotations in the text
    6. You can see source boundaries in the context - use them to avoid mixing information
    7. Pay attention to years mentioned in sources - maintain temporal accuracy

    CONTEXT FROM SOURCES:
    {context}

    QUESTION: {question}

    ANALYSIS PROCESS:
    1. First, identify which sources contain information relevant to the question
    2. For each relevant fact, verify it is explicitly stated (not inferred)
    3. Check if multiple sources confirm the same information
    4. Note any contradictions or gaps
    5. Pay attention to temporal sequence if this is an evolutionary question

    ANSWER FORMAT:
    - Provide a clear, natural answer without citation clutter
    - Only state facts that are explicitly mentioned in the sources
    - If information is missing or unclear, explicitly state this
    - Be precise about what the sources do and do not say
    - For temporal questions, maintain chronological accuracy

    ANSWER:"""

    return ChatPromptTemplate.from_template(template)


def create_evolutionary_answer_prompt() -> ChatPromptTemplate:
    """Create prompt specifically for evolutionary timeline answers"""

    template = """You are a science historian expert on CRISPR technology. Create a chronological narrative showing how scientific understanding evolved over time.

    CRITICAL INSTRUCTIONS:
    1. ONLY use facts EXPLICITLY stated in the chronologically sorted sources
    2. Structure as a temporal story from earliest to latest findings
    3. Use temporal connectors: "Initially...", "Later research showed...", "By [year]...", "Subsequently...", "Today..."
    4. Always mention the year when discussing specific findings
    5. Show how later discoveries built upon or refined earlier understanding
    6. Be precise about what each source contributes to the timeline

    CHRONOLOGICALLY SORTED CONTEXT:
    {context}

    QUESTION: {question}

    Create a temporal narrative that answers the question by showing the evolution of scientific understanding:

    ANSWER:"""

    return ChatPromptTemplate.from_template(template)
