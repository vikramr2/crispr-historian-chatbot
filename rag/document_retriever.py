from typing import List, Tuple
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
        self.last_similarity_scores = []

        # Standard query prompt
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 precise search queries for CRISPR research documents.
            Focus on exact terms, names, and concepts from the question.
            If the question mentions specific researchers or authors, include their names.
            Avoid combining unrelated concepts in a single query.
            
            IMPORTANT: For research before 2002, CRISPR was known by other terms:
            - SRSR (Short Regularly Spaced Repeats) - used before 2002
            - Repetitive sequences
            - Clustered repeats
            - Palindromic repeats
            
            If the question involves early CRISPR research, time periods before 2002, or early discoverers like Mojica, Ishino, or Nakata, include these historical terms in your queries.
            
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
        """Extract years from explicit temporal queries, including ranges"""
        try:
            # First try direct regex extraction with range support
            direct_years = self._extract_years_with_ranges(query)
            
            # If we found years directly, use those
            if direct_years:
                return sorted(list(set(direct_years)))
            
            # Otherwise, use LLM extraction as fallback
            response = self.llm.invoke(self.year_extraction_prompt.format(question=query))
            year_text = response.content.strip()
            
            years = []
            for line in year_text.split('\n'):
                line = line.strip()
                # Try range extraction on LLM output
                llm_years = self._extract_years_with_ranges(line)
                years.extend(llm_years)
            
            return sorted(list(set(years))) if years else []

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error extracting years: {e}")
            return []

    def _extract_years_with_ranges(self, text: str) -> List[int]:
        """Extract years handling various temporal expressions"""
        years = []
        text_lower = text.lower()
        
        # Extract all 4-digit years from text
        all_years = [int(match) for match in re.findall(r'\b((?:19|20)\d{2})\b', text) 
                    if 1950 <= int(match) <= 2030]
        
        if not all_years:
            return []
        
        # Pattern 1: "between YEAR1 and YEAR2" or "from YEAR1 to YEAR2"
        between_patterns = [
            r'between\s+(\d{4})\s+and\s+(\d{4})',
            r'from\s+(\d{4})\s+to\s+(\d{4})',
            r'(\d{4})\s*-\s*(\d{4})',  # 1995-2000
            r'(\d{4})\s+through\s+(\d{4})',
            r'(\d{4})\s+until\s+(\d{4})'
        ]
        
        for pattern in between_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                start_year, end_year = int(match[0]), int(match[1])
                if 1950 <= start_year <= 2030 and 1950 <= end_year <= 2030:
                    # Include all years in range
                    years.extend(range(start_year, end_year + 1))
                    return years  # Found range, return it
        
        # Pattern 2: "before YEAR" - include years from reasonable start to that year
        before_patterns = [
            r'before\s+(\d{4})',
            r'prior\s+to\s+(\d{4})',
            r'up\s+to\s+(\d{4})',
            r'until\s+(\d{4})'
        ]
        
        for pattern in before_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                end_year = int(match)
                if 1950 <= end_year <= 2030:
                    # Include years from 1950 (or reasonable CRISPR start) to end_year
                    start_year = max(1950, 1987)  # CRISPR research started ~1987
                    years.extend(range(start_year, end_year + 1))
                    return years
        
        # Pattern 3: "after YEAR" or "since YEAR" - include years from that year to reasonable end
        after_patterns = [
            r'after\s+(\d{4})',
            r'since\s+(\d{4})',
            r'from\s+(\d{4})\s+onwards?',
            r'post[- ](\d{4})',
            r'following\s+(\d{4})'
        ]
        
        for pattern in after_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                start_year = int(match)
                if 1950 <= start_year <= 2030:
                    # Include years from start_year to 2030 (or current year + few years)
                    end_year = min(2030, 2025)  # Current reasonable end
                    years.extend(range(start_year, end_year + 1))
                    return years
        
        # Pattern 4: Decade expressions "1990s", "early 2000s", "late 1990s"
        decade_patterns = [
            r'early\s+(\d{4})s',    # early 2000s -> 2000-2003
            r'late\s+(\d{4})s',     # late 1990s -> 1997-1999  
            r'mid[- ](\d{4})s',     # mid-1990s -> 1994-1996
            r'(\d{4})s',            # 1990s -> 1990-1999
        ]
        
        for i, pattern in enumerate(decade_patterns):
            matches = re.findall(pattern, text_lower)
            for match in matches:
                decade_start = int(match)
                if 1950 <= decade_start <= 2020:
                    if i == 0:  # early
                        years.extend(range(decade_start, decade_start + 4))
                    elif i == 1:  # late  
                        years.extend(range(decade_start + 7, decade_start + 10))
                    elif i == 2:  # mid
                        years.extend(range(decade_start + 4, decade_start + 7))
                    else:  # full decade
                        years.extend(range(decade_start, decade_start + 10))
                    return years
        
        # Pattern 5: "around YEAR", "circa YEAR", "~YEAR" - include year ± 2
        around_patterns = [
            r'around\s+(\d{4})',
            r'circa\s+(\d{4})',
            r'~\s*(\d{4})',
            r'approximately\s+(\d{4})',
            r'about\s+(\d{4})'
        ]
        
        for pattern in around_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                center_year = int(match)
                if 1950 <= center_year <= 2030:
                    # Include center year ± 2 years
                    years.extend(range(max(1950, center_year - 2), 
                                    min(2030, center_year + 3)))
                    return years
        
        # Pattern 6: No special temporal indicators - return exact years mentioned
        if all_years:
            return all_years
        
        return years
        
    def retrieve_with_year_filter(self, query: str, years: List[int]) -> Tuple[List[Document], List[float]]:
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
            all_scores = []
            seen_content = set()

            # Search with year filtering
            for search_query in queries:
                for year in years:
                    # Create year filter for Pinecone
                    year_filter = {"year": {"$eq": str(year)}}
                    
                    # FIX: Use the correct method name for your vector store
                    docs_with_scores = self.vectorstore.similarity_search_with_score(
                        search_query,
                        k=self.k,
                        filter=year_filter
                    )
                    
                    # Deduplicate
                    for doc, score in docs_with_scores:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_docs.append(doc)
                            all_scores.append(score)

            return all_docs[:self.k * 2], all_scores[:self.k * 2]  # Return more docs for temporal queries

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in year-filtered retrieval: {e}")
            return [], []
        
    def retrieve_for_evolutionary_story(self, query: str) -> Tuple[List[Document], List[float]]:
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
            all_scores = []
            seen_content = set()

            # Standard similarity search (no year filtering)
            for search_query in queries:
                # FIX: Use the correct method name for your vector store
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    search_query,
                    k=self.k * 2,  # Get more docs for better temporal coverage
                    filter={}
                )
                
                # Deduplicate
                for doc, score in docs_with_scores:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
                        all_scores.append(score)

            # Sort by year for evolutionary narrative
            def get_year(doc):
                year = doc.metadata.get('year', 9999)  # Default to future for missing years
                try:
                    return int(year)
                except (ValueError, TypeError):
                    return 9999

            # Zip docs and scores together for sorting
            docs_scores_pairs = list(zip(all_docs, all_scores))
            sorted_pairs = sorted(docs_scores_pairs, key=lambda x: get_year(x[0]))
            
            # Unzip back to separate lists
            if sorted_pairs:
                sorted_docs, sorted_scores = zip(*sorted_pairs)
                max_results = self.k * 2
                return list(sorted_docs[:max_results]), list(sorted_scores[:max_results])
            else:
                return [], []
            
        except Exception as e:    # pylint: disable=broad-except
            print(f"Error in evolutionary retrieval: {e}")
            return [], []
        
    def get_relevant_documents_with_classification(self, query: str, classification: str) -> List[Document]:
        """Main retrieval method that handles different classification types"""
        
        if classification == "EXPLICIT_TEMPORAL":
            # Extract years and filter
            years = self.extract_years_from_query(query)
            if years:
                docs, scores = self.retrieve_with_year_filter(query, years)
                print(f"Temporal retrieval for years {years}: found {len(docs)} documents")
            else:
                # Fallback to standard retrieval if no years found
                docs, scores = self.get_relevant_documents_with_scores(query)
                print("No years extracted, falling back to standard retrieval")
                
        elif classification == "EVOLUTIONARY":
            # Get documents for evolutionary story
            docs, scores = self.retrieve_for_evolutionary_story(query)
            print(f"Evolutionary retrieval: found {len(docs)} documents spanning multiple years")
            
        else:  # STANDARD
            # Use existing standard retrieval
            docs, scores = self.get_relevant_documents_with_scores(query)
            print(f"Standard retrieval: found {len(docs)} documents")

        # Clean and annotate
        cleaned_docs = self.clean_and_annotate_chunks(docs)
        self.last_retrieved_docs = cleaned_docs
        self.last_similarity_scores = scores
        
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

    def get_relevant_documents_with_scores(self, query: str) -> Tuple[List[Document], List[float]]:
        """Retrieve documents with similarity scores - NEW METHOD"""
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
            all_scores = []
            seen_content = set()

            # Search with each query
            for search_query in queries:
                # FIX: Use the correct method name for your vector store
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    search_query, 
                    k=self.k,
                    filter={}
                )

                # Deduplicate
                for doc, score in docs_with_scores:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
                        all_scores.append(score)

            return all_docs[:self.k], all_scores[:self.k]

        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Error in document retrieval: {e}")
            return [], []

    def get_relevant_documents(self, query: str) -> Tuple[List[Document], List[float]]:
        """Updated to return scores for backward compatibility"""
        docs, scores = self.get_relevant_documents_with_scores(query)
        
        # Clean and annotate
        cleaned_docs = self.clean_and_annotate_chunks(docs)
        self.last_retrieved_docs = cleaned_docs
        self.last_similarity_scores = scores
        
        return cleaned_docs, scores

    def get_last_similarity_scores(self) -> List[float]:
        """Get similarity scores from the last retrieval"""
        return self.last_similarity_scores


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
