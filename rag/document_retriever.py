from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import streamlit as st


class DocumentRetriever:
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
