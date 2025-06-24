import streamlit as st
from typing import List, Dict, Any
from langchain_core.documents import Document


# Path to PDF files (for source attribution)
PDF_PATH = "/projects/illinois/eng/cs/chackoge/illinoiscomputes/vikramr2/llm-crispr/data/pdfs"

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
    
def display_enhanced_response(result: Dict[str, Any], message_id: str):
    """Display response with verification information"""

    # Display question classification if available
    if "temporal_classification" in result:
        display_classification_result(result["temporal_classification"])

    # Show context usage indicator
    if result.get('used_conversation_context', False):
        st.markdown('<p style="color: #2E8B57; font-size: 0.85em; margin-bottom: 10px;">💬 Using conversation context</p>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #4682B4; font-size: 0.85em; margin-bottom: 10px;">🆕 Fresh conversation context</p>', 
                   unsafe_allow_html=True)

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

            # Get only the part of the source after the PDF_PATH
            if source.startswith(PDF_PATH):
                source = source[len(PDF_PATH):].lstrip('/')

            # Get paper metadata
            author = doc.metadata.get('author', '')
            year = doc.metadata.get('year', '')
            title = doc.metadata.get('title', '')

            # Format the author part to say et. al. is there are multiple authors
            author_list = author.split(', ')
            if len(author_list) > 1:
                author = f"{author_list[0]} et al."
            else:
                first_author = doc.metadata.get('first_author', '')
                if first_author:
                    author = first_author

            if len(author) or len(year) or len(title):
                cite_string = f"{author}, {year}. {title}. Page {page}."
            else:
                cite_string = f"Source {i+1} - {source} (Page {page})"

            st.markdown(f"**Source {i+1}:** {cite_string}")

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

def display_classification_result(result: Dict[str, Any]):
    """Display classification result in Streamlit with appropriate styling"""
    
    classification = result["classification"]
    reasoning = result["reasoning"]
    confidence = result["confidence"]
    
    # Color coding for different classifications
    color_map = {
        "EXPLICIT_TEMPORAL": "#FF6B6B",  # Red
        "EVOLUTIONARY": "#4ECDC4",       # Teal  
        "STANDARD": "#45B7D1"            # Blue
    }
    
    icon_map = {
        "EXPLICIT_TEMPORAL": "📅",
        "EVOLUTIONARY": "🧬",
        "STANDARD": "❓"
    }
    
    # Display classification badge
    color = color_map.get(classification, "#45B7D1")
    icon = icon_map.get(classification, "❓")
    
    st.markdown(f"""
    <div style="
        background-color: {color}20; 
        border-left: 4px solid {color}; 
        padding: 10px; 
        margin: 10px 0; 
        border-radius: 5px;
    ">
        <strong>{icon} Query Classification: {classification}</strong><br>
        <em style="color: #666; font-size: 0.9em;">{reasoning}</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Show confidence if low
    if confidence == "low":
        st.warning("⚠️ Low confidence classification - please review")
