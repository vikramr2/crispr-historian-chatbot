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

def display_timeline(result: Dict[str, Any]):
    """Display the evolutionary answer in a hoverable timeline format"""
    
    # Extract timeline events
    events = result.get('timeline_events', [])
    
    if not events:
        return
    
    # Format the event descriptions
    for event in events:
        if not event['description'][0] == '[':
            event['description'] = f'[{event["author"]}, {event["year"]}] {event["description"]}'
    
    with st.expander("🕰️ Timeline of Events", expanded=True):
        st.markdown("**Timeline of Key Discoveries**")
        st.markdown("*Hover over the timeline points to see details*")
        
        # Create the timeline visualization
        min_year = min(event['year'] for event in events)
        max_year = max(event['year'] for event in events)
        year_range = max_year - min_year if max_year > min_year else 1
        
        # Build the complete HTML as one string
        css_styles = """
        <style>
        .timeline-container {
            position: relative;
            margin: 20px 0;
            padding: 40px 20px 20px 20px;
            height: 120px;
            overflow: visible;
        }
        
        /* Allow the component to overflow */
        body, html {
            overflow: visible !important;
        }
        
        .timeline-line {
            position: absolute;
            top: 100px;
            left: 20px;
            right: 20px;
            height: 3px;
            background: linear-gradient(90deg, #4ECDC4, #45B7D1);
            border-radius: 2px;
        }
        
        .timeline-point {
            position: absolute;
            top: 100px;
            transform: translate(-50%, -50%);
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #4ECDC4;
            border: 3px solid white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            z-index: 10;
        }
        
        .timeline-point:hover {
            transform: translate(-50%, -50%) scale(1.3);
            background: #FF6B6B;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .timeline-year {
            position: absolute;
            top: -25px;
            transform: translateX(-50%);
            font-size: 12px;
            font-weight: bold;
            color: #4ECDC4;
            white-space: nowrap;
        }
        
        .timeline-tooltip {
            position: fixed;
            bottom: auto;
            top: -80px;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 280px;
            min-width: 180px;
            white-space: normal;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 999999;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            text-align: left;
            pointer-events: none;
        }
        
        /* Left-aligned tooltip for points on the left side */
        .timeline-tooltip.tooltip-left {
            left: -10px;
            transform: none;
        }
        
        .timeline-tooltip.tooltip-left::after {
            content: '';
            position: absolute;
            bottom: -6px;
            left: 20px;
            border: 6px solid transparent;
            border-top-color: rgba(0,0,0,0.9);
        }
        
        /* Right-aligned tooltip for points on the right side */
        .timeline-tooltip.tooltip-right {
            right: -10px;
            transform: none;
        }
        
        .timeline-tooltip.tooltip-right::after {
            content: '';
            position: absolute;
            bottom: -6px;
            right: 20px;
            border: 6px solid transparent;
            border-top-color: rgba(0,0,0,0.9);
        }
        
        /* Center-aligned tooltip for points in the middle */
        .timeline-tooltip.tooltip-center {
            left: 50%;
            transform: translateX(-50%);
        }
        
        .timeline-tooltip.tooltip-center::after {
            content: '';
            position: absolute;
            bottom: -6px;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: rgba(0,0,0,0.9);
        }
        
        .timeline-point:hover .timeline-tooltip {
            opacity: 1;
            visibility: visible;
        }
        
        .discovery-type {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            margin-right: 5px;
        }
        
        .type-discovery {
            background: #4ECDC4;
            color: white;
        }
        
        .type-source {
            background: #45B7D1;
            color: white;
        }
        </style>
        """
        
        # Build timeline container with padding to prevent overflow
        timeline_container = '<div class="timeline-container"><div class="timeline-line"></div>'
        
        # Add timeline points with constrained positioning
        for event in events:
            # Calculate position (10-90% to leave room for tooltips)
            position = 10 + ((event['year'] - min_year) / year_range) * 80
            
            # Determine tooltip alignment based on position
            if position <= 25:
                tooltip_class = "tooltip-left"
            elif position >= 75:
                tooltip_class = "tooltip-right"
            else:
                tooltip_class = "tooltip-center"
            
            # Clean and truncate description
            description = event['description'].strip()
            if len(description) > 100:
                description = description[:97] + "..."
            
            # Escape HTML characters
            description = (description.replace('&', '&amp;')
                                   .replace('<', '&lt;')
                                   .replace('>', '&gt;')
                                   .replace('"', '&quot;')
                                   .replace("'", '&#39;'))
            
            timeline_container += f'''
            <div class="timeline-point" style="left: {position}%;">
                <div class="timeline-year">{event['year']}</div>
                <div class="timeline-tooltip {tooltip_class}">
                    <div class="discovery-type type-{event['type']}">{event['type'].upper()}</div>
                    <strong>{event['year']}</strong><br>
                    {description}
                </div>
            </div>
            '''
        
        timeline_container += '</div>'
        
        # Combine CSS and HTML
        complete_html = css_styles + timeline_container
        
        # Display the timeline with increased height for tooltips
        st.components.v1.html(complete_html, height=180)
        
        # Add a legend using markdown
        st.markdown("""
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #4ECDC4; border-radius: 50%; margin-right: 5px;"></span>
            Discovery mentioned in answer
            &nbsp;&nbsp;
            <span style="display: inline-block; width: 12px; height: 12px; background: #45B7D1; border-radius: 50%; margin-right: 5px;"></span>
            Source document
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed list below timeline
        st.markdown("**Detailed Timeline:**")
        for event in events:
            icon = "🔬" if event['type'] == 'discovery' else "📄"
            st.markdown(f"- **{event['year']}** {icon} {event['description']}")
