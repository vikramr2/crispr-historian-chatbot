"""
Main application for the CRISPR Historian chatbot.

This application uses an enhanced RAG (Retrieval-Augmented Generation) approach
to provide precise answers about CRISPR technology, its history, and related topics.
"""

import time

import ollama
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from utilities.display_utils import display_enhanced_response, display_timing, display_timeline
from utilities.icon import page_icon

from utilities.constants import PINECONE_INDEX_NAME
from utilities.rag_utils import initialize_enhanced_rag_chain

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="CRISPR Historian",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def initialize_regular_chat():
    """Initialize a regular chat without RAG for comparison"""
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def extract_model_names(models_info: list) -> tuple:
    """Extract model names from models information."""
    return tuple(model["model"] for model in models_info["models"])

def main():
    """Enhanced main function with new conversation toggle"""
    page_icon("🧬")
    st.subheader("CRISPR Historian Chatbot", divider="red", anchor=False)

    # Sidebar
    st.sidebar.header("🧬 Configuration")
    st.sidebar.info(f"Using Index: **{PINECONE_INDEX_NAME}**")
    
    use_enhanced_rag = st.sidebar.toggle("Use Enhanced CRISPR Knowledge Retrieval", value=True)
    
    # NEW CONVERSATION TOGGLE
    st.sidebar.markdown("---")
    new_conversation_mode = st.sidebar.toggle(
        "🆕 New Conversation", 
        value=False,
        help="When enabled, each question is treated as a fresh conversation without using previous chat history for context."
    )
    
    if new_conversation_mode:
        st.sidebar.info("💬 **Fresh context mode**: Each question starts a new conversation")
    else:
        st.sidebar.info("💬 **Continuous mode**: Using conversation history for context")
    
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

    # Add clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

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
                        # Determine conversation history based on toggle
                        if new_conversation_mode:
                            # Don't use conversation history - fresh context
                            conversation_history = []
                            use_conversation_context = False
                        else:
                            # Use conversation history (excluding current prompt)
                            conversation_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                            use_conversation_context = True
                        
                        result = chain(prompt, conversation_history, use_conversation_context)
                        elapsed_time = time.time() - start_time
                        
                        display_enhanced_response(result, f"current_{len(st.session_state.messages)}")

                        if result.get('retrieval_strategy') == 'EVOLUTIONARY':
                            display_timeline(result)

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
                    
                    # Determine messages to send based on toggle
                    if new_conversation_mode:
                        # Only send the current prompt
                        messages_to_send = [{"role": "user", "content": prompt}]
                    else:
                        # Send all conversation history
                        messages_to_send = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=messages_to_send,
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
    