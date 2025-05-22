import ollama                                                       # type: ignore
import streamlit as st                                              # type: ignore    
import time
from utilities.icon import page_icon
from langchain.prompts import ChatPromptTemplate, PromptTemplate    # type: ignore
from langchain.retrievers.multi_query import MultiQueryRetriever    # type: ignore
from langchain_community.chat_models import ChatOllama              # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings    # type: ignore
from langchain_community.vectorstores import Chroma                 # type: ignore
from langchain_core.output_parsers import StrOutputParser           # type: ignore
from langchain_core.runnables import RunnablePassthrough            # type: ignore

st.set_page_config(
    page_title="CRISPR Historian",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize the complete RAG chain
@st.cache_resource
def initialize_rag_chain():
    """Initialize the complete RAG chain exactly as in your original script"""
    persist_directory = "rag/vectorstore"
    
    # Much smaller and faster model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Only ~80MB vs larger medical models
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    local_model = "gemma3:latest"
    llm = ChatOllama(model=local_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant specializing in CRISPR technology and its history. Your task is to generate five
        different versions of the given user question to retrieve relevant documents about CRISPR, its discovery, development,
        key scientists, applications, and ethical considerations. By generating multiple perspectives on the user question, your
        goal is to help retrieve comprehensive information about CRISPR history and advancements. Provide these alternative questions
        separated by newlines.
        Original question: {question}""",
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vectorstore.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    
    # The exact template from your original script
    template = """Answer the question. based ONLY on the following context:
{context}
Question: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # The exact chain from your original script
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

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

def main():
    """The main function that runs the application."""
    page_icon("🧬")
    st.subheader("CRISPR Historian Chatbot", divider="red", anchor=False)

    # Toggle for RAG vs regular chat
    use_rag = st.sidebar.toggle("Use CRISPR Knowledge Retrieval", value=True)
    
    if use_rag:
        st.sidebar.info("🧬 Using RAG with CRISPR knowledge base")
    else:
        st.sidebar.info("💬 Using regular chat mode")

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if not available_models:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/settings.py")
        return

    selected_model = st.selectbox(
        "Pick a model available locally on your system ↓", available_models
    )

    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            # Display timing for assistant messages if available
            if message["role"] == "assistant" and "timing" in message:
                display_timing(message["timing"])

    if prompt := st.chat_input("Ask a question about CRISPR..."):
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                start_time = time.time()
                
                if use_rag:
                    # Use your original RAG chain
                    with st.spinner("Retrieving CRISPR knowledge..."):
                        chain = initialize_rag_chain()
                        response = chain.invoke(prompt)  # Direct invoke, not streaming
                        elapsed_time = time.time() - start_time
                        
                        st.markdown(response)
                        display_timing(elapsed_time)
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

        except Exception as e:
            st.error(f"Error: {e}", icon="⛔️")
            st.error("Make sure your vectorstore directory exists and contains CRISPR data.")


if __name__ == "__main__":
    main()
    