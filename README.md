# 🧬 CRISPR Historian ChatBot

A specialized chatbot designed to provide in-depth answers on the history of CRISPR as a field, powered by local AI models and retrieval-augmented generation.

---

## ✨ Features

- **🖥️ Interactive UI**: User-friendly interface built with Streamlit
- **🏠 Local Model Execution**: Run open-source Ollama models locally without API keys
- **🔗 Prompt Chaining**: Uses LangChain to decompose questions into specialized subquestions for improved answer quality
- **✅ Fact Verification**: Built-in module to fact-check and improve responses
- **📚 Retrieval Augmented Generation (RAG)**: Access to a database of 976 research articles on CRISPR

---

## 🛠️ Prerequisites

### Installing Ollama

#### With Root Privileges
Download Ollama directly from [ollama.com](https://ollama.com/) and follow the installation instructions.

#### Without Root Privileges
Install Ollama in binary format:

```bash
# Create a local bin directory
mkdir -p $HOME/.local/bin

# Download the binary
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 -o $HOME/.local/bin/ollama

# Make it executable
chmod +x $HOME/.local/bin/ollama

# Add to PATH (add this to your ~/.bashrc or ~/.zshrc)
export PATH=$HOME/.local/bin:$PATH

# Start a server
ollama serve
```

### 🤖 Recommended Models

- **Generator**: `Gemma3` - Excellent performance for CRISPR-related queries
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` - Automatically installed from HuggingFace

**Installing Gemma3:**
```bash
# Start Ollama server (if not running)
ollama serve

# In another terminal, pull the model
ollama pull gemma3:latest
```

---

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/vikramr2/crispr_historian_chatbot.git
    cd crispr_historian_chatbot
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

Start the application:
```bash
./run_chat.sh
```

Navigate to the URL provided by Streamlit in your browser to interact with the chatbot.

---

## 🎯 Key Features

### ✅ Fact Checking
The chatbot provides an initial answer, then performs fact-checking to refine and improve the response.

<div align="center">
  <img src="assets/fact_check.jpeg" alt="Fact Checking Process" width="600">
  <br><em>Initial fact-checking process</em>
</div>

<div align="center">
  <img src="assets/fact_check2.jpg" alt="Refined Response" width="600">
  <br><em>Fact-checked and refined response</em>
</div>

### 📖 Source Listing
Every response includes source citations, allowing historians to examine text snippets for further research.

<div align="center">
  <img src="assets/finds_sources.jpeg" alt="Source Citations" width="600">
  <br><em>Comprehensive source listing</em>
</div>

### ⚙️ Dialable Settings
**(Testing purposes only)** Toggle between document-informed RAG and the LLM's background knowledge.

<div align="center">
  <img src="assets/dialable_settings.jpeg" alt="Settings Panel" width="200">
  <br><em>Configurable response settings</em>
</div>

---

## 🙏 References and Acknowledgments

**Inspiration & Resources:**
- [Original Ollama Streamlit Template](https://github.com/tonykipkemboi/ollama_streamlit_demos) - Foundation for RAG implementation
- [RAG Tutorial](https://www.youtube.com/watch?v=bAI_jWsLhFM) - Helpful development guidance

**Special Thanks:**
👏 Kudos to the [Ollama](https://ollama.com/) team for making open-source models more accessible!

---

<div align="center">
  <em>Built for historians, researchers, and CRISPR enthusiasts</em>
</div>
