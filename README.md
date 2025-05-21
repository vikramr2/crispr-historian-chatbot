# CRISPR Historian ChatBot

This is a specialized chatbot meant to give more in depth answers on the history of CRISPR as a field.

## Features

From the original repository
- **Interactive UI**: Utilize Streamlit to create a user-friendly interface.
- **Local Model Execution**: Run your Ollama models locally without the need for external APIs.
- **Real-time Responses**: Get real-time responses from your models directly in the UI.

Updates
- **Prompt Chaining**: Uses langchain to decompose questions into specialized subquestions to improve answer quality
- **Retrieval Augmented Generation (RAG)**: This chatbot retrieves from a database of 976 pdfs of research articles on CRISPR

## Installation

Before running the app, ensure you have Python installed on your machine. Then, clone this repository and install the required packages using pip:

```bash
git clone https://github.com/vikramr2/crispr_historian_chatbot.git
```

```bash
cd crispr_historian_chatbot
```

```bash
pip install -r requirements.txt
```

## Usage

To start the app, run the following command in your terminal:

```bash
streamlit run chat.py --server.address=0.0.0.0 --server.port=8501
```

Navigate to the URL provided by Streamlit in your browser to interact with the app.

**NB: Make sure you have downloaded [Ollama](https://ollama.com/) to your system.**

## References and Acknowledgments

References:
- This [original repository](https://github.com/tonykipkemboi/ollama_streamlit_demos) served as a great template to build this RAG on top of.
- This [tutorial](https://www.youtube.com/watch?v=bAI_jWsLhFM) was also really helpful.

👏 Kudos to the [Ollama](https://ollama.com/) team for their efforts in making open-source models more accessible!
