from langchain.prompts import ChatPromptTemplate, PromptTemplate  # type: ignore
from langchain.retrievers.multi_query import MultiQueryRetriever  # type: ignore
from langchain_community.chat_models import ChatOllama  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_core.runnables import RunnablePassthrough  # type: ignore

persist_directory = "vectorstore"

# Much smaller and faster model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Only ~80MB vs larger medical models
)

print(f"Loading existing vector store from {persist_directory}")
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

retriver = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

template = """Answer the question. based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriver, "question": RunnablePassthrough()}    # pylint: disable=unsupported-binary-operation
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": "What is CRISPR?"}))
