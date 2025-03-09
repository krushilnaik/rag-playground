import os

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

llm_config = {"base_url": "http://localhost:1234/v1", "api_key": SecretStr("not-needed")}

KNOWLEDGE_BASE = "./documents"
VECTOR_DB_PATH = "./vector_db"


# Initialize Vector Store
def init_vectordb(local_dir: str, persist_directory: str) -> Chroma:
    """
    Build a vector store from local HTML files
    """

    # Load existing vector database
    if os.path.exists(persist_directory):
        print("Loading existing vector database...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(
                **llm_config,
                model="text-embedding-granite-embedding-125m-english",
                check_embedding_ctx_length=False,
            ),
        )

    documents = []

    # Load all HTML files from the directory
    for filename in os.listdir(local_dir):
        if filename.endswith(".html"):
            file_path = os.path.join(local_dir, filename)
            try:
                loader = BSHTMLLoader(file_path, open_encoding="utf8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        persist_directory=VECTOR_DB_PATH,
        embedding=OpenAIEmbeddings(
            **llm_config,
            model="text-embedding-granite-embedding-125m-english",
            check_embedding_ctx_length=False,
        ),
    )

    return vectorstore


vector_db = init_vectordb(KNOWLEDGE_BASE, VECTOR_DB_PATH)

retriever = vector_db.as_retriever()


# Prompt
prompt = PromptTemplate(
    template="""
        You are an assistant for question-answering tasks. You must answer the question using only the context provided. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Do not mention your sources in your answer. Use two sentences maximum and keep the answer concise. 
        Question: {question} 
        Context: {context} 
        Answer: 
    """,
    input_variables=["question", "context"],
)

llm = ChatOpenAI(**llm_config, model="llama-3.2-3b-instruct", temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
question = "Who was the first Pokemon?"
docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)
