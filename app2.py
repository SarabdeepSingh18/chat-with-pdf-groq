import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit UI
st.set_page_config(page_title="Chat with PDFs using Llama3")
st.title("🔍 Chat with your PDFs using Llama3 via Groq")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Use the context below to answer the question accurately.

<context>
{context}
</context>

Question: {input}
Answer:
""")

# Function to create vector store
def vector_embedding():
    with st.spinner("Loading and indexing documents..."):
        # Load PDFs
        loader = PyPDFDirectoryLoader("./uspdf")
        raw_docs = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_docs[:20])  # Limit if needed

        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Store in session
        st.session_state.vectors = vectorstore
        st.session_state.documents = documents
        st.success("✅ Vector store created successfully!")

# User input
user_question = st.text_input("Ask a question about your documents")

# Vector creation button
if st.button("🔃 Embed Documents"):
    vector_embedding()

# QA Process
if user_question and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("🔎 Thinking..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        end = time.process_time()

    st.markdown(f"⏱️ **Response time**: {end - start:.2f} seconds")
    st.subheader("📌 Answer")
    st.write(response.get('answer', "No answer found."))

    # Show context
    with st.expander("📄 Retrieved Document Chunks"):
        context_docs = response.get("context", [])
        if context_docs:
            for doc in context_docs:
                st.markdown(f"- {doc.page_content[:500]}...")  # limit preview
        else:
            st.info("No relevant documents retrieved.")
else:
    if user_question:
        st.warning("Please embed documents first by clicking the 'Embed Documents' button.")
