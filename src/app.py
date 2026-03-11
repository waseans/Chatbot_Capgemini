import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Capgemini GenAI Intern Bot", page_icon="💙")
st.title("🤖 Capgemini Knowledge Assistant")
st.markdown("Query the internal company policy and 2026 strategic data.")

# --- Initialization ---
@st.cache_resource  # Keeps the DB loaded so it's fast
def load_rag_system():

    # Use the exact string from your terminal output
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Load the database we created in ingestion.py
    vector_db = Chroma(
        persist_directory="./chromadb",
        embedding_function=embeddings
    )
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", # High-volume, high-quota model for 2026
        temperature=0.2
    )
    
    # Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain


qa_system = load_rag_system()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me about Capgemini 2026 strategy..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Consulting internal documents..."):

            response = qa_system.invoke({"query": prompt})

            answer = response["result"]

            # Show the answer
            st.markdown(answer)

            # Show sources
            with st.expander("🔍 View Sources"):
                for doc in response["source_documents"]:
                    st.write(
                        f"- {doc.metadata.get('source','Document')} (Page {doc.metadata.get('page','N/A')})"
                    )
                    st.caption(f"Snippet: {doc.page_content[:150]}...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )