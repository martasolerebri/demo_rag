import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Goodreads AI Librarian", page_icon="📖", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.title("Goodreads AI Librarian")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    st.markdown("Get your keys at [Groq](https://groq.com/) and [HuggingFace](https://huggingface.co/).")
    
    st.divider()
    
    st.header("Upload Library")
    uploaded_file = st.file_uploader("Upload Goodreads CSV", type="csv")
    
    if uploaded_file:
        st.success("Library loaded")
    
    st.divider()
    st.markdown("""
    **How to export:**
    1. Go to Goodreads desktop
    2. Go to My Books
    3. Click on Import and export under Tools on the left
    4. Click on the Export Library button
    5. Click on Your export from (date) - (time) to download the csv file
    """)
    
    id_model = "llama-3.3-70b-versatile"
    temperature = 0.5

if not groq_api_key or not hf_api_key:
    st.warning("Please enter both API Keys in the sidebar to begin")
    st.stop()

@st.cache_resource
def load_models(groq_key, hf_key):
    llm = ChatGroq(api_key=groq_key, model=id_model, temperature=temperature)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

llm, embeddings = load_models(groq_api_key, hf_api_key)

@st.cache_data
def parse_goodreads_csv(file):
    df = pd.read_csv(file)
    
    required_cols = ['Title', 'Author']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return None, None
    
    books = []
    for idx, row in df.iterrows():
        exclusive_shelf = str(row.get('Exclusive Shelf', 'read')).lower()
        
        book = {
            'id': idx,
            'title': str(row.get('Title', 'Unknown')),
            'author': str(row.get('Author', 'Unknown')),
            'rating': row.get('My Rating', 0),
            'year_published': row.get('Year Published', ''),
            'date_read': str(row.get('Date Read', '')),
            'date_added': str(row.get('Date Added', '')),
            'exclusive_shelf': exclusive_shelf,
            'shelves': str(row.get('Bookshelves', '')),
            'review': str(row.get('My Review', '')),
            'avg_rating': row.get('Average Rating', 0),
            'num_pages': row.get('Number of Pages', 0),
        }
        
        text_parts = [f"{book['title']} by {book['author']}"]
        if book['shelves'] and book['shelves'] != 'nan':
            text_parts.append(f"Genres/Tags: {book['shelves']}")
        if book['review'] and book['review'] != 'nan' and len(book['review']) > 10:
            text_parts.append(f"Review: {book['review'][:500]}")
        
        book['embedding_text'] = ". ".join(text_parts)
        books.append(book)
    
    return books, df

def process_csv_to_retriever(books, embeddings_model):
    documents = []
    for book in books:
        content = f"Title: {book['title']}\nAuthor: {book['author']}\n"
        content += f"My Rating: {book['rating']}/5\n"
        content += f"Status: {book['exclusive_shelf']}\n"
        content += f"Shelves/Tags: {book['shelves']}\n"
        if book['review'] and book['review'] != 'nan':
            content += f"My Review: {book['review']}\n"
            
        doc = Document(page_content=content, metadata={"title": book['title'], "author": book['author']})
        documents.append(doc)
        
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

if uploaded_file:
    if 'books' not in st.session_state or 'df' not in st.session_state:
        books, df = parse_goodreads_csv(uploaded_file)
        if books is None:
            st.stop()
        
        st.session_state.books = books
        st.session_state.df = df
    
    books = st.session_state.books
    df = st.session_state.df
    
    if 'retriever' not in st.session_state:
        with st.spinner("Processing library for chat..."):
            st.session_state.retriever = process_csv_to_retriever(books, embeddings)
    
    st.markdown("### Chat with your personal Librarian")
    st.markdown("*Ask questions like: 'Recommend me something from my to-read list'*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about your books..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        system_prompt = (
            "You are a friendly, knowledgeable AI librarian analyzing a user's Goodreads library. "
            "Use the following retrieved context (which contains books the user has shelved, their ratings, and their reviews) "
            "to answer their question or provide recommendations. "
            "If they ask for recommendations, prioritize books from their own library context. "
            "Keep your responses conversational, insightful, and concise. "
            "If you don't know the answer, just say so.\n\n"
            "Context: {context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = (
            {"context": st.session_state.retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Scanning the shelves..."):
                response = chain.invoke(prompt)
                clean_response = response.split("</think>")[-1].strip() if "</think>" in response else response
                
                st.write(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})

else:
    st.info("Upload your Goodreads library CSV from the sidebar to get started")