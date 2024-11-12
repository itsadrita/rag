from dotenv import load_dotenv
from gtts import gTTS
import os
import io
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    """Loads a PDF document using PyMuPDF and returns the extracted text."""
    print("Loading document using PyMuPDF...")
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
    print(f"Loaded {len(text)} characters of text.")
    # Return a list of Document objects with page_content as the text
    return [Document(page_content=text)]

def setup_vectorstore(documents):
    """Sets up a FAISS vector store from the loaded documents."""
    print("Setting up vector store...")
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Use newline as separator
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(doc_chunks)} chunks.")
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    print("Vector store setup complete.")
    return vectorstore

def get_response(vectorstore, question, chat_history):
    """Fetches the response using the vector store and ChatGroq, with TTS."""
    print("Searching in vector store and generating response...")
    
    # Search in vector store for relevant chunks
    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # Build prompt with conversation history and context
    full_prompt = "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in chat_history])
    full_prompt += f"\nUser: {question}\nContext: {context}\nAssistant:"

    # Call the LLaMA model via ChatGroq
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    response = llm.invoke(full_prompt)  # Use `invoke` instead of direct call
    
    # Extract the response content
    assistant_response = response.content
    
    # Convert response to speech
    tts = gTTS(assistant_response, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)  # Reset to start of the stream for playback

    return assistant_response, audio_fp

# Streamlit page setup
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with Doc - LLAMA 3.1")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader to upload PDF files
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"

    # Save the uploaded PDF
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        print(f"File uploaded and saved as: {file_path}")

    # Load document and setup vectorstore
    if "vectorstore" not in st.session_state:
        print("Loading document and setting up vectorstore...")
        documents = load_document(file_path)
        st.session_state.vectorstore = setup_vectorstore(documents)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field for asking questions
user_input = st.chat_input("Ask your question...")

if user_input:
    print(f"User input: {user_input}")

    # Save user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response using vectorstore and LLaMA model
    with st.chat_message("assistant"):
        assistant_response, audio_fp = get_response(st.session_state.vectorstore, user_input, st.session_state.chat_history)
        print(f"Assistant response: {assistant_response}")

        # Display assistant response and play audio
        st.markdown(assistant_response)
        st.audio(audio_fp, format="audio/mp3")
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})