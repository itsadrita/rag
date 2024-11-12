# Signify RAG

Signify RAG is an advanced Retrieval-Augmented Generation (RAG) tool designed to help users, especially older adults, understand legal documents more effectively. It uses cutting-edge natural language processing techniques, including embeddings and transformers, to enable interactive document exploration, simplifying complex legal language and reducing the risk of scams or misunderstandings.

## Features

- **Legal Document Upload & Embedding**: Users can upload legal documents, which are then embedded for efficient analysis.
- **Multi-Language Chat Interface**: Interact with the document in multiple languages, making legal text accessible to a broader audience.
- **Retrieval-Augmented Generation (RAG)**: Uses RAG to accurately retrieve relevant sections from the document and generate coherent, context-aware responses.
- **Text-to-Speech (TTS)**: Provides audio responses to enhance accessibility, especially for users who may struggle with reading on screens.
- **Intuitive UI via Streamlit**: Easy-to-use interface hosted on Streamlit

## Tech Stack

- **Python**: Core language for backend and processing logic.
- **LangChain**: Framework for RAG and handling document embeddings.
- **PyMuPDF**: PDF handling and manipulation.
- **Faiss**: Efficient similarity search on embeddings.
- **Transformers**: NLP capabilities powered by transformer models, including BERT.
- **Streamlit**: Frontend interface for interactive document exploration.
- **Flutter**: Mobile app wrapper using WebView for accessing the Streamlit app on mobile devices.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/itsadrita/rag.git
   cd rag_github


## Install Dependencies: Ensure you have Python 3.7 or later, then install dependencies:

```bash
pip install -r requirements.txt

Set Up Environment Variables: Create a .env file in the root directory to store environment variables such as API keys and configuration settings.

Run the Application: Start the Streamlit app:

streamlit run app.py
Access the Application: Open http://localhost:8501 in your browser. For mobile access, open the URL in the WebView of your Flutter app.


