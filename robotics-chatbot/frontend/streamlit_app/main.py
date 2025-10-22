import streamlit as st
import os
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import tempfile
from pathlib import Path
import json

# Install required packages if not available
try:
    import google.generativeai as genai
except ImportError:
    st.error("Please install google-generativeai: pip install google-generativeai")
    st.stop()

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    st.error("Please install chromadb: pip install chromadb")
    st.stop()

try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    st.error("Please install PyMuPDF: pip install PyMuPDF")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Please install sentence-transformers: pip install sentence-transformers")
    st.stop()

try:
    from docx import Document
except ImportError:
    st.warning("python-docx not installed. DOCX files won't be supported. Install with: pip install python-docx")
    Document = None

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./data/chroma_db"

class RAGChatbot:
    """
    RAG-based chatbot using Gemini API and ChromaDB
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup_gemini()
        self.setup_chroma()
        self.setup_embeddings()
        
    def setup_gemini(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            st.success("✅ Gemini API initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
            st.stop()
    
    def setup_chroma(self):
        """Initialize ChromaDB"""
        try:
            # Create data directory
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="document_collection",
                metadata={"description": "RAG chatbot documents"}
            )
            
            st.success("✅ ChromaDB initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
            st.stop()
    
    def setup_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            st.success("✅ Embedding model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {str(e)}")
            st.stop()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        if Document is None:
            return ""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Failed to extract text from DOCX: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Failed to extract text from TXT: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks for embedding"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(self, file_path: str, filename: str) -> bool:
        """Add document to the vector database"""
        try:
            # Extract text based on file extension
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_ext in ['.txt', '.md']:
                text = self.extract_text_from_txt(file_path)
            else:
                st.error(f"Unsupported file type: {file_ext}")
                return False
            
            if not text.strip():
                st.error("No text could be extracted from the document")
                return False
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings and store
            doc_id = str(uuid.uuid4())
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Store in ChromaDB
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "filename": filename,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[chunk_id]
                )
            
            st.success(f"✅ Document '{filename}' added successfully! Created {len(chunks)} chunks.")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to add document: {str(e)}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        try:
            # Check if collection has any documents
            if self.collection.count() == 0:
                return []  # No documents to search
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini with RAG context when available, or general knowledge otherwise"""
        try:
            # Check if we have relevant document context
            has_relevant_context = context_docs and len(context_docs) > 0
            
            if has_relevant_context:
                # Check if documents are actually relevant (similarity threshold)
                relevant_docs = [doc for doc in context_docs if doc.get('similarity', 0) > 0.3]
                has_relevant_context = len(relevant_docs) > 0
                context_docs = relevant_docs
            
            if has_relevant_context:
                # Document-based response
                context = "Based on the following documents:\n\n"
                for i, doc in enumerate(context_docs, 1):
                    context += f"Document {i} (from {doc['metadata']['filename']}):\n"
                    context += f"{doc['content']}\n\n"
                
                prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

{context}

User Question: {query}

Instructions:
- Use the provided context to answer the question
- Be specific and cite which document you're referencing when possible
- If the context doesn't fully answer the question, mention what's missing
- Provide a helpful and comprehensive answer

Answer:"""
            else:
                # General knowledge response
                prompt = f"""You are a helpful AI assistant. Answer the user's question using your general knowledge.

User Question: {query}

Instructions:
- Provide a helpful, accurate, and comprehensive answer
- Use your general knowledge and understanding
- If you're not certain about something, mention it
- Be conversational and helpful

Answer:"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"❌ Failed to generate response: {str(e)}"
    
    def get_document_count(self) -> int:
        """Get total number of documents in the database"""
        try:
            count = self.collection.count()
            return count
        except:
            return 0

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        # Create initial conversation
        conversation_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = conversation_id
        st.session_state.conversations[conversation_id] = {
            "name": f"Conversation {datetime.now().strftime('%m/%d %H:%M')}",
            "messages": [],
            "created_at": datetime.now().isoformat()
        }
    if "conversation_storage_path" not in st.session_state:
        st.session_state.conversation_storage_path = "./data/conversations.json"

def load_conversations():
    """Load conversations from persistent storage"""
    try:
        storage_path = st.session_state.conversation_storage_path
        if os.path.exists(storage_path):
            with open(storage_path, 'r', encoding='utf-8') as f:
                loaded_conversations = json.load(f)
                st.session_state.conversations.update(loaded_conversations)
    except Exception as e:
        st.error(f"Failed to load conversations: {str(e)}")

def save_conversations():
    """Save conversations to persistent storage"""
    try:
        storage_path = st.session_state.conversation_storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        with open(storage_path, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save conversations: {str(e)}")

def create_new_conversation():
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    conversation_name = f"Conversation {datetime.now().strftime('%m/%d %H:%M')}"
    
    st.session_state.conversations[conversation_id] = {
        "name": conversation_name,
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    
    # Save current conversation before switching
    if st.session_state.current_conversation_id in st.session_state.conversations:
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages.copy()
    
    # Switch to new conversation
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = []
    
    save_conversations()
    return conversation_id

def switch_conversation(conversation_id: str):
    """Switch to a different conversation"""
    if conversation_id in st.session_state.conversations:
        # Save current conversation
        if st.session_state.current_conversation_id in st.session_state.conversations:
            st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages.copy()
        
        # Switch to selected conversation
        st.session_state.current_conversation_id = conversation_id
        st.session_state.messages = st.session_state.conversations[conversation_id]["messages"].copy()
        
        save_conversations()

def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in st.session_state.conversations and len(st.session_state.conversations) > 1:
        del st.session_state.conversations[conversation_id]
        
        # If we deleted the current conversation, switch to the first available one
        if conversation_id == st.session_state.current_conversation_id:
            first_conv_id = list(st.session_state.conversations.keys())[0]
            switch_conversation(first_conv_id)
        
        save_conversations()
        return True
    return False

def rename_conversation(conversation_id: str, new_name: str):
    """Rename a conversation"""
    if conversation_id in st.session_state.conversations:
        st.session_state.conversations[conversation_id]["name"] = new_name
        save_conversations()

def main():
    st.set_page_config(
        page_title="RAG Chatbot with Document Integration",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    load_conversations()  # Load saved conversations
    
    st.title("🤖 RAG Chatbot with Document Integration")
    st.markdown("Upload documents and ask questions based on their content!")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("💬 Conversations")
        
        # Create new conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            create_new_conversation()
            st.rerun()
        
        # Conversation selector
        conversation_options = []
        conversation_ids = []
        for conv_id, conv_data in st.session_state.conversations.items():
            conversation_options.append(conv_data["name"])
            conversation_ids.append(conv_id)
        
        if conversation_options:
            current_index = 0
            if st.session_state.current_conversation_id in conversation_ids:
                current_index = conversation_ids.index(st.session_state.current_conversation_id)
            
            selected_conversation = st.selectbox(
                "Select Conversation:",
                options=conversation_options,
                index=current_index,
                key="conversation_selector"
            )
            
            selected_id = conversation_ids[conversation_options.index(selected_conversation)]
            if selected_id != st.session_state.current_conversation_id:
                switch_conversation(selected_id)
                st.rerun()
        
        # Conversation management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Delete", use_container_width=True, disabled=len(st.session_state.conversations) <= 1):
                if delete_conversation(st.session_state.current_conversation_id):
                    st.rerun()
        
        with col2:
            if st.button("📝 Rename", use_container_width=True):
                st.session_state.show_rename = True
        
        # Rename dialog
        if getattr(st.session_state, 'show_rename', False):
            current_name = st.session_state.conversations[st.session_state.current_conversation_id]["name"]
            new_name = st.text_input("New name:", value=current_name, key="rename_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", use_container_width=True):
                    if new_name.strip():
                        rename_conversation(st.session_state.current_conversation_id, new_name.strip())
                        st.session_state.show_rename = False
                        st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_rename = False
                    st.rerun()
        
        st.divider()
        
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if api_key and not st.session_state.chatbot:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = RAGChatbot(api_key)
        
        if st.session_state.chatbot:
            st.success("🤖 Chatbot is ready!")
            
            # Document count
            doc_count = st.session_state.chatbot.get_document_count()
            
            # Conversation statistics
            current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
            message_count = len(current_conv["messages"])
            total_conversations = len(st.session_state.conversations)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_count)
                st.metric("Total Conversations", total_conversations)
            with col2:
                st.metric("Messages in Current", message_count)
                if current_conv["messages"]:
                    st.caption(f"Started: {datetime.fromisoformat(current_conv['created_at']).strftime('%m/%d %H:%M')}")
            
            st.divider()
            
            st.header("📚 Document Management")
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT, MD"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if st.button(f"Process {uploaded_file.name}", key=f"btn_{uploaded_file.name}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Add to RAG system
                            success = st.session_state.chatbot.add_document(tmp_file_path, uploaded_file.name)
                            
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                            if success:
                                st.rerun()
            
            # Clear current conversation button
            if st.button("🗑️ Clear Current Conversation"):
                st.session_state.confirm_clear_conv = True
            
            if getattr(st.session_state, 'confirm_clear_conv', False):
                st.warning("Are you sure you want to clear the current conversation?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Clear", type="secondary", use_container_width=True):
                        st.session_state.messages = []
                        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = []
                        save_conversations()
                        st.session_state.confirm_clear_conv = False
                        st.success("Current conversation cleared!")
                        st.rerun()
                with col2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.confirm_clear_conv = False
                        st.rerun()
            
            # Clear database
            if st.button("🗑️ Clear All Documents"):
                st.session_state.confirm_clear_docs = True
            
            if getattr(st.session_state, 'confirm_clear_docs', False):
                st.warning("Are you sure you want to clear all documents?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Clear All", type="secondary", use_container_width=True):
                        try:
                            st.session_state.chatbot.chroma_client.delete_collection("document_collection")
                            st.session_state.chatbot.setup_chroma()
                            st.session_state.confirm_clear_docs = False
                            st.success("All documents cleared!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear documents: {str(e)}")
                with col2:
                    if st.button("Cancel Clear", use_container_width=True):
                        st.session_state.confirm_clear_docs = False
                        st.rerun()
    
    # Main chat interface
    if not api_key:
        st.warning("👈 Please enter your Gemini API key in the sidebar to get started.")
        st.info("""
        To get your Gemini API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Click "Create API Key"
        3. Copy and paste it in the sidebar
        """)
        return
    
    if not st.session_state.chatbot:
        st.error("Failed to initialize chatbot. Please check your API key.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source['metadata']['filename']}")
                        st.write(f"*Similarity: {source['similarity']:.2f}*")
                        st.write(f"```\n{source['content'][:300]}...\n```")
                        st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Save to current conversation
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages.copy()
        save_conversations()
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Search for relevant documents first
            relevant_docs = st.session_state.chatbot.search_documents(prompt, n_results=5)
            
            # Check if we have documents and if they're relevant
            has_relevant_docs = relevant_docs and len(relevant_docs) > 0 and any(doc.get('similarity', 0) > 0.3 for doc in relevant_docs)
            
            if has_relevant_docs:
                with st.spinner("🔍 Searching documents and generating response..."):
                    response = st.session_state.chatbot.generate_response(prompt, relevant_docs)
            else:
                doc_count = st.session_state.chatbot.get_document_count()
                if doc_count == 0:
                    with st.spinner("💭 Generating response using general knowledge..."):
                        response = st.session_state.chatbot.generate_response(prompt, [])
                else:
                    with st.spinner("💭 No relevant documents found, using general knowledge..."):
                        response = st.session_state.chatbot.generate_response(prompt, [])
            
            st.markdown(response)
            
            # Show mode indicator
            if has_relevant_docs:
                st.info("📚 Response based on your uploaded documents")
            else:
                doc_count = st.session_state.chatbot.get_document_count()
                if doc_count == 0:
                    st.info("💭 Response using general knowledge (no documents uploaded)")
                else:
                    st.info("💭 Response using general knowledge (no relevant documents found)")
            
            # Show sources if available
            if has_relevant_docs:
                with st.expander("📚 Sources Used", expanded=False):
                    for i, doc in enumerate(relevant_docs, 1):
                        if doc.get('similarity', 0) > 0.3:  # Only show relevant sources
                            st.write(f"**Source {i}:** {doc['metadata']['filename']}")
                            st.write(f"*Similarity: {doc['similarity']:.2f}*")
                            st.write(f"```\n{doc['content'][:300]}...\n```")
                            st.write("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": relevant_docs
        })
        
        # Save updated conversation
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages.copy()
        save_conversations()
    
    # Usage instructions
    if not st.session_state.messages:
        st.markdown("""
        ### 🚀 How to use:
        1. **Add your Gemini API key** in the sidebar
        2. **Upload documents** (PDF, DOCX, TXT, MD) using the sidebar *(optional)*
        3. **Ask any questions** in the chat - both document-based and general questions!
        
        ### 📚 Document-based questions (when documents are uploaded):
        - "What is the main topic of the uploaded document?"
        - "Summarize the key points from the documents"
        - "What does the document say about [specific topic]?"
        
        ### 💭 General knowledge questions (works anytime):
        - "Explain quantum physics in simple terms"
        - "What's the weather like?" *(Note: I don't have real-time data)*
        - "Help me write a Python function"
        - "Tell me about machine learning"
        
        ### 🎯 Smart Response Mode:
        - **With documents**: I'll search your documents first, then use general knowledge if needed
        - **Without documents**: I'll use my general knowledge to help you
        - **Mixed mode**: I can combine document insights with general knowledge
        
        The chatbot automatically adapts to provide the best possible answer! 🤖✨
        """)

if __name__ == "__main__":
    main()
