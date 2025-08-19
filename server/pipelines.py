import os
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"
os.environ["HF_HOME"] = "/app/cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/cache"
import logging
import time
from haystack.utils import Secret
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter
import google.generativeai as genai
from google.generativeai import types

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Document store and components
document_store = InMemoryDocumentStore()

# Optimized for CPU
doc_embedder = SentenceTransformersDocumentEmbedder(
    model="BAAI/bge-base-en-v1.5",  # Smaller model
)
text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-base-en-v1.5"
)
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=3) 

def initialize_ranker():
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        try:
            logger.info(f"Attempt {attempts + 1} to load ranker model...")
            ranker = SentenceTransformersSimilarityRanker(
                model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
            )
            ranker.warm_up()
            return ranker
        except Exception as e:
            attempts += 1
            if attempts == max_attempts:
                logger.warning(f"All {max_attempts} attempts failed: {e}")
                raise
            wait_time = min(2 ** attempts, 10)  # Exponential backoff
            time.sleep(wait_time)

# Initialize ranker with retry logic and fallback
try:
    reranker = initialize_ranker()
except Exception as e:
    logger.error(f"All retries failed for ranker model: {e}")
    logger.info("Proceeding without ranker - using simple retrieval only")
    class DummyRanker:
        def run(self, query, documents):
            return {"documents": documents[:3]}
    reranker = DummyRanker()

# Configure GenAI SDK
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize generator with safety settings
generator = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
)

splitter = DocumentSplitter(
    split_by="word",
    split_length=300,  # Smaller chunks
    split_overlap=30
)

# Warm up components
try:
    logger.info("Warming up components...")
    doc_embedder.warm_up()
    text_embedder.warm_up()
    if hasattr(reranker, 'warm_up'):  # Only warm up if the component supports it
        reranker.warm_up()
    logger.info("Components warmed up")
except Exception as e:
    logger.error(f"Warmup failed: {e}")

def add_documents(texts: list[str], meta_list: list[dict]):
    """Process and store documents with chunking"""
    # Create base documents
    docs = [
        Document(content=text, meta=meta)
        for text, meta in zip(texts, meta_list)
        if text and text.strip()
    ]
    
    if not docs:
        return 0
        
    # Split into chunks
    split_result = splitter.run(docs)
    split_docs = split_result.get("documents", [])
    
    if not split_docs:
        return 0
        
    # Batch embedding with reduced batch size
    embedded_docs = []
    batch_size = 8  # Reduced for CPU
    
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i+batch_size]
        try:
            embedded_batch = doc_embedder.run(batch).get("documents", [])
            embedded_docs.extend(embedded_batch)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
    
    if embedded_docs:
        document_store.write_documents(embedded_docs)
    return len(embedded_docs)

def query_rag(question: str, session_id: str):
    """Query the RAG system with session filtering"""
    try:
        # Validate input
        if not question.strip():
            return {
                "answer": "Please provide a non-empty question.",
                "sources": []
            }
            
        # Embed question
        embedding_result = text_embedder.run(question)
        query_emb = embedding_result.get("embedding")
        
        if not query_emb:
            return {
                "answer": "Failed to process your question.",
                "sources": []
            }
        
        # Retrieve documents with session filter
        filters = {"field": "meta.session_id", "operator": "==", "value": session_id}
        retrieved_docs = retriever.run(
            query_embedding=query_emb, 
            filters=filters
        ).get("documents", [])
        
        if not retrieved_docs:
            return {
                "answer": "No documents found for this session. Please upload a file first.",
                "sources": []
            }
        
        # Rerank documents (limit to top 3)
        reranked_docs = reranker.run(
            query=question, 
            documents=retrieved_docs[:5]  # Limit input
        ).get("documents", [])[:3]  # Return top 3
        
        # Generate answer with context
        context = "\n\n".join([doc.content for doc in reranked_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Handle generator response safely
        response = generator.generate_content(prompt)
        answer = response.text if response and hasattr(response, 'text') else "No response generated"
        
        # Format sources
        sources = [
            {
                "filename": d.meta.get("filename", "Unknown"),
                "page": d.meta.get("page", 1),
                "snippet": d.content[:200] + "..." if len(d.content) > 200 else d.content
            } 
            for d in reranked_docs
        ]
        
        return {"answer": answer, "sources": sources}
    
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        return {
            "answer": "Sorry, I encountered an error processing your request.",
            "sources": []
        }