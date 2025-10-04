import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.vectorstores import SKLearnVectorStore

#
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

GEMINI_EMBEDDING_MODEL = "models/text-embedding-004" 
# 'text-embedding-004' is used

RAG_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use bullet points and other formatting methods where neccessary. You can make deductions and logical conclusions
from given context.

Conversation: {conversation}
Context: {context} 
Question: {question}
Answer:"""

def get_vector_db_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    
    # Reworked for Gemini Embeddings
    # Use task_type="RETRIEVAL_DOCUMENT" for creating documents
    # and "RETRIEVAL_QUERY" for the query itself (handled by the retriever by default)
    gemini_embedding = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL
        
    )

   
    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=gemini_embedding,
            persist_path=persist_path,
            serializer="parquet"
        )
        # Note: The retriever automatically uses the embedding for the query.
        return vectorstore.as_retriever(lambda_mult=0)

    # Otherwise, index LangSmith documents and create new vector store
    print("Vector store not found. Indexing documents...")
    ls_docs_sitemap_loader = SitemapLoader(web_path="https://docs.smith.langchain.com/sitemap.xml", continue_on_failure=True)
    ls_docs = ls_docs_sitemap_loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(ls_docs)
    print(f"Split {len(ls_docs)} documents into {len(doc_splits)} chunks.")

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=gemini_embedding, # UPDATED
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    print(f"Vector store saved to {persist_path}")
    return vectorstore.as_retriever(lambda_mult=0)