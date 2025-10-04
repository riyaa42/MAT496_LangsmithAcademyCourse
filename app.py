
import os
import tempfile
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from typing import List
from langchain.schema import Document
import nest_asyncio


MODEL_NAME = "gemini-2.0-flash"
APP_VERSION = 1.0
MODEL_PROVIDER = "google"
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. You can use text formatting like bullet points if neccessary to be clear.
"""

os.environ["GOOGLE_API_KEY"] = "AIzaSyCQamN3b9pMy_5QUWs7s4K4n85RXbBGH-I"

llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

def get_vector_db_retriever():
    persist_path = os.path.join(tempfile.gettempdir(), "high_fantasy.parquet")
    
    # Local HuggingFace embeddings - no API, no quota limits
    embd = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_path):
        vectorstore = SKLearnVectorStore(
            embedding=embd,
            persist_path=persist_path,
            serializer="parquet"
        )
        return vectorstore.as_retriever(lambda_mult=0)

    # High fantasy documents
    custom_docs = [
        Document(page_content="The Lord of the Rings is an epic fantasy trilogy about a hobbit named Frodo who must destroy a powerful ring to save Middle-earth from the dark lord Sauron. The story follows his journey with a fellowship of companions across dangerous lands. It explores themes of friendship, courage, and the corrupting nature of power.", metadata={"source": "lotr"}),
        Document(page_content="The Name of the Wind was written by Patrick Rothfuss. It's the first book in The Kingkiller Chronicle series, following the story of Kvothe, a legendary figure recounting his past. The book was published in 2007 and became a bestseller.", metadata={"source": "notw"}),
        Document(page_content="Brandon Sanderson is known for creating hard magic systems with clear rules and limitations. His magic systems are logical and internally consistent, allowing readers to understand and predict how magic works. Examples include Allomancy in Mistborn and Stormlight in The Stormlight Archive.", metadata={"source": "sanderson"}),
        Document(page_content="The Wheel of Time is a 14-book epic fantasy series by Robert Jordan, completed by Brandon Sanderson. It follows Rand al'Thor and his friends as they navigate a world where the Dark One threatens to break free. The series features complex magic, political intrigue, and an expansive world with diverse cultures.", metadata={"source": "wot"}),
        Document(page_content="The major houses in A Song of Ice and Fire include Stark, Lannister, Targaryen, Baratheon, and Greyjoy. Each house has distinct characteristics, sigils, and words. The series explores their political conflicts and power struggles across the Seven Kingdoms.", metadata={"source": "asoiaf"}),
        Document(page_content="High fantasy is set in completely fictional worlds with their own rules and magic systems, like Middle-earth or Roshar. Low fantasy is set in the real world with magical elements added, or has minimal magic. High fantasy typically features epic quests and world-threatening stakes.", metadata={"source": "fantasy_types"}),
        Document(page_content="The main characters in The Stormlight Archive include Kaladin, a former soldier with depression, Shallan, a scholar with a dark past, and Dalinar, a warlord seeking redemption. Each character has their own flashback book exploring their history. The series is known for its deep character development and mental health representation.", metadata={"source": "stormlight"}),
        Document(page_content="Mistborn's magic system, called Allomancy, allows users to ingest and burn metals to gain different powers. Each metal provides a specific ability, like enhanced strength or emotional manipulation. Mistborn can burn all metals, while Mistings can only burn one type.", metadata={"source": "mistborn"}),
        Document(page_content="The Malazan Book of the Fallen is challenging because it starts in medias res with little explanation, features hundreds of characters across multiple continents, and has a complex non-linear timeline. The series doesn't hold the reader's hand and requires careful attention. However, it's praised for its scope, philosophical depth, and payoff.", metadata={"source": "malazan"}),
        Document(page_content="Grimdark fantasy is a subgenre characterized by morally ambiguous characters, graphic violence, and cynical or pessimistic tones. Examples include The First Law trilogy by Joe Abercrombie and The Black Company by Glen Cook. It often subverts traditional fantasy tropes and avoids clear good versus evil narratives.", metadata={"source": "grimdark"}),
    ]

    vectorstore = SKLearnVectorStore.from_documents(
        documents=custom_docs,
        embedding=embd,
        persist_path=persist_path,
        serializer="parquet"
    )
    vectorstore.persist()
    return vectorstore.as_retriever(lambda_mult=0)

nest_asyncio.apply()
retriever = get_vector_db_retriever()

@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "user",
            "content": f"{RAG_SYSTEM_PROMPT}\n\nContext: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_gemini(messages)

@traceable(run_type="llm", metadata={"ls_provider": MODEL_PROVIDER, "ls_model_name": MODEL_NAME})
def call_gemini(messages: List[dict]) -> str:
    return llm.invoke(messages)

@traceable(run_type="chain")
def langsmith_rag(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.content