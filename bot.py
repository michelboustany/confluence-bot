import os
from dotenv import load_dotenv

# Slack Libraries
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_core.prompts import PromptTemplate

# AI Libraries (The Classic Stack)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# 1. Load Secrets
load_dotenv()

# --- 2. Setup the AI Brain ---
print("🧠 Initializing AI Components...")

# A. Embeddings (Must match ingest.py)
# We set this to the 3072-dimension model to match your new Pinecone index
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# --- CUSTOM PROMPT ---
friendly_template = """You are a helpful and friendly assistant for our team.
Use the following pieces of context to answer the question at the end.
If the answer is not in the context, or if the user is just saying hello,
use your general knowledge to be helpful and polite.
Do NOT just say "I don't know" if it's a simple greeting.

Context:
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=friendly_template, 
    input_variables=["context", "question"]
)

# B. Vector Store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# C. The LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# D. The Chain (Classic RetrievalQA)
# This automatically handles the "prompt" and "stuffing" documents for you
# D. The Chain (Classic RetrievalQA)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # <--- THIS IS THE FIX
)

# --- 3. Setup Slack Interface ---
print("⚡️ Connecting to Slack...")
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

@app.event("app_mention")
def handle_mention(event, say):
    user_text = event["text"]
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])

    print(f"   - Question: {user_text}")
    temp_msg = say(text="Let me check the docs... 🧠", thread_ts=thread_ts)

    try:
        # Run the Classic Chain
        # Note: The input key is "query", not "input" for this specific chain type
        response = qa_chain.invoke({"query": user_text})
        
        answer = response["result"]
        source_docs = response["source_documents"]

        # Build Source List
        sources = [doc.metadata.get("title", "Untitled") for doc in source_docs]
        unique_sources = list(set(sources))
        
        source_text = ""
        if unique_sources:
            source_text = "\n\n*📚 Sources:* " + ", ".join(unique_sources)

        # Update Slack
        app.client.chat_update(
            channel=channel_id,
            ts=temp_msg["ts"],
            text=f"{answer}{source_text}"
        )
        print("   - ✅ Answer sent!")

    except Exception as e:
        print(f"   - ❌ Error: {e}")
        app.client.chat_update(
            channel=channel_id,
            ts=temp_msg["ts"],
            text=f"I crashed! Error: {str(e)}"
        )

# --- 4. Start ---
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    print("🚀 Bot is up and running!")
    handler.start()