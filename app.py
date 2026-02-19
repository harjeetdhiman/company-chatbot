
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
#from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
import requests
import os


app = FastAPI(title="Simple RAG Chatbot")

API_KEY = os.getenv("OPENROUTER_API_KEY", "")           # ← fill this
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


def embed(text: str) -> list:
    if not API_KEY:
        return [0.0] * 1536
    try:
        r = requests.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/text-embedding-3-small",
                "input": text
            }
        )
        data = r.json()
        if "data" not in data or not data["data"]:
            print("Embedding error:", data)
            return [0.0] * 1536
        return data["data"][0]["embedding"]
    except Exception as e:
        print("Embed exception:", e)
        return [0.0] * 1536

def ask_llm(context: str, question: str) -> str:
    if not API_KEY:
        return "API key missing"

    try:
        r = requests.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": f"""
You are a strict company information assistant. You MUST follow these rules EXACTLY, no exceptions:

1. ONLY use information from the provided Context below. Do NOT use your own knowledge, do NOT guess, do NOT add extra information.
2. If the question has NOTHING to do with the company, profile, employees, services, or ANY data in the Context → reply EXACTLY with this sentence and nothing else: "Not related to FDA"
3. If the question is even slightly related or can be answered using the Context → give a short, direct answer using ONLY the Context.
4. Mixing Hindi and English is allowed ONLY if the user's question uses Hinglish. Otherwise use simple English or Hindi based on the question.
5. Answers must be very short: 1-3 sentences maximum.
6. NEVER make up facts. If unsure or info missing → say "Not related to profile"
7. If the user message is a short polite follow-up, acknowledgement, greeting or casual opener 
   (examples: thank you, thanks, ok, okay, got it, understood, great, nice, bye, good, 
   hello, hlo, hey, hi, हैलो, हेलो, हाय, है, नमस्ते, bie,bye-bye, 
   थैंक्स, ओके, अच्छा, ठीक, बढ़िया, हां, जी, सुप, यो)
   → reply very briefly and politely in the same language/style, examples:
      - "You're welcome"
      - "Glad I could help"
      - "Okay"
      - "Hi! 😊"
      - "हैलो!"
      - "हाय"
      - "नमस्ते"
      - "ठीक है"
      - "बढ़िया"
      - "हां जी"
   → Do NOT say "Not related to FDA" for these short polite/greeting messages.

Context (this is the ONLY information you can use):
{context}
"""
                    },
                    {"role": "user", "content": question}
                ],
                "temperature": 0.4,
                "max_tokens": 400
            }
        )
        data = r.json()
        if "choices" not in data or not data["choices"]:
            return "Error or quota finished"
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LLM error:", e)
        return "Sorry, something went wrong"


#client = chromadb.Client()

persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data") # add
client = chromadb.PersistentClient(path=persist_dir) #add 


collection = client.get_or_create_collection("company")



if collection.count() == 0:
    print("Loading knowledge base...")
    if not os.path.exists("data.txt"):
        print("⚠️ data.txt not found!")
    else:
        with open("data.txt", encoding="utf-8") as f:
            text = f.read()
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[str(i)],
                embeddings=[embed(chunk)],
                documents=[chunk]
            )
        print(f"✅ Loaded {len(chunks)} chunks")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    q_emb = embed(req.message)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=4
        
    )

    if not results["documents"] or not results["documents"][0]:
        context = ""
    else:
        context = "\n\n".join(results["documents"][0])

    answer = ask_llm(context, req.message)
    return {"response": answer}

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    with open("index.html", encoding="utf-8") as f:
        return f.read()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)