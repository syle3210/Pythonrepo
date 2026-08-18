import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="NVIDIA NIM Gemma 4 Fix Proxy")

# ADJUST THIS: 10 to 14 messages stops the Gemma prefill attention crash
MAX_HISTORY_MESSAGES = 12  
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY environment variable is missing on Render.")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload sent from JanitorAI.")

    # 1. OPTIMIZATION: Extract the essential text data and strip massive layout arrays
    if "messages" in payload and isinstance(payload["messages"], list):
        cleaned_messages = []
        
        # Always retain your character card/system instructions
        system_msg = next((m for m in payload["messages"] if m.get("role") == "system"), None)
        if system_msg:
            cleaned_messages.append(system_msg)
            
        # Extract the user and bot conversational dialogue
        chat_history = [m for m in payload["messages"] if m.get("role") != "system"]
        
        # 2. OPTIMIZATION: Cut down context length to stop FlashAttention stalls
        if len(chat_history) > MAX_HISTORY_MESSAGES:
            chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
            
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            
            # If JanitorAI sends structural multi-part text blocks, collapse them into flat strings
            if isinstance(content, list):
                text_pieces = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_pieces)
            
            if role and content:
                cleaned_messages.append({"role": role, "content": str(content)})
                
        payload["messages"] = cleaned_messages

    # 3. OPTIMIZATION: Clean out conflicting features (biases, deep-chain structures)
    payload.pop("extra_body", None)
    payload.pop("logit_bias", None)
    
    # Cap excessive generations to prevent the cloud instance from choking midway
    if payload.get("max_tokens", 0) > 800:
        payload["max_tokens"] = 800

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    is_stream = payload.get("stream", False)

    # 4. OPTIMIZATION: Directly feed incoming streaming tokens to eliminate long front-end pauses
    if is_stream:
        async def stream_generator():
            try:
                async with client.stream("POST", NVIDIA_API_URL, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        yield f"data: {{\"error\": \"Nvidia NIM error code {response.status_code}\"}}\\n\\n".encode("utf-8")
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as e:
                yield f"data: {{\"error\": \"Proxy stream drop: {str(e)}\"}}\\n\\n".encode("utf-8")
            finally:
                await client.aclose()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            res = await client.post(NVIDIA_API_URL, json=payload, headers=headers)
            await client.aclose()
            return res.json()
        except Exception as e:
            await client.aclose()
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
