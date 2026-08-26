import os
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NVIDIA NIM Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")

@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "service": "NVIDIA NIM Proxy"}

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY (or NIM_API_KEY) is missing")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Remove junk that can break NIM
    payload.pop("extra_body", None)
    payload.pop("logit_bias", None)
    payload.pop("thinking", None)          # wrong for NIM MiniMax
    payload.pop("reasoning_split", None)   # MiniMax-native only
    payload.pop("include_reasoning", None)

    model_name = (payload.get("model") or "").lower()

    # ---------- Thinking (NIM-correct) ----------
    if "gemma" in model_name:
        # Official for Gemma 4 on NIM
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    elif "minimax" in model_name:
        # Official for MiniMax M3 on NVIDIA NIM (not MiniMax's own API)
        payload["chat_template_kwargs"] = {"thinking_mode": "enabled"}

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/event-stream" if payload.get("stream") else "application/json",
    }

    is_stream = payload.get("stream", False)
    timeout = httpx.Timeout(300.0, connect=30.0)

    if is_stream:
        async def stream_generator():
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream("POST", NVIDIA_API_URL, json=payload, headers=headers) as response:
                            if response.status_code == 429:
                                wait = 20 + (attempt * 20)
                                if attempt < 3:
                                    await asyncio.sleep(wait)
                                    continue
                                yield b'data: {"error":{"message":"NVIDIA rate limit (429). Wait 5-10 min."}}\n\n'
                                return

                            if response.status_code != 200:
                                body = await response.aread()
                                msg = body.decode(errors="ignore")[:400].replace('"', "'")
                                yield f'data: {{"error":{{"message":"NVIDIA {response.status_code}: {msg}"}}}}\n\n'.encode()
                                return

                            async for chunk in response.aiter_bytes():
                                yield chunk
                            return
                except Exception as e:
                    if attempt == 3:
                        yield f'data: {{"error":{{"message":"Proxy error: {str(e)}"}}}}\n\n'.encode()
                    else:
                        await asyncio.sleep(5)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(4):
                res = await client.post(NVIDIA_API_URL, json=payload, headers=headers)

                if res.status_code == 429:
                    if attempt < 3:
                        await asyncio.sleep(20 + (attempt * 20))
                        continue
                    raise HTTPException(status_code=429, detail="NVIDIA rate limit (429)")

                if res.status_code != 200:
                    raise HTTPException(status_code=res.status_code, detail=res.text[:500])

                return res.json()
