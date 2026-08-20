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

    payload.pop("extra_body", None)
    payload.pop("logit_bias", None)

    model_name = (payload.get("model") or "").lower()

    # Enable thinking for Gemma
    if "gemma" in model_name:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True
        }

    # Stronger activation for MiniMax-M3
    if "minimax" in model_name:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True
        }
        # Some MiniMax versions respond better to this extra parameter
        payload["reasoning"] = True

    # Note: mistral-nemotron removed because it crashes with enable_thinking on free NIM

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    is_stream = payload.get("stream", False)

    if is_stream:
        async def stream_generator():
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(150.0, connect=25.0)) as client:
                        async with client.stream("POST", NVIDIA_API_URL, json=payload, headers=headers) as response:
                            if response.status_code == 429:
                                wait = 20 + (attempt * 20)
                                if attempt < 3:
                                    await asyncio.sleep(wait)
                                    continue
                                else:
                                    yield b'data: {"error": "NVIDIA rate limit (429). Please wait 5-10 minutes."}\n\n'
                                    return

                            if response.status_code != 200:
                                yield f'data: {{"error": "NVIDIA error {response.status_code}"}}\n\n'.encode()
                                return

                            async for chunk in response.aiter_bytes():
                                yield chunk
                            return
                except Exception as e:
                    if attempt == 3:
                        yield f'data: {{"error": "Proxy error: {str(e)}"}}\n\n'.encode()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(150.0, connect=25.0)) as client:
            for attempt in range(4):
                res = await client.post(NVIDIA_API_URL, json=payload, headers=headers)

                if res.status_code == 429:
                    if attempt < 3:
                        await asyncio.sleep(20 + (attempt * 20))
                        continue
                    else:
                        raise HTTPException(status_code=429, detail="NVIDIA rate limit (429). Please wait 5-10 minutes.")

                if res.status_code != 200:
                    raise HTTPException(status_code=res.status_code, detail=res.text)

                return res.json()
