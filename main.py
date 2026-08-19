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

    # Clean only problematic fields
    payload.pop("extra_body", None)
    payload.pop("logit_bias", None)

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    is_stream = payload.get("stream", False)
    max_retries = 3

    async def do_request():
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            if is_stream:
                async def stream_generator():
                    for attempt in range(max_retries):
                        try:
                            async with client.stream("POST", NVIDIA_API_URL, json=payload, headers=headers) as response:
                                if response.status_code == 429:
                                    if attempt < max_retries - 1:
                                        wait_time = (attempt + 1) * 15  # 15s, 30s, 45s
                                        await asyncio.sleep(wait_time)
                                        continue
                                    else:
                                        yield f"data: {{\"error\": \"NVIDIA rate limit (429) - try again later\"}}\n\n".encode()
                                        return

                                if response.status_code != 200:
                                    yield f"data: {{\"error\": \"NVIDIA error {response.status_code}\"}}\n\n".encode()
                                    return

                                async for chunk in response.aiter_bytes():
                                    yield chunk
                                return
                        except Exception as e:
                            if attempt == max_retries - 1:
                                yield f"data: {{\"error\": \"Proxy error: {str(e)}\"}}\n\n".encode()

                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            
            else:
                for attempt in range(max_retries):
                    res = await client.post(NVIDIA_API_URL, json=payload, headers=headers)
                    
                    if res.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 15
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise HTTPException(status_code=429, detail="NVIDIA rate limit (429). Please wait a few minutes.")
                    
                    if res.status_code != 200:
                        raise HTTPException(status_code=res.status_code, detail=res.text)
                    
                    return res.json()

    return await do_request()
