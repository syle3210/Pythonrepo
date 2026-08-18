import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="NVIDIA NIM Proxy")

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

    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
        if is_stream:
            async def stream_generator():
                async with client.stream("POST", NVIDIA_API_URL, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        yield f"data: {{\"error\": \"NVIDIA error {response.status_code}\"}}\n\n".encode()
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            res = await client.post(NVIDIA_API_URL, json=payload, headers=headers)
            if res.status_code != 200:
                raise HTTPException(status_code=res.status_code, detail=res.text)
            return res.json()
