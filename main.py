from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import os
import re
from urllib.parse import urljoin, urlencode

app = Flask(__name__)
CORS(app)

# Patterns that cause the <channel|> spam
CHANNEL_PATTERNS = [
    re.compile(r'<\|channel\|>?', re.IGNORECASE),
    re.compile(r'<channel\|>', re.IGNORECASE),
    re.compile(r'<\|channel>thought\n?', re.IGNORECASE),
    re.compile(r'thought\n?<channel\|>', re.IGNORECASE),
]

def clean_channel_tokens(text: str) -> str:
    if not text:
        return text
    for pattern in CHANNEL_PATTERNS:
        text = pattern.sub('', text)
    # Collapse repeated leftover fragments
    text = re.sub(r'(channel\|?>?){2,}', '', text, flags=re.IGNORECASE)
    return text

@app.route('/', methods=['GET'])
def home():
    return {
        "status": "JProxy-style proxy is running (channel tokens stripped)",
        "usage": "/proxy?url=https://integrate.api.nvidia.com/v1"
    }

@app.route('/proxy', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/proxy/', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/proxy/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(subpath=None):
    if request.method == 'OPTIONS':
        return Response('', status=204)

    target = request.args.get('url')
    if not target:
        return {"error": "Missing ?url= parameter"}, 400

    target = target.strip()
    if not target.startswith('http'):
        target = 'https://' + target
    target = target.rstrip('/') + '/'

    if subpath:
        path = subpath.lstrip('/')
    else:
        path = 'chat/completions'

    full_url = urljoin(target, path)

    extra = {k: v for k, v in request.args.items() if k != 'url'}
    if extra:
        full_url += ('&' if '?' in full_url else '?') + urlencode(extra)

    headers = {}
    for key, value in request.headers:
        kl = key.lower()
        if kl not in ['host', 'content-length', 'transfer-encoding', 'connection']:
            headers[key] = value

    try:
        resp = requests.request(
            method=request.method,
            url=full_url,
            headers=headers,
            data=request.get_data(),
            stream=True,
            timeout=300,
            allow_redirects=False
        )

        excluded = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]

        content_type = resp.headers.get('Content-Type', '')

        def generate():
            buffer = ''
            for chunk in resp.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                text = chunk.decode('utf-8', errors='ignore')
                buffer += text

                # Clean the buffer
                cleaned = clean_channel_tokens(buffer)

                # Keep a small trailing buffer in case a tag is split across chunks
                if len(cleaned) > 40:
                    to_send = cleaned[:-20]
                    buffer = cleaned[-20:]
                    if to_send:
                        yield to_send.encode('utf-8')
                else:
                    buffer = cleaned

            # Flush remaining
            if buffer:
                yield clean_channel_tokens(buffer).encode('utf-8')

        return Response(
            stream_with_context(generate()),
            status=resp.status_code,
            headers=response_headers
        )

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
