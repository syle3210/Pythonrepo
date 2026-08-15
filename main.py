from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import os
from urllib.parse import urlencode, urlparse

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return {
        "status": "JProxy-style proxy is running",
        "usage": "Use: /proxy?url=https://integrate.api.nvidia.com/v1"
    }

@app.route('/proxy', methods=['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE'])
@app.route('/proxy/', methods=['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE'])
@app.route('/proxy/<path:subpath>', methods=['GET', 'POST', 'OPTIONS', 'PUT', 'DELETE'])
def proxy(subpath=None):
    if request.method == 'OPTIONS':
        return Response('', status=204)

    target = request.args.get('url')
    if not target:
        return {"error": "Missing ?url= parameter"}, 400

    # Clean target
    target = target.rstrip('/')

    # If Janitor or the client appended a path after /proxy, add it
    if subpath:
        full_url = f"{target}/{subpath.lstrip('/')}"
    else:
        full_url = target

    # Preserve other query parameters
    extra_params = {k: v for k, v in request.args.items() if k != 'url'}
    if extra_params:
        full_url += ('&' if '?' in full_url else '?') + urlencode(extra_params)

    # Forward headers
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

        def generate():
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        return Response(
            stream_with_context(generate()),
            status=resp.status_code,
            headers=response_headers
        )

    except Exception as e:
        return {"error": str(e)}, 500


# Catch-all so we can see what path is actually being requested
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    return {
        "error": "Route not found",
        "requested_path": path,
        "method": request.method,
        "full_url": request.url,
        "hint": "You should use /proxy?url=https://integrate.api.nvidia.com/v1"
    }, 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
