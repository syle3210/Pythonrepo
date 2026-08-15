from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import os
from urllib.parse import urlencode

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    return {
        "status": "JProxy-style proxy is running",
        "usage": "/proxy?url=https://integrate.api.nvidia.com/v1"
    }

@app.route('/proxy', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/proxy/', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path=''):
    if request.method == 'OPTIONS':
        return '', 204

    target_url = request.args.get('url')
    if not target_url:
        return {"error": "Missing ?url= parameter"}, 400

    # Build full target URL
    if path:
        full_url = f"{target_url.rstrip('/')}/{path.lstrip('/')}"
    else:
        full_url = target_url.rstrip('/')

    # Keep extra query params (except url)
    query_params = {k: v for k, v in request.args.items() if k != 'url'}
    if query_params:
        full_url += '?' + urlencode(query_params)

    # Forward headers
    headers = {}
    for key, value in request.headers:
        key_lower = key.lower()
        if key_lower not in ['host', 'content-length', 'transfer-encoding']:
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
        response_headers = [
            (name, value) for name, value in resp.headers.items()
            if name.lower() not in excluded
        ]

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
