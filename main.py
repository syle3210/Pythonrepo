from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import os
from urllib.parse import urljoin, urlencode

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return {
        "status": "JProxy-style proxy is running",
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

    # Make sure target has proper scheme
    target = target.strip()
    if not target.startswith('http'):
        target = 'https://' + target

    target = target.rstrip('/') + '/'

    # Decide the final path
    if subpath:
        path = subpath.lstrip('/')
    else:
        path = 'chat/completions'

    # Properly join the URL (this fixes the https:/ bug)
    full_url = urljoin(target, path)

    # Add extra query parameters if any
    extra = {k: v for k, v in request.args.items() if k != 'url'}
    if extra:
        full_url += ('&' if '?' in full_url else '?') + urlencode(extra)

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
