from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

@app.route('/proxy', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path=''):
    target_url = request.args.get('url')
    if not target_url:
        return {"error": "Missing ?url= parameter"}, 400

    # Build the full target URL
    if path:
        full_url = f"{target_url.rstrip('/')}/{path.lstrip('/')}"
    else:
        full_url = target_url.rstrip('/')

    # Forward query parameters (except the url one)
    query_params = {k: v for k, v in request.args.items() if k != 'url'}
    if query_params:
        from urllib.parse import urlencode
        full_url += '?' + urlencode(query_params)

    # Prepare headers - forward everything useful
    headers = {}
    for key, value in request.headers:
        if key.lower() not in ['host', 'content-length']:
            headers[key] = value

    # Make sure we have Authorization if the client sent it
    if 'Authorization' not in headers and request.headers.get('Authorization'):
        headers['Authorization'] = request.headers.get('Authorization')

    try:
        # Stream the request
        resp = requests.request(
            method=request.method,
            url=full_url,
            headers=headers,
            data=request.get_data(),
            stream=True,
            timeout=300
        )

        # Build response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        response_headers = [
            (name, value) for name, value in resp.headers.items()
            if name.lower() not in excluded_headers
        ]

        def generate():
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        return Response(
            stream_with_context(generate()),
            status=resp.status_code,
            headers=response_headers
        )

    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/')
def home():
    return {"status": "JProxy-style proxy is running", "usage": "/proxy?url=https://integrate.api.nvidia.com/v1"}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
