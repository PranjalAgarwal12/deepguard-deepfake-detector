from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
import requests, os

app = Flask(__name__)
CORS(app)

FRONTEND = os.path.join(os.path.dirname(__file__), 'frontend')
BACKEND  = 'http://127.0.0.1:8000'

@app.route('/')
def index():
    return send_from_directory(FRONTEND, 'index.html')

@app.route('/api/v1/<path:path>', methods=['GET','POST','OPTIONS'])
def proxy(path):
    url  = f'{BACKEND}/api/v1/{path}'
    resp = requests.request(method=request.method, url=url,
           data=request.get_data(), headers={'Content-Type': request.content_type or ''})
    return Response(resp.content, resp.status_code,
           {'Access-Control-Allow-Origin': '*',
            'Content-Type': resp.headers.get('Content-Type','application/json')})

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(FRONTEND, path)

app.run(port=3000, debug=False)
