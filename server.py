from flask import Flask, request, url_for
from concurrent.futures import ThreadPoolExecutor
import json
app = Flask(__name__)
executor = ThreadPoolExecutor(10000)
def singleThreadRecieve(number):
    return number
@app.route('/', methods=['POST', 'GET'])
def receive():
    executor.submit(singleThreadRecieve, json.loads(request.data))
    return json.loads(request.data)
if __name__=='__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 5000)

