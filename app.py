from flask import Flask, request, jsonify
from Worker import WorkerNode
import socket

app = Flask(__name__)

# each container = one worker
worker = WorkerNode(socket.gethostname())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    task = {
        "task_id": data.get("id", 0),
        "user_query": data.get("query", "")
    }

    result = worker.processTask(task)

    return jsonify(result)

app.run(host="0.0.0.0", port=5000)