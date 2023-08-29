from flask import Flask, request
from functions import get_answer, add_texts

app = Flask(__name__)

DEFAULT_COLLECTION_NAME = "kp_sharma_oli"

@app.route("/add", methods=["POST"])
def add():
    # TODO: handle file uploads from frontend
    add_texts(DEFAULT_COLLECTION_NAME, "./docs/test.txt")
    return {
        "success": True,
        "message": "added"
    }


@app.route("/ask", methods=["POST"])
def answer():
    body = request.json
    if not body:
        raise TypeError("empty request body")
    if "question" not in body or "history" not in body:
        raise TypeError("question or history missing in request body")
    if "collection" not in body:
        collection_name = DEFAULT_COLLECTION_NAME
    else:
        collection_name = body["collection"]
    history = [(h["question"], h["answer"]) for h in body["history"]]
    answer = get_answer(collection_name,
                        body["question"], history)
    return {
        "question": body["question"],
        "answer": answer
    }


@app.errorhandler(404)
def not_found(e):
    return "404 NOT FOUND"


@app.errorhandler(500)
def internal_server_error(e):
    return "500 INTERNAL SERVER ERROR"
