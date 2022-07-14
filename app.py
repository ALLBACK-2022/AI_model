"""
python Flask AI model project
"""
from crypt import methods
from webbrowser import get
from flask import Flask

app = Flask(__name__)

@app.route("/hello", methods=['GET'])
def hello():
    return "hello world"

