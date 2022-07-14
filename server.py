"""
python Flask AI model project
"""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "hello world Hi Wow"

app.run(port=5000, debug=True)
print('test')