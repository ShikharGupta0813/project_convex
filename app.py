from flask import Flask,request,jsonify
from model import *

app = Flask(__name__)

ans = predict("What this document is all about?")
print(ans)


# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     user_query = data["query"]
#     print(user_query)
#     ans = predict("What this document is all about?")
#     print(ans)
#     return jsonify({"ans": ans})

@app.route("/home")
def home():
    return jsonify({"hello":"utk"})


if __name__ == '__main__':
    app.run(debug=True)
