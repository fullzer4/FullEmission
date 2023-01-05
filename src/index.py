from flask import Flask
hello = Flask(__name__)

@hello.route("/")
def run():
    return "sucess"

if __name__ == "__main__":
    hello.run(host="0.0.0.0", port=int("3000"), debug=True)