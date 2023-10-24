from flask import Flask, render_template, request
from firebase import firebase


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('visit.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091)  # 포트 번호를 9091로 변경