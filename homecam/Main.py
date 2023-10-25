from flask import Flask, render_template, request
from firebase import firebase
from registration import registration_bp
from login_bp import login_bp

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('login.html')

app.register_blueprint(login_bp) #로그인관련
app.register_blueprint(registration_bp) #회원가입부분관련 파이썬 코드




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9092)  # 포트 번호를 9091로 변경
