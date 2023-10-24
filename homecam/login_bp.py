
from flask import Blueprint, render_template, request
from firebase import firebase

# Firebase 초기화
firebase_app = firebase.FirebaseApplication('https://your-firebase-project.firebaseio.com', None)

login_bp = Blueprint("login", __name__)

@login_bp.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Firebase Authentication을 사용하여 로그인 시도
    # 여기서 Firebase Realtime Database에 사용자 정보를 저장하고 확인합니다.
    result = firebase_app.get('/users', None)
    if result:
        for key, user_data in result.items():
            if user_data['username'] == username and user_data['password'] == password:
                return '로그인 성공! 사용자 ID: ' + key

    return '로그인 실패' 
