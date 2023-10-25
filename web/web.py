from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth, db

app = Flask(__name__)

# Firebase 초기화
cred = credentials.Certificate("/home/KHM/HomeCamera_FaceOpenDoorLock/Artifical Intelligence/serviceAccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-storage-ea381-default-rtdb.firebaseio.com'
})

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register')
def register_index():
    return render_template('register.html')

@app.route('/register_success', methods=['POST', 'GET'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    name = data['name']
    phoneNumber = data['phoneNumber']
    email = data['username']
    print("사용자 정보:", username, password, name, phoneNumber, email)
   

    try:
        # Firebase Authentication을 사용하여 사용자 등록
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )

        # 사용자 정보를 Firebase Realtime Database에 저장
        user_data = {
            'username': username,
            'name': name,
            'phoneNumber': phoneNumber
        }
        print("사용자 정보:", user)
        print("사용자 정보:", user_data)
        db.reference('/users/' + user.uid).update(user_data)  # update 메서드를 사용하여 업데이트

        return "회원가입 성공!"
    except Exception as e:
        print(str(e))
        return "회원가입 실패: " + str(e)



   
@app.route('/choice')
def choice():
    return render_template('choice.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9092)
