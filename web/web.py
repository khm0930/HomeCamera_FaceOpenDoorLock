from flask import Flask, render_template, request, redirect, url_for,jsonify,session
import firebase_admin
from firebase_admin import credentials, auth, db
import cv2
import numpy as np
import os
from datetime import datetime




app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 비밀 키 설정


# Firebase 초기화
cred = credentials.Certificate('/home/KHM/HomeCamera_FaceOpenDoorLock/service_key/serviceAccount.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-storage-ea381-default-rtdb.firebaseio.com'
})

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register')
def register_index():
    return render_template('register.html')

@app.route('/homecam')
def homacam_index():
    return render_template('homecam.html')


@app.route('/login', methods=['POST'])
def login():
    try:
        user_data = request.get_json()  # 클라이언트에서 전송한 사용자 정보를 받음
        email = user_data['email']
        uid = user_data['uid']

        # 사용자 정보를 세션에 저장
        session['uid'] = uid
        session['email'] = email

        # 여기에서 사용자 정보를 처리하거나 필요한 동작을 수행할 수 있습니다.

        return jsonify({'success': True, 'message': '로그인 성공'})  # 성공 응답
    except Exception as e:
        return jsonify({'success': False, 'message': '로그인 실패: ' + str(e)})  # 실패 응답

# 사용자 엔드포인트
@app.route('/user')
def user_index():
    try:
        # 세션에서 사용자 정보를 가져옵니다.
        uid = session.get('uid')
        email = session.get('email')

        if uid is None:
            return jsonify({'error': '사용자 정보를 가져올 수 없습니다.'})

        # 파이어베이스에서 사용자 정보 가져오기
        user = auth.get_user(uid)

        created_timestamp = user.user_metadata.creation_timestamp / 1000  # 밀리초를 초로 변환
        created_datetime = datetime.utcfromtimestamp(created_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        user_data = {
            'uid': user.uid,
            'email': user.email,
            'name': user.display_name,  # Firebase에서 설정한 사용자 이름
            'created_at': created_datetime  # 변환된 생성 시간
        }

        return render_template('user.html', user_data=user_data)

    except Exception as e:
        return jsonify({'error': '사용자 정보를 가져올 수 없습니다.', 'message': str(e)})




    

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
    
# # 얼굴 이미지 저장을 위한 디렉토리 생성
# prototxt = '/home/KHM/HomeCamera_FaceOpenDoorLock/web/ai/deploy.proto.txt'
# model = '/home/KHM/HomeCamera_FaceOpenDoorLock/web/ai/res10_300x300_ssd_iter_140000_fp16.caffemodel'
# net = cv2.dnn.readNetFromCaffe(prototxt, model)
# os.makedirs('train', exist_ok=True)

# #얼굴저장,얼굴 학습
# @app.route('/faceid', methods=['GET', 'POST'])
# def faceid_index():
#     if request.method == 'POST':
#         face_id = request.form.get('face_id')
#         max_images = 100  # 최대 저장할 얼굴 이미지 수

#         # 이미지 처리 및 출력 루프
#         face_images = []

#         capture = cv2.VideoCapture(0)
#         capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#         capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

#         image_counter = 0  # 저장된 이미지 수를 세기 위한 카운터
#         capturing = False  # 이미지 캡처 상태

#         while len(face_images) < max_images:
#             ret, frame = capture.read()
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # DNN 모델을 통한 얼굴 감지
#             blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 240)), 1.0, (320, 240), (104.0, 177.0, 123.0))
#             net.setInput(blob)
#             detections = net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]

#                 if confidence > 0.5:
#                     box = detections[0, 0, i, 3:7] * np.array([320, 240, 320, 240])
#                     (startX, startY, endX, endY) = box.astype(int)

#                     # 감지된 얼굴 주변에 큰 사각형 표시
#                     cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)

#                     if capturing:
#                         if len(face_images) < max_images:
#                             face_image = gray[startY:endY, startX:endX]
#                             if not face_image is None and not face_image.size == 0:
#                                 # 얼굴 이미지를 저장
#                                 current_image_count = len(face_images)
#                                 image_filename = os.path.join('train', f'{face_id}.{current_image_count}.jpg')
#                                 cv2.imwrite(image_filename, face_image)
#                                 print(f'얼굴 이미지 저장: {image_filename}')
#                                 face_images.append(face_image)  # 얼굴 이미지 저장

#                     if len(face_images) >= max_images:
#                         break

#             cv2.imshow('image', frame)
#             key = cv2.waitKey(1)

#             if key > 0:
#                 if key == ord('q'):
#                     break
#                 elif key == ord('s'):
#                     # 이미지 캡처 시작 또는 종료
#                     capturing = not capturing

#         # 카메라 종료
#         capture.release()
#         cv2.destroyAllWindows()

#         # train 폴더에 저장된 얼굴 이미지를 사용하여 학습 수행
#         if len(face_images) > 0:
#             labels = np.array([0] * len(face_images))
#             recognizer = cv2.face_LBPHFaceRecognizer.create()
#             recognizer.train(face_images, labels)
#             model_file = 'trained_model.yml'
#             recognizer.save(model_file)
#             print("모델:",model_file)

#             return f'학습이 완료되었습니다. 모델이 {model_file}로 저장되었습니다.'
#         else:
#             return '학습을 위한 얼굴 이미지가 캡처되지 않았습니다.'
        
#     return render_template('choice.html')

image_directory = 'test_images'
os.makedirs(image_directory, exist_ok=True)
# 이미지를 받을 엔드포인트 및 핸들러 함수 정의
@app.route('/receive_image', methods=['POST'])
def receive_image():
    # POST 요청에서 이미지 데이터 가져오기
    image_data = request.files['image']
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    image_filename = f'received_image_{current_time}.jpg'

    # 이미지를 파일로 저장
    image_path = os.path.join(image_directory, image_filename)
    image_data.save(image_path)

    print("이미지를 받아 저장했습니다:", image_path)

    # 이미지 처리 또는 분석을 여기에서 수행할 수 있습니다.

    return '이미지를 성공적으로 받았습니다.'

@app.route('/choice')
def choice():
    return render_template('choice.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9092)