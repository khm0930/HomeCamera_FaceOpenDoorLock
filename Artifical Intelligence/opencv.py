import cv2
import pyrebase
import os

# Firebase 구성 정보
config = {
  "apiKey": "AIzaSyDpXmvo_nceTWz8YZh2wVWQjVHsaxmu6wc",
  "authDomain": "fir-storage-ea381.firebaseapp.com",
  "projectId": "fir-storage-ea381",
  "storageBucket": "fir-storage-ea381.appspot.com",
  "messagingSenderId": "1065706594226",
  "appId": "1:1065706594226:web:1e5f0738e12cc735fbceab",
  "measurementId": "G-2R77WYCCSP",
  "serviceAccount": "serviceAccount.json",
  "databaseURL": "https://fir-storage-ea381-default-rtdb.firebaseio.com"
}

# Firebase 초기화
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# 얼굴 감지기 (분류기) 로드
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 비디오 캡처 설정
capture = cv2.VideoCapture(0)  # 초기화, 카메라 번호.
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 비디오 프레임 너비 설정.
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 비디오 프레임 높이 설정.

# 콘솔 메시지
face_id = input('\n사용자 이름을 입력하고 Enter 키를 누르세요 ==> ')
print("\n [INFO] 얼굴 캡처 초기화 중. 카메라를 보고 기다려주세요...")
count = 0  # 캡처된 얼굴 이미지 수

# 이미지 처리 및 출력 루프
while True:
    ret, frame = capture.read()  # 카메라 상태 및 프레임 읽기.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환.

    # 얼굴 감지
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # 감지된 얼굴 주위에 큰 사각형 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)  # 큰 녹색 사각형 표시
        count += 1
        image_filename = f"{face_id}.{count}.jpg"
        image_path = os.path.join("train", image_filename)
        cv2.imwrite(image_path, gray[y:y+h, x:x+w])  # 얼굴 이미지 저장

        # 이미지를 Firebase Storage에 업로드
        destination_path = f"train/{image_filename}"
        storage.child(destination_path).put(image_path)
        print(f"{image_filename}이(가) {destination_path}로 업로드되었습니다.")

    cv2.imshow('image', frame)

    # 종료 조건
    key = cv2.waitKey(1)  # 키 입력 대기.
    if key > 0:
        break  # 키 입력이 있을 때 루프 종료
    elif count >= 100:
        break  # 100개의 얼굴 샘플을 모두 캡처했을 때 종료.

print("\n [INFO] 프로그램을 종료하고 정리합니다.")

capture.release()  # 카메라 메모리 해제.
cv2.destroyAllWindows()









# 모든 윈도우 창을 닫습니다.
# from flask import Flask, render_template, request, jsonify
# import cv2
# import base64
# import os

# app = Flask(__name__)

# # 얼굴 검출기(classifier)를 로드합니다.
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # 비디오 캡처 설정
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # 콘솔 메시지
# face_id = None
# count = 0

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/dataset/capture', methods=['POST'])
# def capture_image():
#     global face_id
#     global count
#     imageData = request.get_data()  # get_data() 메서드로 데이터 읽기

#     if face_id is None:
#         face_id = request.form.get('face_id')

#     if count < 100:
#         # 이미지 데이터를 파일로 저장합니다.
#         count += 1
#         image_path = f'dataset/User.{face_id}.{count}.jpg'
#         with open(image_path, 'wb') as file:
#             file.write(imageData)

#     return jsonify(success=True)


# if __name__ == '__main__':
#     # 'dataset' 폴더를 정적 파일 경로에 추가
#     app.static_folder = os.path.join(os.getcwd(), 'dataset')
#     app.run(debug=True)
