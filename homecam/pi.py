import cv2
import numpy as np
import picamera
import picamera.array
import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, send_file
from firebase_admin import credentials, initialize_app
from firebase_admin import storage
import os
import cv2
import numpy as np
import picamera
import picamera.array
import os
import time
import threading
import RPi.GPIO as GPIO
import signal  # 시그널 모듈을 임포트합니다.
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from uuid import uuid4
import schedule
from datetime import datetime

app = Flask(__name__)

PROJECT_ID = "fir-storage-ea381"

# Firebase 서비스 계정 키를 로드합니다.
cred = credentials.Certificate("/home/KHM/HomeCamera_FaceOpenDoorLock/Artifical Intelligence/serviceAccount.json")
firebase_app = initialize_app(cred, {
    'storageBucket': 'fir-storage-ea381.appspot.com'
})

# Firebase Storage 클라이언트를 초기화합니다. 
bucket  = storage.bucket()

def fileUpload(file):
    blob = bucket.blob('image_store/'+file) #저장한 사진을 파이어베이스 storage의 image_store라는 이름의 디렉토리에 저장
    #new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token} #access token이 필요하다.
    blob.metadata = metadata
 
    #upload file
    blob.upload_from_filename(filename='/home/KHM/HomeCamera_FaceOpenDoorLock/'+file, content_type='image/png') #파일이 저장된 주소와 이미지 형식(jpeg도 됨)
    #debugging hello
    print("저장완료 ")
    print(blob.public_url)


prev_frame = None
motion_threshold = 210000  # 움직임을 감지하기 위한 임계값 (조절 가능)
min_area = 1000  # 윤곽선을 감지하기 위한 최소 영역 크기 (조절 가능)

motion_detected = False  # 모션 감지 여부를 저장할 변수
capture_directory = 'home_captures/'  # 이미지 캡처를 저장할 디렉토리

# 이미지 캡처를 저장할 디렉토리 생성
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

last_capture_time = 0  # 마지막으로 사진을 찍은 시간을 저장하는 변수
capture_interval = 2  # 사진 찍기 간격 (초)



def detect_motion(frame):
    global prev_frame, motion_detected, last_capture_time

    # 프레임 간 차이 계산
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

        # 차이 이미지에서 움직임을 감지하는 로직 추가
        motion_detected = np.count_nonzero(gray_diff) > motion_threshold

        if motion_detected and (time.time() - last_capture_time) >= capture_interval:
            # 모션 감지된 경우, 지정된 간격 이상이 지났을 때 사진 캡처
            capture_time = datetime.now().strftime("%Y%m%d%H%M%S")  # 현재 날짜와 시간을 문자열로 포맷팅
            capture_file_path = os.path.join(capture_directory, f'{capture_time}.jpg')
            cv2.imwrite(capture_file_path, frame)
            last_capture_time = time.time()
            fileUpload(capture_file_path)

    prev_frame = frame.copy()

    return frame

def run_camera():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.rotation = 180
        raw_capture = picamera.array.PiRGBArray(camera, size=(640, 480))

        for _ in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame = raw_capture.array

            # 모션 감지 수행
            frame = detect_motion(frame)

            if motion_detected:
                # 모션 감지된 경우 중앙에 "MOTION" 문구 표시
                cv2.putText(frame, "MOTION", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # OpenCV 창에 비디오 스트림 표시
            cv2.imshow("Motion Detection", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            raw_capture.truncate(0)

    # 비디오 캡처 종료 후 OpenCV 창 닫기
    cv2.destroyAllWindows()

download_folder = '/home/KHM/HomeCamera_FaceOpenDoorLock/homecam/static/homeimage'

# 이미 다운로드한 파일들을 추적할 집합(set)을 생성
downloaded_files = set()
# 이미지 파일 경로 리스트
IMG_LIST = os.listdir("/home/KHM/HomeCamera_FaceOpenDoorLock/homecam/static/homeimage")
IMG_FOLDER = os.path.join("static", "homeimage")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
IMG_FOLDER = "homeimage"


@app.route('/')
def download_images():
    # 'train' 폴더에 있는 모든 파일 목록 가져오기
    IMG_LIST = os.listdir("/home/KHM/HomeCamera_FaceOpenDoorLock/homecam/static/homeimage")
    blobs = bucket.list_blobs(prefix='image_store/home_captures/')

    for blob in blobs:
        # 각 파일을 다운로드할 로컬 경로 설정
        file_name = os.path.basename(blob.name)

        # 이미 다운로드한 파일인지 확인
        if file_name not in downloaded_files:
            download_path = os.path.join(download_folder, file_name)
            print("다운로드 주소:",download_path)
            # 이미지 다운로드 및 로컬 저장
            blob.download_to_filename(download_path)
            downloaded_files.add(file_name)  # 다운로드한 파일을 집합에 추가

            print(f"{file_name} 이미지가 로컬 다운로드 폴더에 저장되었습니다.")
        else:
            print(f"{file_name} 이미지는 이미 다운로드되었습니다.")


        
    IMG_LIST = sorted(IMG_LIST, key=lambda x: os.path.getmtime(os.path.join(download_folder, x)), reverse=True)
    IMG_LIST = [os.path.join(IMG_FOLDER, i) for i in IMG_LIST]

    return render_template('homevisit.html',image_files=IMG_LIST)

if __name__ == '__main__':
    # 카메라 스레드 시작
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()
    app.run(host='0.0.0.0', port=9092)  # 포트 번호를 9092로 변경