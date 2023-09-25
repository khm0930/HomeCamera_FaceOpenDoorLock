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

PROJECT_ID = "fir-storage-ea381"

cred = credentials.Certificate("/home/KHM/HomeCamera_FaceOpenDoorLock/Artifical Intelligence/serviceAccount.json") # Service Key Path

default_app = firebase_admin.initialize_app(cred, {
    'storageBucket': f"{PROJECT_ID}.appspot.com"
})
bucket = storage.bucket()
#---------------------파이어베이스 키 접속
def fileUpload(file):
    blob = bucket.blob('image_store/'+file) #저장한 사진을 파이어베이스 storage의 image_store라는 이름의 디렉토리에 저장
    #new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token} #access token이 필요하다.
    blob.metadata = metadata
 
    #upload file
    blob.upload_from_filename(filename='/home/KHM/HomeCamera_FaceOpenDoorLock/'+file, content_type='image/png') #파일이 저장된 주소와 이미지 형식(jpeg도 됨)
    #debugging hello
    print("hello ")
    print(blob.public_url)
# GPIO 핀 번호 설정
TRIG_PIN = 14
ECHO_PIN = 15

# GPIO 핀 모드 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

prev_frame = None
min_area = 1000  # 윤곽선을 감지하기 위한 최소 영역 크기 (조절 가능)

capture_directory = 'lock_captures/'  # 이미지 캡처를 저장할 디렉토리

# 이미지 캡처를 저장할 디렉토리 생성
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

last_capture_time = 0  # 마지막으로 사진을 찍은 시간을 저장하는 변수
capture_interval = 2  # 사진 찍기 간격 (초)



# 초음파 센서로부터 거리를 측정하는 함수
def measure_distance():
    # 초음파 센서 트리거 핀을 10us 동안 활성화
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()

    # 에코 핀에서 신호가 들어올 때까지의 시간 측정
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()

    # 초음파 센서로부터 거리 계산 (음속: 343m/s)
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # 거리(cm) 계산
    return distance




def run_camera():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.rotation = 180
        raw_capture = picamera.array.PiRGBArray(camera, size=(640, 480))

        for _ in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame = raw_capture.array



            distance = measure_distance()


             # 거리가 15cm 미만이고 모션 감지된 경우 사진 캡처
            if distance < 15 :
                capture_time = cv2.getTickCount()
                capture_file_path = os.path.join(capture_directory, f'capture_{capture_time}.jpg')
                cv2.imwrite(capture_file_path, frame)  
                time.sleep(capture_interval)
                fileUpload(capture_file_path)


            # OpenCV 창에 비디오 스트림 표시
            cv2.imshow("Motion Detection", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            raw_capture.truncate(0)

    # 비디오 캡처 종료 후 OpenCV 창 닫기  
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 카메라 스레드 시작
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()

    # 웹 서버 시작
    while True:
        pass