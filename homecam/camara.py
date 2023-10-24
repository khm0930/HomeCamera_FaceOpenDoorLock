import cv2
import numpy as np
import picamera
import picamera.array
from flask import Flask, Response, render_template
import os
import time
import threading
from multiprocessing import Process
from firebase_admin  import credentials
from firebase_admin import storage
from uuid import uuid4
import subprocess
import firebase_admin
from multiprocessing import Process

# 
PROJECT_ID = "fir-storage-ea381" # Owner Project ID
camera_process = None  # 카메라 프로세스를 저장할 변수

cred = credentials.Certificate("/home/KHM/HomeCamera_FaceOpenDoorLock/Artifical Intelligence/serviceAccount.json") # Service Key Path

default_app = firebase_admin.initialize_app(cred, {
    'storageBucket': f"{PROJECT_ID}.appspot.com"
})
bucket = storage.bucket()
#---------------------파이어베이스 키 접속
app = Flask(__name__)



prev_frame = None
motion_threshold = 220000  # 움직임을 감지하기 위한 임계값 (조절 가능)
min_area = 1000  # 윤곽선을 감지하기 위한 최소 영역 크기 (조절 가능)

motion_detected = False  # 모션 감지 여부를 저장할 변수
capture_directory = 'home_captures/'  # 이미지 캡처를 저장할 디렉토리
last_capture_time = 0  # 마지막으로 사진을 찍은 시간을 저장하는 변수
capture_interval = 4  # 사진 찍기 간격 (초)
# 이미지 캡처를 저장할 디렉토리 생성
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

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
            capture_time = cv2.getTickCount()
            capture_file_path = os.path.join(capture_directory, f'capture_{capture_time}.jpg')
            cv2.imwrite(capture_file_path, frame)
            last_capture_time = time.time()

            # 사진을 Firebase Storage에 업로드
            fileUpload(capture_file_path)
    prev_frame = frame.copy()

    return frame

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

def generate_frames():
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

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            raw_capture.truncate(0)

def generate_frames1():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)  # 해상도 설정
        camera.framerate = 30
        camera.rotation = 180
        raw_capture = picamera.array.PiRGBArray(camera, size=(640, 480))
         
        for _ in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame = raw_capture.array
            frame = detect_motion(frame)
            print("1")
            
            # raw_capture 객체를 정리하여 카메라 리소스를 해제
            raw_capture.truncate(0)

def start_camera_process():
    global camera_process
    if camera_process is None or not camera_process.is_alive():
        stop_camera_process()  # 먼저 카메라 프로세스를 중지
        camera_process = Process(target=generate_frames())
        camera_process.start()

def start_camera_process1():
    global camera_process
    if camera_process is None or not camera_process.is_alive():
        stop_camera_process()  # 먼저 카메라 프로세스를 중지
        camera_process = Process(target=generate_frames1())
        camera_process.start()

def stop_camera_process():
    global camera_process
    if camera_process is not None and camera_process.is_alive():
        camera_process.terminate()
        camera_process.join()
        time.sleep(2)  # 충분한 대기 시간 추가
        
def stop_camera_process1():
    global camera_process
    if camera_process is not None and camera_process.is_alive():
        camera_process.terminate()
        camera_process.join()
        time.sleep(2)  # 충분한 대기 시간 추가





if __name__ == '__main__':
    camera_process = Process(target=generate_frames1)
    camera_process.start()