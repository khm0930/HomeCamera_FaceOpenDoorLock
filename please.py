
import cv2
import numpy as np
import os

# 얼굴 검출기 (classifier) 불러오기
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 비디오 캡처 설정
capture = cv2.VideoCapture(0)  # 초기화, 카메라 번호
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 30000)  # 비디오 프레임 너비 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 30000)  # 비디오 프레임 높이 설정


# 사용자로부터 이름 입력받기
face_id = input('\n사용자 이름을 입력하고 Enter 키를 누르세요 ==> ')
print("\n [INFO] 얼굴 캡처 초기화 중. 카메라를 바라보고 기다려주세요...")

# 이미지 처리 및 출력 루프
face_images = []
max_images = 50  # 최대 저장할 얼굴 이미지 수

while True:
    ret, frame = capture.read()  # 카메라 상태 및 프렘 읽기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환

    # 얼굴 감지
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(100, 100)
    )

    # 감지된 얼굴 주변에 큰 사각형 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)  # 큰 녹색 사각형 표시

        if len(face_images) < max_images:
            face_images.append(gray[y:y + h, x:x + w])  # 얼굴 이미지 저장
            # 현재 저장된 얼굴 이미지 수 출력
            current_image_count = len(face_images)
            print(f'현재 저장된 얼굴 이미지 수: {current_image_count}/{max_images}')

    cv2.imshow('image', frame)

    # 종료 조건
    key = cv2.waitKey(1)  # 키 입력 대기
    if key > 0:
        if key == ord('q') or len(face_images) >= max_images:
            break  # 'q' 키 입력 또는 최대 이미지 수 도달 시 종료

# 카메라 종료
capture.release()
cv2.destroyAllWindows()

# train 폴더에 얼굴 이미지 저장
train_folder = 'train'
os.makedirs(train_folder, exist_ok=True)
for i, face_image in enumerate(face_images):
    image_filename = os.path.join(train_folder, f'{face_id}.{i}.jpg')
    cv2.imwrite(image_filename, face_image)

# 얼굴 인식 모델 학습
if len(face_images) > 0:
    labels = np.array([0] * len(face_images))  # 모든 얼굴 샘플에 레이블 0 할당
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.train(face_images, labels)
    model_file = 'trained_model.yml'
    recognizer.save(model_file)
    print('\n [INFO] 학습이 완료되었습니다. 모델이 trained_model.yml로 저장되었습니다.')
else:
    print('\n [INFO] 학습을 위한 얼굴 이미지가 캡처되지 않았습니다.')
