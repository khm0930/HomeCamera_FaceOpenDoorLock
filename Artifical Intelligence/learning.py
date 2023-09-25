import cv2
import os
import numpy as np

# 얼굴 인식기 생성
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 학습 데이터 경로 설정
data_path = 'train'

def get_images_and_labels(data_path):
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    face_samples = []
    labels = []

    label_dict = {}  # 라벨을 정수로 매핑하기 위한 딕셔너리

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = os.path.splitext(os.path.basename(image_path))[0].split('.')[0]  # 파일 이름에서 "kyung" 추출

        # 라벨을 정수로 매핑
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        label_id = label_dict[label]

        face_samples.append(img)
        labels.append(label_id)  # 매핑된 정수 라벨 사용

    return face_samples, labels

print('\n [INFO] 얼굴 학습 중입니다. 잠시 기다려 주십시오...')
faces, labels = get_images_and_labels(data_path)

# 얼굴 인식 모델 생성 및 학습
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.train(faces, np.array(labels))

print('\n [INFO] 학습이 완료되었습니다.')

# 테스트 이미지 폴더 경로 설정
test_image_folder = 'test'

# 테스트 이미지 파일 목록 가져오기
test_image_files = [os.path.join(test_image_folder, f) for f in os.listdir(test_image_folder) if os.path.isfile(os.path.join(test_image_folder, f))]
# 임계값 설정
confidence_threshold = 50

for test_image_file in test_image_files:
    test_image = cv2.imread(test_image_file, cv2.IMREAD_GRAYSCALE)

    # 파일명에서 파일명만 추출
    file_name = os.path.basename(test_image_file)
    user_name = file_name.split('.')[0]  # 파일명에서 "."을 기준으로 앞부분 추출

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(test_image[y:y+h, x:x+w])

        if confidence < confidence_threshold:
            accuracy = f'{100 - confidence:.2f}%'
            user_name_display = f' {user_name}'  # 정확도가 낮을 때도 사용자 이름을 표시
        else:
            accuracy = f'{100 - confidence:.2f}%'
            user_name_display = 'unknown'

        print(f' 사용자: {user_name_display}, {accuracy}')  # 파일명과 사용자 이름 출력 추가
        print(confidence)
        print(confidence_threshold)
        # 얼굴 주위에 테두리 그리기
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 사용자 이름 및 정확도 표시
        cv2.putText(test_image, f'{user_name_display}: {accuracy}', (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow('Test Image', test_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os

# # 얼굴 인식기 생성
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # 사용자 이름 및 ID 설정
# user_names = {0: '미확인', 1: 'kyungjune', 2: 'hyunmin'}

# # 학습 데이터 경로 설정
# data_path = 'dataset'

# def get_images_and_labels(data_path):
#     image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
#     face_samples = []
#     ids = []

#     for image_path in image_paths:
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         id = int(os.path.split(image_path)[-1].split(".")[1])  # 파일 이름에서 ID 추출
#         face_samples.append(img)
#         ids.append(id)

#     return face_samples, ids

# print('\n [INFO] 얼굴 학습 중입니다. 잠시 기다려 주십시오...')
# faces, ids = get_images_and_labels(data_path)

# recognizer = cv2.face_LBPHFaceRecognizer.create()
# recognizer.train(faces, np.array(ids))
# recognizer.write('trainer/trainer.yml')
# print('\n [INFO] 학습완료')
# recognizer.read('trainer/trainer.yml')
# # 임계값 설정
# confidence_threshold = 70  # 임계값 설정

# # 테스트 이미지 폴더 경로 설정
# test_image_folder = 'test'

# # 테스트 이미지 파일 목록 가져오기
# test_image_files = [os.path.join(test_image_folder, f) for f in os.listdir(test_image_folder) if os.path.isfile(os.path.join(test_image_folder, f))]

# for test_image_file in test_image_files:
#     test_image = cv2.imread(test_image_file, cv2.IMREAD_GRAYSCALE)

#     # 파일 이름에서 ID 추출
#     test_id = int(os.path.split(test_image_file)[-1].split(".")[1])

#     # 얼굴 검출
#     faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # 파일 이름의 ID와 일치하는 경우에만 인식 결과 출력
#     if test_id in user_names:
#         for (x, y, w, h) in faces:
#             face_id, confidence = recognizer.predict(test_image[y:y+h, x:x+w])

#             if confidence < confidence_threshold:
#                 user_name = user_names.get(face_id, '미확인')
#                 accuracy = f'정확도: {100 - confidence:.2f}%'
#             else:
#                 user_name = '미확인'
#                 accuracy = '인식 실패'

#             # 얼굴 주위에 테두리 그리기
#             cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             # 사용자 이름 및 정확도 표시
#             cv2.putText(test_image, f'{user_name} ({accuracy})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     else:
#         # 파일 이름의 ID와 일치하지 않는 경우 "미확인"으로 처리
#         for (x, y, w, h) in faces:
#             cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(test_image, 'unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow('Test Image', test_image)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import os

# # 얼굴 인식기와 LBPHFaceRecognizer 생성
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# recognizer = cv2.face_LBPHFaceRecognizer.create()

# # 사용자 이름 및 ID 설정
# user_names = {0: '미확인', 1: 'kyungjune',2: 'hyunmin'}

# # 학습 데이터 경로 설정
# data_path = 'dataset'

# def get_images_and_labels(data_path):
#     image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
#     face_samples = []
#     ids = []

#     for image_path in image_paths:
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         id = int(os.path.split(image_path)[-1].split(".")[1])  # 파일 이름에서 ID 추출
#         face_samples.append(img)
#         ids.append(id)

#     return face_samples, ids

# print('\n [INFO] 얼굴 학습 중입니다. 잠시 기다려 주십시오...')
# faces, ids = get_images_and_labels(data_path)

# recognizer.train(faces, np.array(ids))

# # 웹캠을 통한 얼굴 인식
# video_capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

#         if confidence < 100:
#             user_name = user_names.get(face_id, '미확인')
#             accuracy = f'정확도: {100 - confidence:.2f}%'
#             print(f'사용자: {user_name}, {accuracy}')
#         else:
#             user_name = '미확인'
#             accuracy = '인식 실패'

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f'{user_name} ({accuracy})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()
