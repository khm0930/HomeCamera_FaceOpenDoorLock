import cv2

# 로드할 모델 파일의 경로
model_file = 'trained_model.yml'

# LBPH 얼굴 인식기 초기화
recognizer = cv2.face_LBPHFaceRecognizer.create()

# 모델 파일 로드
recognizer.read(model_file)

# 테스트 이미지 로드
test_image = cv2.imread('test/JIn.jpg', cv2.IMREAD_GRAYSCALE)

# 이미지를 얼굴 인식 모델에 적용
label, confidence = recognizer.predict(test_image)

# 결과 출력
if confidence < 100:
    # confidence 값이 낮을수록 정확도가 높습니다.
    print(f"예측된 레이블: {label}, 신뢰도: {confidence}, 정확도: {100 - confidence}%")
else:
    print("얼굴이 감지되지 않았거나 신뢰도가 너무 낮습니다.")
    print(confidence)
