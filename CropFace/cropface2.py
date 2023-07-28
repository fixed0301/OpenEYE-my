import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


#다 잘 detect 댐
# 이미지 파일의 경우 이것을 사용하세요:
IMAGE_FILES = ['multiperson.jpg']
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # 작업 전에 BGR 이미지를 RGB로 변환합니다.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	# 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('hi')
      #print(mp_face_detection.get_key_point(
          #detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)

    cv2.imshow('ann', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
