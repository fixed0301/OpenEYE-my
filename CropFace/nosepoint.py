import cv2
import mediapipe as mp

# 코끝 인덱스 번호
NOSE_INDEX = 1

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=20,
)
frame = cv2.imread("E:/im/ordFace.png")
image_height, image_width, _ = frame.shape

        # 얼굴 검출
results = face_mesh.process(frame)
if results.multi_face_landmarks:
    for single_face_landmarks in results.multi_face_landmarks:
        # 코끝의 좌표값 구하기
        coordinates = single_face_landmarks.landmark[NOSE_INDEX]
        x = coordinates.x * image_width
        y = coordinates.y * image_height
        z = coordinates.z

        # x, y 좌표 화면에 그리기
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)               

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyALlWindows()