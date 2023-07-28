import cv2, dlib
import numpy as np
from imutils import face_utils

#눈만 잘라내는거 성공.. 근데 네모네모 모양임.. dlib로 눈 경계선으로 잘라내는거 해봐야함

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eye_blink_detector\shape_predictor_68_face_landmarks.dat')

def crop_eye(img, eye_points):
 
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int) #걍 사각형 대각선 점 좌표같은데

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]] #눈 부분 사진 자른거

  return eye_img, eye_rect

img = cv2.imread('E:/im/x.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes) 
#print(shapes) 뭔가 엄청 긴 [,]...들인데
eye_img_l, eye_rect_l = crop_eye(img, eye_points=shapes[36:42])
eye_img_r, eye_rect_r = crop_eye(img, eye_points=shapes[42:48]) 
cv2.imshow('asdf', eye_img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(eye_img_l, eye_rect_l)
#print(eye_img_l.shape)
#마스킹을 해보고싶었다 망한거같지만 이건 무시하자
img2 = np.zeros_like(eye_img_l)
gray = cv2.cvtColor(eye_img_l, cv2.COLOR_BGR2GRAY) #채널수가 안맞아서 여기 오류가 뜨는뎅
_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cnt, labels =  cv2.connectedComponents(th)
for i in range(cnt):
  img2[labels == i] = [int(j) for j in np.random.randint(0, 255, 3)]


