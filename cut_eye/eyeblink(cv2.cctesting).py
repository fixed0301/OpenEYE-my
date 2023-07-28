import cv2 
import os
#cv2의 cascasdeclassifier로 했는데 얘는 위치만 잡는거고 눈 경계선 따려면 dlib를 써야해서 이건 버리기

face_cascade = cv2.CascadeClassifier('comp/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('comp/haarcascade_eye.xml')

img = cv2.imread('E:/im/o.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class Cuteye:
    def __init__(self, img):
        self.img = img

    def find_faces(self):
        faces = face_cascade.detectMultiScale(gray, )

    def cut(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray,1.31,2)
        cut_parts = [] 
        for (x,y,w,h) in eyes: 
            cutpart = self.img[x:x+w, y:y+h]
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
            cut_parts.append(cutpart)
        return cut_parts
    def cutout(self, img, cut_parts):
        for i in cut_parts:
            img[i]
            i = []
        return img[]
    def show(self, img):
        cv2.imshow("wow", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cuteye = Cuteye(img)
cut_parts = cuteye.cut()
print(len(cut_parts)) 
   
for cut_img in cut_parts:
    cuteye.show(cut_img)
# 얼굴과 눈을 인식하기 위한 xml 파일을 읽음

# 이미지를 읽고 gray scale 로 변경


'''
# 이미지에서 얼굴 인식
faces = face_cascade.detectMultiScale(gray, 1.3, 1)
print(faces)
# 검출된 얼굴 개수만큼 for 루프 동작
for (x,y,w,h) in faces:

    # 원본 이미지의 얼굴 위치에 사각형 그리기
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # 눈 위치 인식은 검출된 얼굴 영역 내부에서만 진행
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # 눈 위치 인식
    eyes = eye_cascade.detectMultiScale(roi_gray,1.31,2) 

    # 검출된 눈 개수만큼 for 루프 동작
    for (ex,ey,ew,eh) in eyes: 

        # 원본 이미지에 사각형으로 눈 위치 표시
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    
# 결과를 화면에 표시함
cv2.imshow('modified',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''