import cv2
from retinaface import RetinaFace
faces = RetinaFace.extract_faces(img_path = "multiperson.jpg", align = False)
cnt = 0
for face in faces:
    cv2.imwrite(f'{cnt}.jpg', cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    cnt += 1