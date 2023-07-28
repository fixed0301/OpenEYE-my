import cv2
from retinaface import RetinaFace
img_path = "multiperson.jpg"
faces = RetinaFace.detect_faces(img_path)
img = cv2.imread(img_path)

for faceNum in faces.keys():
    identity = faces[f'{faceNum}']
    facial_area = identity["facial_area"]
    landmarks = identity["landmarks"]

    # highlight facial area
    cv2.rectangle(img, (facial_area[2], facial_area[3])
                  , (facial_area[0], facial_area[1]), (255, 255, 255), 1)


# extract facial area

# facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

cv2.imshow('asdf', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# highlight the landmarks

# cv2.circle(img, tuple(landmarks["left_eye"]), 1, (0, 0, 255), -1)
# cv2.circle(img, tuple(landmarks["right_eye"]), 1, (0, 0, 255), -1)
# cv2.circle(img, tuple(landmarks["nose"]), 1, (0, 0, 255), -1)
# cv2.circle(img, tuple(landmarks["mouth_left"]), 1, (0, 0, 255), -1)
# cv2.circle(img, tuple(landmarks["mouth_right"]), 1, (0, 0, 255), -1)

# import matplotlib.pyplot as plt
# faces = RetinaFace.extract_faces(img_path = "img.jpg", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()