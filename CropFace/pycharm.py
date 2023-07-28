import cv2
from retinaface import RetinaFace

class Face:
    def __init__(self, img):
        self.img = img
        self.faces = RetinaFace.extract_faces(img_path=self.img, align = False)
        self.detect_faces = RetinaFace.detect_faces(img)
        self.sorted_people = [] #from left
    def detect(self, show_results = True):
        if self.detect_faces is None:
            return []
        else:
            tmp = []
            for faceNum in self.detect_faces.keys():
                identity = self.detect_faces[f'{faceNum}']
                facial_area = identity["facial_area"]
                facial_img = self.img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                tmp.append([faceNum, facial_img, facial_area[0]])


                if show_results == True:
                    cv2.rectangle(self.img, (facial_area[0], facial_area[1])
                                  , (facial_area[2], facial_area[3]), (255, 255, 255), 1)
                    cv2.putText(self.img, f'{faceNum}', (facial_area[0], facial_area[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            tmp = sorted(tmp, key=lambda x: x[2])
            self.sorted_people = list(map(lambda x: x[:2], tmp))

            if show_results == True:
                return self.img

        return self.sorted_people #[faceNum, facial_img]

    def num_people(self):
        return len(self.sorted_people)
    def save_img(self, faceNum, facial_img):
        cv2.imwrite('/7multi/'+f'{faceNum}'+'.jpg', facial_img)



img = cv2.imread("E:/im/7multi.jpg")

face = Face(img)
sorted_people = face.detect(show_results=False)
for person in sorted_people:
    faceNum, facial_img = person[0], person[1]
    #face.save_img(faceNum, facial_img)
    facial_img.classify() #classify를 여기에 넣자

cv2.imshow('asdf', face.detect(show_results=True))

cv2.waitKey(0)
cv2.destroyAllWindows()




