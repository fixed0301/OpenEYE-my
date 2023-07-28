import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

#모듈화됐으나 detect 낮음

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
#E:/2023/2023_1_1/comp/CropFace/multiperson.jpg
class Face:
    def __init__(self, img):
        self.img = img
        self.results = mp_face.process(img)

    def cropface(self, image_input, draw=False):
        if self.results.detections is None:
            return ['N']
        else:
            people = []
            for detection in self.results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                rect_start = _normalized_to_pixel_coordinates(
                  relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                  image_rows)
                rect_end = _normalized_to_pixel_coordinates(
                  relative_bounding_box.xmin + relative_bounding_box.width,
                  relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                  image_rows)
                xleft, ytop = rect_start
                xright, ybot = rect_end

                if draw:
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.rectangle(image_input, rect_start, rect_end, color, thickness)
                crop_img = image_input[ytop: ybot, xleft: xright]
                people.append([crop_img, xleft])
            return people

    def sort_from_left(self, people):

        return people
    def num_people(self, people):
        return len(people)

    def imshow(self, img):
        cv2.imshow('show', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#E:/2023/2023_1_1/comp/CropFace/multiperson.jpg
img = cv2.imread("E:/im/7multi.jpg")
cv2.resize(img, (1000, 1260))
image_rows, image_cols, _ = img.shape

face = Face(img)
people = face.cropface(img, draw=True)

num_people = face.num_people(people)
print(num_people)

cropped1 = people[1][0]
idx = people[0][1]


cv2.imshow('show', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#if cropped != 'N':
#    face.imshow('asdf', cropped)




