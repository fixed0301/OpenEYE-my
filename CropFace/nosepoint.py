import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret:
        h, w, c = img.shape
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            box = result.detections[0].location_data.relative_bounding_box
            x1 = round(box.xmin*w)
            x2 = round((box.xmin+box.width)*w)
            y1 = round(box.ymin*h)
            y2 = round((box.ymin+box.height)*h)
            roi_img = img[y1:y2, x1:x2]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            if not result.detections:
                continue
        cv2.imshow("asdf", img)
    else:
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()