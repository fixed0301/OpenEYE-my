import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates



# load face detection model
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)
dframe= cv2.imread("E:/im/7multi.jpg")
image_rows, image_cols, _ = dframe.shape
image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
results = mp_face.process(image_input)
for detection in results.detections:
  location = detection.location_data

  relative_bounding_box = location.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
  rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
      image_rows)


  #draw bbox
  color = (255, 0, 0)
  thickness = 2
  cv2.rectangle(image_input, rect_start_point, rect_end_point, color, thickness)
  xleft,ytop=rect_start_point
  xright,ybot=rect_end_point

crop_img = image_input[ytop: ybot, xleft: xright]


cv2.imshow('show', image_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

