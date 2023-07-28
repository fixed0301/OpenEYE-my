import cv2, dlib
import numpy as np
from imutils import face_utils

#경계선 검출하려고 눈에서 랜드마크 어떻게 지정되었는지 확인하기
#근데 포인트가 눈에 너무 가까이 붙어있는데. 마스크 처리할때 이 포인트로부터 조금 떨어진 지점까지를 잡아야겠다
#눈 감은 곳에는 eye_rect_cut으로 위치를 못잡았는데.. 위치를 왜 못잡았지
class eye:
    def __init__(self, img, gray, faces, detector, predictor):
        self.img = img
        self.gray = gray(img)
        self.faces = faces
        self.detector = detector
        self.predictor = predictor

    def gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLRO_BGR2GRAY) #사실 BGR인지는 모름
        return gray
    def detect_faces(self, img):
        num_faces = len(self.faces)
        return num_faces
    
    def crop_eye(self, img, eye_points): #눈 뜬거에서 잘라내자
        
        IMG_SIZE = (34, 26)

        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * IMG_SIZE[1] / IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int) #사각형 대각선 점 좌표

        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]] #눈 부분 사진 자른거

        return eye_img, eye_rect
    '''
    def eye_location(self, eye_rect_l, eye_rect_r): #눈 감은 사진에 붙일거니까 location은 감은 사진에서 찾자. eye_rect의 중심점이라고 생각해줌
        midx_l = (eye_rect_l[1] + eye_rect_l[3])/2 #눈 감은 사진에 대해 rect를 잘못잡으니까 mid 좌표도 잘못된거구나
        midy_l = (eye_rect_l[0] + eye_rect_l[2])/2 #eye point 잡은걸 사용해야겠다.
        
        midx_r = (eye_rect_r[1] + eye_rect_r[3])/2
        midy_r = (eye_rect_r[0] + eye_rect_r[2])/2 

        eye_location = (int(midx_l), int(midy_l)), (int(midx_r), int(midy_r)) 
        return eye_location
    '''
    #def put_cropped_on_closed(self, img, cropped_img, eye_location):
        #img[:, :] = cropped_img로 채우는건데.. 전경 영상 크기로 배경 영상에서 roi잘라내야함
        #img_fg = cv2.cvtColor(img)

    def find_ndraw_eye_points(self, img, draw1 = True, draw2 = True): #eye points 쩜들 찍은거 리턴
        #for face in faces: #face: [(676, 320) (943, 587)]
        for i, face in enumerate(faces):
            points = np.matrix([[p.x, p.y] for p in predictor(gray, face).parts()]) #얘는 왜 predictor지 predict랑 detect는 뭐가다르지
            left_parts = points[left_eye] #눈 point만 골라봤다
            right_parts = points[right_eye]
            for i, point in enumerate(left_parts): #show_parts = [[711 386] [726 379] [745 380] [761 391]... [859 395]] 12개
                                                #enumerate는 몇번째 점인지만 보려고
                x = point[0, 0]
                y = point[0, 1]
                if draw1 == True:
                    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
            for i, point in enumerate(right_parts):
                x = point[0, 0]
                y = point[0, 1]
                if draw2 == True:
                    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
            # cv2.imshow('d', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return left_parts, right_parts
    
    def eye_location(self, eye_parts): #find_ndraw의 return 써서 location(,)잡기
        eye_parts = eye_parts.tolist()
        w_start, w_end  = eye_parts[0][0], eye_parts[3][0]
        h_start, h_end = eye_parts[0][1], eye_parts[3][1]
        mid_w = (w_start + w_end)/2
        mid_h = (h_start + h_end)/2
        eye_loc_l = (int(mid_w), int(mid_h))
        return eye_loc_l
    
    def connect_points(self, img, left_points, right_points): #points가 show_parts랑 같은데 클래스 안 다른 메소드 쓰는 법이 뭐더라
        connected_img = img.copy()
        cv2.polylines(connected_img,[left_points], True, (0, 0, 255), 1)
        cv2.polylines(connected_img,[right_points], True, (0, 0, 255), 1)
        return connected_img
    #이제 마스킹해서 오려내는것만 남앋다 #bitwise로
    def fill_polygons(self, connected_img):
        img2 = np.zeros_like(self.img)
        gray = cv2.cvtColor(connected_img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnt, labels =  cv2.connectedComponents(th)
        for i in range(cnt):
            img2[labels == i] = [int(j) for j in np.random.randint(0, 255, 3)]
        return img2
    
    def masking(self, img, eye_img_l, eye_img_r, eye_loc_l, eye_loc_r): #눈 감은 사진에서 새로 검출한 네모로 해야겠당
        img[eye_loc_l[0]-eye_img_l/2: eye_loc_l+eye_img_l/2, eye_loc_l ] = eye_img_l
        img[eye_loc_r:] = eye_img_r
        #img[eye_rect_l[1]:eye_rect_l[3], eye_rect_l[0]:eye_rect_l[2]] = eye_img_l
        #img[eye_rect_r[1]:eye_rect_r[3], eye_rect_r[0]:eye_rect_r[2]] = eye_img_r
        
        return img

predictor_file = 'eye_blink_detector\shape_predictor_68_face_landmarks.dat'   

detector = dlib.get_frontal_face_detector() #(img, upsample_num_times)인자로 주면 faces 대해 리턴
predictor = dlib.shape_predictor(predictor_file)

right_eye = list(range(36, 42))
left_eye = list(range(42, 48))
eyes = np.arange(36, 48)

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    opened_img = img
    closed_img = cv2.imread('E:/im/x.jpg')
    gray = cv2.cvtColor(opened_img, cv2.COLOR_BGR2GRAY)
    xgray = cv2.cvtColor(closed_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    xfaces = detector(xgray, 1)

    print('number of faces detected:{}'.format(len(faces)))
        
    A = eye(opened_img, gray, faces, detector, predictor)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes) 

    for face in xfaces:
        xshapes = predictor(xgray, face)
        xshapes = face_utils.shape_to_np(xshapes) 

    eye_img_l, eye_rect_l = eye.crop_eye(eye, opened_img, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = eye.crop_eye(eye, opened_img, eye_points=shapes[42:48]) 
    #xeye_img_l, xeye_rect_l = eye.crop_eye(eye, closed_img, eye_points=xshapes[36:42])
    #xeye_img_r, xeye_rect_r = eye.crop_eye(eye, closed_img, eye_points=xshapes[42:48]) 
    #아 직사각형으로 잘라내는거로는 눈 감은 위치를 잘 모르니까 직사각형 써서 location찾으면 안된다.

    left_parts, right_parts = eye.find_ndraw_eye_points(eye, closed_img) #눈 감은거에서는 eyepoint도 제대로 안찍히네

    eye_loc_l = eye.eye_location(eye, left_parts) #eye_location은 눈 중심좌표 반환
    eye_loc_r = eye.eye_location(eye, right_parts)
    cv2.circle(closed_img, eye_loc_l, 10, (0, 255, 0), 10)

    masked = eye.masking(eye, closed_img, eye_loc_l, eye_loc_r) #closed에 opened 눈을 붙임

    #cv2.imshow('img', closed_img) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



left_points, right_points = eye.find_ndraw_eye_points(eye, img)
connected_img = A.connect_points(img, left_points, right_points)
filled_img = A.fill_polygons(connected_img)
# cv2.imshow('img', connected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




