import cv2

img = cv2.imread('comp/faceimg.webp')
x, y = 100, 100
w, h = 200, 200
cut = img[x:x+h, y:y+w]
cv2.rectangle(cut,(0, 0), (w-1, h-1), (0, 255, 255), 5 )
img[x:x+h, y-w:y] = cut



cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

