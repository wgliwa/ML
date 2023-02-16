import cv2
import imutils
import numpy as np


def show(x):
    cv2.imshow(str(x), x)
    cv2.moveWindow(str(x),500,100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("im.jpg")
roi = image[200:400, 100:300]
resized = cv2.resize(image, (200, 200))
show(image)
show(roi)
show(resized)
h, w = image.shape[0:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
show(rotated)
blurred = cv2.blur(image, (20, 20))
show(blurred)
resized = imutils.resize(image, width=460)
bresized = imutils.resize(blurred, width=460)
suming = np.hstack((resized, bresized))
show(suming)
output = image.copy()
cv2.rectangle(output, (150, 200), (300, 350), (0, 0, 255), 2)
show(output)
img = np.zeros((1000, 1000, 3), np.uint8)
cv2.line(img, (600, 200), (300, 300), (255, 0, 0), 5)
points = np.array([[600, 200], [910, 641], [300, 300], [0, 0]])
cv2.polylines(img, np.int32([points]), 1, (255, 255, 255))
cv2.circle(img, (450, 250), 160, (0, 0, 255), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'witold gliwa', (10, 300), font, 3, (0, 0, 0), 5, cv2.LINE_8)
show(img)
show(image)
