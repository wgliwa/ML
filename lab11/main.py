import cv2
import numpy as np
import math

cap = cv2.VideoCapture("video.mp4")
frames = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_img, (15, 15), 0)
        blur_canny = cv2.Canny(gray_blur, 20, 100)
        x = 230
        tmp = np.zeros_like(blur_canny)
        blur_canny = blur_canny[x:, :]
        tmp[x:, :] = blur_canny
        dst = cv2.Canny(tmp, 50, 200, None, 3)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 300, None, 0, 0)
        if lines is not None:
            for i in lines:
                rho = i[0][0]
                theta = i[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * -b), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * -b), int(y0 - 1000 * a))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 30, 10)
        if linesP is not None:
            for i in linesP:
                l = i[0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f'Witold Gliwa {frames}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 0), 2, cv2.LINE_AA)
        cdstP = cv2.addWeighted(cdstP, 0.8, frame, 1, 0)
        cv2.imshow("das", cdstP)
        frames += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
