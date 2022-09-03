import cv2
import os
import shutil
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


import cv2
import numpy as np
cap = cv2.VideoCapture('shoot.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    if not ret:
        break
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
