import os
import numpy as np
import pandas as pd
import cv2

PATH = 'C:\\Users\\gbour\\Desktop\\sysvision\\train_1'

PATH_PORCINE_1 = os.path.join(PATH, 'Porcine')
PATH_PORCINE_DISSECTION_1 = os.path.join(PATH_PORCINE_1, 'Dissection')

PATH_VR_1 = os.path.join(PATH, 'VR')
PATH_VR_DISSECTION_1 = os.path.join(PATH_VR_1, 'Dissection')

print(os.path.join(PATH_VR_DISSECTION_1, 'DS_0001.mp4'), os.path.isfile(os.path.join(PATH_VR_DISSECTION_1, 'DS_0001.mp4')))

cap = cv2.VideoCapture(os.path.join(PATH_VR_DISSECTION_1, 'DS_0001.mp4'))

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    if frame is not None:
        #cv2.imshow('frame',gray)
        print(frame.shape) # -> (540, 960, 3)
        cv2.imshow('frame', cv2.resize(frame[60:490, 165:795], (256, 256)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
