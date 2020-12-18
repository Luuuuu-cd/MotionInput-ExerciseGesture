import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
import utils

import pyautogui
import pydirectinput
import time
import threading
import os

# Status 0 - No motion Detected
# Status 1 - Riding

status=0

# Allow some time to open the game
for i in range(3,0,-1):
    time.sleep(1)
    print (i)
print("Start!")

def forward():
    pydirectinput.keyDown('w')
    time.sleep(0.5)
    pydirectinput.keyUp('w')


def backward():
    pydirectinput.keyDown('d')
    time.sleep(0.5)
    pydirectinput.keyUp('d')

def checkStatus():
    threading.Timer(1.5, checkStatus).start()
    if(status==1):
        t = threading.Thread(target=forward)
        t.start()
        #forward()
        print ("Forward!")
    else:
        print("Not moving!")

checkStatus()

# get the lables
class_names = utils.class_names
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
# load the model onto the computation device
model = model.eval().to(device)

cap = cv2.VideoCapture("ExerciseVideo/25sExerciseBike.mp4")
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')


# a clips list to append and store the individual frames
clips = []
winname = "Test"
cv2.namedWindow(winname)
cv2.moveWindow(winname, 40, 30)
cv2.resizeWindow(winname, 450, 300)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
# read until end of video
while cap.isOpened():
    if ret == True:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        image = frame1.copy()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = utils.transform(image=frame1)['image']
        clips.append(frame1)

        if len(clips) == 8:
            with torch.no_grad():  # we do not want to backprop any gradients
                input_frames = np.array(clips)
                # add an extra dimension
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                input_frames = input_frames.to(device)
                # forward pass to get the predictions
                outputs = model(input_frames)
                # get the prediction index
                _, preds = torch.max(outputs.data, 1)

                # map predictions to the respective class names
                label = class_names[preds].strip()


            if len(contours) != 0:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if (area > 2000) and label == "ridingabike":
                    status = 1
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, "Yeah! This guy is riding!", (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                lineType=cv2.LINE_AA)
                else:
                    status = 0
                    cv2.putText(image, "No Motion Detected", (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                lineType=cv2.LINE_AA)
            else:
                status = 0
                cv2.putText(image, "No Motion Detected", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
            clips.pop(0)

        image = cv2.resize(image, (450, 300))
        cv2.imshow(winname, image)
        frame1 = frame2
        ret, frame2 = cap.read()
        if cv2.waitKey(10) == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
os._exit(1)


