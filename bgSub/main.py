import numpy as np
import cv2 as cv

frames = []

# create background subtractor
backSub = cv.createBackgroundSubtractorMOG2()

# load video
videoPath = r'E:\research\birdDrone\droneVideos\flight1.mp4'
capture = cv.VideoCapture(videoPath)
if not capture.isOpened():
    print('Unable to open: ' + videoPath)
    exit(0)

# loop through 300 frames of the video
for i in range(1400):
    ret, frame = capture.read()
    if frame is None:
        break

    # skip the first 10 seconds
    if i < 900:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(gray, (3,3), 1)
    fgMask = backSub.apply(gray, learningRate=0.01)

    if i < 1000:
        continue

    contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    centroids = []

    for i,cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        if area >= 3:
            M = cv.moments(cnt)
            if (M['m00'] == 0):
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 1)

            # add centroids for this frame
            centroids.append((cx,cy,area,x,y,w,h))

    frames.append(centroids)
            
    cv.imshow('Frame', frame)
    #cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30) & 0xFF
    if keyboard == 'q' or keyboard == 27:
        break

np.save('frames',frames)
print(frames)