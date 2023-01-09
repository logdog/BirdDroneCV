from cgitb import small
import numpy as np
import cv2 as cv

# local file imports
from findTracks import Centroid

MIN_RADIUS = 1.5

def main():

    centroidHistory = []

    # save videos to curent folder
    frameRecording = cv.VideoWriter('output/frameRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))
    maskRecording = cv.VideoWriter('output/maskRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))

    # create background subtractor
    backSub = cv.createBackgroundSubtractorMOG2()

    # load video
    videoPath = r'E:\research\birdDroneSystem\droneVideos\flight1.mp4'
    capture = cv.VideoCapture(videoPath)
    if not capture.isOpened():
        print('Unable to open: ' + videoPath)
        exit(0)

    # loop through 60 frames of the video
    for i in range(3000):
        _, frame = capture.read()
        if frame is None:
            break

        # skip the first few frames
        print(i)
        if i < 400:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 1)
        fgMask = backSub.apply(blur, learningRate=0.006)
        _, fgMask = cv.threshold(fgMask, 1, 255, cv.THRESH_BINARY)

        # train on 30 frame (1 second of data)
        if i < 500:
            continue

        # once the background subtractor has been trained, start processessing the images
        contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        centroids = []

        smaller_i = 0

        # if the camera moves a little bit, thousands of contours are drawn
        if len(contours) > 3000:
            print('too many')
            continue

        for cnt in contours:
            _, r = cv.minEnclosingCircle(cnt)

            if r >= MIN_RADIUS:
                smaller_i += 1
                M = cv.moments(cnt)
                if (M['m00'] == 0):
                    continue

                # calculate centroid
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                cv.circle(frame, (cx,cy), int(6+r), (0,0,255),3)
                cv.putText(frame, f'{smaller_i}', (cx,cy), cv.FONT_HERSHEY_PLAIN, 1.4, (0,255,0), 2)

                # add centroids for this frame
                centroid = Centroid(cx,cy,np.pi*r**2)
                centroids.append(centroid)

        centroidHistory.append(centroids)
        cv.putText(frame, f'{i}', (0,20), cv.FONT_HERSHEY_PLAIN, 1.4, (255,255,0), 2)

        # resize the images 
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        dim = (width, height)
  
        # resize image
        resizedFrame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        resizedMask = cv.resize(fgMask, dim, interpolation=cv.INTER_AREA)

        # convert black and white to a "color" image to save to video
        resizedMask = cv.merge([resizedMask,resizedMask,resizedMask])

        cv.imshow('Frame', resizedFrame)
        cv.imshow('FG Mask', resizedMask)

        frameRecording.write(resizedFrame)
        maskRecording.write(resizedMask)
        
        #keyboard = cv.waitKey(0)
        keyboard = cv.waitKey(5)
        if keyboard == ord('q'):
            return

    centroidHistory = np.asanyarray(centroidHistory, dtype=object)
    np.save('output/centroidHistory', centroidHistory)

    frameRecording.release()
    maskRecording.release()


if __name__ == "__main__":
    main()
    cv.destroyAllWindows()