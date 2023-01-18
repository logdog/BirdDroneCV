import cv2 as cv
import numpy as np
import os

video_path = r'E:\research\birdDroneSystem\droneVideos\flight1.mp4'
frameRecording_path = r'output\frameRecording.mp4'
maskRecording_path = r'output\maskRecording.mp4'
tracker_path = r'output\trackerRecording.mp4'

splitScreenRecording = cv.VideoWriter('output/splitScreen.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))

FRAME_SKIP = 500    # skip all frames before this frame number
FRAME_START = 600   # begin to find contours at this frame number
FRAME_END = 7309    # stop analyzing the video at this frame number (use videoStats.py to determine video length)

def main():

    # load the videos
    capture_0 = cv.VideoCapture(video_path)
    capture_1 = cv.VideoCapture(frameRecording_path)
    capture_2 = cv.VideoCapture(maskRecording_path)
    capture_3 = cv.VideoCapture(tracker_path)
    captures = [capture_0, capture_1, capture_2, capture_3]

    # make sure we can read all of the videos
    for capture in captures:
        if not capture.isOpened():
            print('Unable to open: ' + str(capture))
            exit(0)

    # the original flight video will be longer, so throw out the first $FRAME_START$ frames
    for i in range(FRAME_START):
        capture_0.read()

    # now read in each frame, and create a split screen with labels
    for i in range(FRAME_START, FRAME_END+1):
        _, frame_0 =  capture_0.read()
        _, frame_1 =  capture_1.read()
        _, frame_2 =  capture_2.read()
        _, frame_3 =  capture_3.read()

        frameNum = i - FRAME_START # frameNum is relative to FRAME_START

        # resize the original video
        frame_0 = cv.resize(frame_0, (1920//2,1080//2), interpolation=cv.INTER_AREA)
        frame_1 = cv.resize(frame_1, (1920//2,1080//2), interpolation=cv.INTER_AREA)
        frame_2 = cv.resize(frame_2, (1920//2,1080//2), interpolation=cv.INTER_AREA)
        frame_3 = cv.resize(frame_3, (1920//2,1080//2), interpolation=cv.INTER_AREA)

        # create a grid
        topRow = cv.hconcat([frame_0, frame_1])
        bottomRow = cv.hconcat([frame_2, frame_3])
        frame2x2 = cv.vconcat([topRow, bottomRow])

        # add additional text to the screen
        flightTitle = os.path.basename(video_path).split(".")[0].capitalize()
        cv.putText(frame2x2, f'{frameNum}/{FRAME_END-FRAME_START-1} ({i}) {flightTitle}', 
        (0,30), cv.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        splitScreenRecording.write(frame2x2)

        # resize image
        width = int(frame2x2.shape[1] * 0.7)
        height = int(frame2x2.shape[0] * 0.7)
        dim = (width, height)
  
        resizedFrame = cv.resize(frame2x2, dim, interpolation=cv.INTER_AREA)
        cv.imshow('Frame', resizedFrame)

        keyboard = cv.waitKey(10)
        if keyboard == ord('q'):
            return

    splitScreenRecording.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()