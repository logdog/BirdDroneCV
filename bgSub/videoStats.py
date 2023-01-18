import cv2 as cv
import numpy as np

video_path = r'E:\research\birdDroneSystem\droneVideos\flight1.mp4'

def main():

    # load the video
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print('Unable to open: ' + str(capture))
        exit(0)

    print('Video Path: ', video_path)
    print('FPS: ', capture.get(cv.CAP_PROP_FPS))

    # determine how many frames exist
    # and frame size
    numFrames = 0
    while True:
        _, frame = capture.read()
        if frame is None:
            break
        numFrames += 1
        
        if numFrames == 1:
            print('Width:  ', frame.shape[1], 'px')
            print('Height: ', frame.shape[0], 'px')
        print('Num Frames: ', numFrames,end='\r')
    capture.release()
    print()
    print(f'Runtime: {int(numFrames/(30*60)):02d}:{int((numFrames/30)%60):02d} (mm:ss)')
    print('Done!')

if __name__ == '__main__':
    main()