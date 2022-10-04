import numpy as np
import cv2 as cv
from findTracks import Centroid, AerialTrack

def main():
    tracks = np.load('tracks.npy',allow_pickle=True)

    trackerRecording = cv.VideoWriter('trackerRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))
    # load video
    videoPath = r'E:\research\birdDrone\droneVideos\flight5.mp4'
    capture = cv.VideoCapture(videoPath)
    if not capture.isOpened():
        print('Unable to open: ' + videoPath)
        exit(0)

    # loop through 100 frames of the video
    for i in range(1100):
        ret, frame = capture.read()  
        if frame is None:
            break

        # skip the first 1000 frames
        if i < 30:
            continue
        
        frameNum = i - 30
        for track_id, track in enumerate(tracks):
            if centroid := track.getCentroid(frameNum):
                col = (0,255,0)
                radius = 4+int(np.sqrt(centroid.area/np.pi))
                cv.circle(frame, centroid.point, radius, col,2)
                cv.putText(frame, f'{track_id}', centroid.point+(2,0)+(radius,0), cv.FONT_HERSHEY_PLAIN, 2, col, 2)

                # draw trail (super hacky)
                for i in range(frameNum - track.startFrame - 1):
                    c0 = track.centroids[i]
                    c1 = track.centroids[i+1]
                    cv.line(frame, c0.point, c1.point, col, 2)
                    


        # resize the images 
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        dim = (width, height)
  
        # resize image
        resizedFrame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        cv.imshow('Frame', resizedFrame)
        trackerRecording.write(resizedFrame)

        keyboard = cv.waitKey(10)
        if keyboard == ord('q'):
            return

    trackerRecording.release()

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()