import cv2 as cv
import numpy as np

class Centroid:
    def __init__(self, x, y, area):
        self.point = np.array([x,y])
        self.area = area

    def distanceTo(self, other):
        return np.linalg.norm(self.point - other.point)

    def __lt__(self, other):
        return self.area  < other.area

    def __repr__(self):
        return f'c=({self.point[0]}, {self.point[1]}, {self.area})'

    def __array__(self, dtype=None):
        return np.array([self.point[0], self.point[1], self.area])

    def __getitem__(self, index):
        return self.__array__()[index]

class AerialTrack:
    def __init__(self, centroid, frameNum):
        self.centroids = [centroid]
        self.startFrame = frameNum
        self.lifetime = 1

    def update(self, centroid):
        self.centroids.append(centroid)
        self.lifetime += 1

    # return centroid at a particular frame
    def getCentroid(self, frameNum):
        if self.activeOn(frameNum):
            return self.centroids[frameNum - self.startFrame]
        return None

    # check if AerialTrack is active on a particular frame
    def activeOn(self, frameNum):
        return frameNum >= self.startFrame and frameNum < self.startFrame + self.lifetime

    # we can only update if last update occurred on previous frame
    def canUpdate(self, frameNum):
        return frameNum == self.startFrame + self.lifetime

    def distanceTo(self, centroid):
        return self.centroids[-1].distanceTo(centroid)
    
    def __repr__(self):
       return f'AT=(start: {self.startFrame}, Lifetime: {self.lifetime})'
       # return f'AT={self.centroids}'

    def __getitem__(self, index):
        return self.centroids[index]

    def __array__(self, dtype=None):
        # create a 2D array, where each row is a centroid (converted to an array)
        return np.array([
            np.array(x) for x in self.centroids
        ])

video_path = r'E:\research\birdDroneSystem\droneVideos\flight1.mp4'

FRAME_SKIP = 400    # skip all frames before this frame number
FRAME_START = 500   # begin to find contours at this frame number
FRAME_END = 3000    # stop analyzing the video at this frame number

def get_centroids():
    centroidHistory = []
    MIN_RADIUS = 1.5

    # save videos to curent folder
    # frameRecording = cv.VideoWriter('output/frameRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))
    # maskRecording = cv.VideoWriter('output/maskRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))

    # create background subtractor
    backSub = cv.createBackgroundSubtractorMOG2()

    # load video
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print('Unable to open: ' + video_path)
        exit(0)

    # loop through frames of the video
    for i in range(FRAME_END):
        _, frame = capture.read()
        if frame is None:
            break

        # skip the first few frames
        print(i)
        if i < FRAME_SKIP:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 1)
        fgMask = backSub.apply(blur, learningRate=0.006)
        _, fgMask = cv.threshold(fgMask, 1, 255, cv.THRESH_BINARY)

        # train on 30 frame (1 second of data)
        if i < FRAME_START:
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
        
        # ========== OUTPUT VIDEO =============== #
        # cv.putText(frame, f'{i}', (0,20), cv.FONT_HERSHEY_PLAIN, 1.4, (255,255,0), 2)

        # resize the images 
        # width = int(frame.shape[1] * 0.5)
        # height = int(frame.shape[0] * 0.5)
        # dim = (width, height)
  
        # resize image
        # resizedFrame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        # resizedMask = cv.resize(fgMask, dim, interpolation=cv.INTER_AREA)

        # convert black and white to a "color" image to save to video
        # resizedMask = cv.merge([resizedMask,resizedMask,resizedMask])

        # cv.imshow('Frame', resizedFrame)
        # cv.imshow('FG Mask', resizedMask)

        # frameRecording.write(resizedFrame)
        # maskRecording.write(resizedMask)
        
        #keyboard = cv.waitKey(0)
        # keyboard = cv.waitKey(5)
        # if keyboard == ord('q'):
        #     return

    # frameRecording.release()
    # maskRecording.release()
    return centroidHistory

def find_tracks(centroid_history):
    print('find_tracks()')

# for each frame
    # UPDATE AerialTracks
        # for each remaining centroid in this frame
            # loop over AerialTracks that can update (last update was previous frame)
            # create list of AerialTracks within some radius < delta
            # if no objects < delta, create a new AerialTrack
        # begin to update AerialTracks, starting with shortest distance first
        # AerialTracks can only be updated once per frame
        # continue until the list is empty or every AerialTrack has been updated
        # the remaining centroids are new AerialTracks
        # Note: depth search only goes back 1 frame
    
    # active means track was updated last frame
    stale_tracks = []
    active_tracks = []

    DELTA = 20

    for frameNumber, centroids_in_frame in enumerate(centroid_history):
        print(frameNumber)
        
        # UPDATE CENTROIDS
        # print(f'\nFrame Number = {frameNumber}')
        new_tracks = []
        all_candidate_tracks = []

        updated_track_ids = []
        updated_centroid_ids = []

        for centroid_index, centroid in enumerate(centroids_in_frame):

            # create list of AerialTracks within some distance < delta
            candidate_tracks = []  # (centroid index, aerial object index, distance)
            
            # loop over AerialTracks that are still active
            for track_index, track in enumerate(active_tracks):
                distance = track.distanceTo(centroid)
                if distance < DELTA:
                    candidate_tracks.append((centroid_index, track_index, distance))

            # if no objects < delta, create a new AerialTrack
            if not candidate_tracks:
                new_tracks.append(AerialTrack(centroid, frameNumber))
                updated_centroid_ids.append(centroid_index)
            else:
                # add candidate parents to list
                for c in candidate_tracks:
                    all_candidate_tracks.append(c)

        all_candidate_tracks.sort(key=lambda x: x[2]) # sort by radius, smallest first

        # print(f'active tracks: {active_tracks}')
        # print(f'new_tracks: {new_tracks}')
        # print(f'all_candidate_tracks: {all_candidate_tracks}')

        # update active AerialTracks, starting with the shortest radius first
        while all_candidate_tracks:
            centroid_index, track_index, distance = all_candidate_tracks[0]
            active_tracks[track_index].update(centroids_in_frame[centroid_index])
            updated_track_ids.append(track_index)
            updated_centroid_ids.append(centroid_index)

            # print(f'\tcentroid_index: {centroid_index}, track_index: {track_index}')

            # can no longer update "active_tracks[track_index]"
            # can no longer use "centroids_in_frame[centroid_index]"
            all_candidate_tracks = [ct for ct in all_candidate_tracks if centroid_index != ct[0] and track_index != ct[1]]
            # print(f'\tall_candidate_tracks: {all_candidate_tracks}')

        if len(centroids_in_frame) > len(updated_centroid_ids):
            for centroid_index, centroid in enumerate(centroids_in_frame):
                if centroid_index not in updated_centroid_ids:
                    new_tracks.append(AerialTrack(centroid, frameNumber))
                    updated_centroid_ids.append(centroid_index)
        
        if len(centroids_in_frame) > len(updated_centroid_ids):
            print('\tlen(centroids_in_frame) > len(updated_centroid_ids): ' + f'{len(centroids_in_frame)} > {len(updated_centroid_ids)}')

        # copy, then remove stale tracks
        # print(f'Updated {len(updated_track_ids)} tracks')
        for track_index in range(len(active_tracks)):
            if track_index not in updated_track_ids:
                stale_tracks.append(active_tracks[track_index])
                active_tracks[track_index] = None
                # print(f'\tremoving track {track_index}...')

        # remove the flagged tracks
        active_tracks = [track for track in active_tracks if track is not None]

        # save the new tracks for the next frame
        for track in new_tracks:
            active_tracks.append(track)

        # print(f'Finished processing frame {frameNumber}')
        # print(f'\tactive tracks: {active_tracks}')
        # print(f'\tstale tracks: {stale_tracks}')
        #key = input() # pause

    # END UPDATE CENTROIDS
    # add the active tracks to all tracks
    for track in active_tracks:
        stale_tracks.append(track)

    # only care about aerial objects that appear for at least x tracks
    min_frames = 100
    long_tracks = list(filter(lambda o: o.lifetime >= min_frames, stale_tracks))
    long_tracks.sort(key=lambda o: o.lifetime, reverse=True)
    print(f'{len(long_tracks)} tracks found (min {min_frames} frames)')

    return long_tracks

# overlay the tracks on the original video
def show_tracks(tracks):
    print('show_tracks()')

    #trackerRecording = cv.VideoWriter('output/trackerRecording.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (960,540))
    trackerRecordingFS = cv.VideoWriter('output/trackerRecordingFS.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, (1920,1080))
    
    # load video
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print('Unable to open: ' + video_path)
        exit(0)

    # loop through frames of the video
    for i in range(FRAME_END):
        print(i)
        _, frame = capture.read()  
        if frame is None:
            break

        # skip the first few frames
        if i < FRAME_START:
            continue
        
        frameNum = i - FRAME_START # frameNum is relative to FRAME_START
        for track_id, track in enumerate(tracks):
            if centroid := track.getCentroid(frameNum):
                col = (187*(track_id+1)%255,287*(track_id+1)%255,387*(track_id+1)%255) # color
                radius = 4+int(np.sqrt(centroid.area/np.pi))
                cv.circle(frame, centroid.point, radius, col,2)
                cv.putText(frame, f'{track_id}', centroid.point+(2,0)+(radius,0), cv.FONT_HERSHEY_PLAIN, 2, col, 2)

                # draw trail (super hacky)
                for i in range(frameNum - track.startFrame - 1):
                    if i+1 >= len(track.centroids):
                        break
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

        #trackerRecording.write(resizedFrame)
        trackerRecordingFS.write(frame)

        keyboard = cv.waitKey(10)
        if keyboard == ord('q'):
            return

    #trackerRecording.release()
    trackerRecordingFS.release()
    cv.destroyAllWindows()

def main():
    centroids = get_centroids()
    long_tracks = find_tracks(centroids)
    
    input('Press Enter to show video')
    show_tracks(long_tracks)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()