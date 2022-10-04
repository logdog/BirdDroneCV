from cgitb import reset
from json.encoder import INFINITY
import numpy as np
import math
import matplotlib.pyplot as plt

import cv2 as cv

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
       return f'AT={self.centroids}'

###########################################################################################
##                                 Algorithm Pseudocode                                  ##
###########################################################################################
# for each frame
    # MERGE CENTROIDS where distance between centroids < epsilon
        # need to find distances between all centroids,
        # then eliminate the smaller centroid based on area

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
###########################################################################################
def main():
    centroidHistory = np.load('centroidHistory.npy',allow_pickle=True)
    
    # active means track was updated last frame
    stale_tracks = []
    active_tracks = []

    EPSILON = 0
    DELTA = 20

    for frameNumber, centroids_in_frame in enumerate(centroidHistory):
        
        # MERGE CENTROIDS
        # indices = set()
        # for i, centroid_i in enumerate(centroids_in_frame):
        #     for j, centroid_j in enumerate(centroids_in_frame, start=i):

        #         # check distance between points is less than epsilon
        #         if centroid_i.distanceTo(centroid_j) < EPSILON:
                    
        #             # flag the smaller contour for deletion
        #             if centroid_i < centroid_j:
        #                 indices.add(i)
        #             else:
        #                 indices.add(j)

        # # remove the smaller centroids
        # indices = sorted(list(indices), reverse=True)
        # for idx in indices:
        #     if idx < len(centroids_in_frame):
        #         centroids_in_frame.pop(idx)

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

    # for track in long_tracks:
    #     print(track)

    print(f'{len(long_tracks)} tracks found (min {min_frames} frames)')  
    np.save('tracks.npy', long_tracks)
    

if __name__ == "__main__":
    main()