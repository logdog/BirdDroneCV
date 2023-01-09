import numpy as np
import cv2 as cv
from findTracks import Centroid, AerialTrack
import matplotlib.pyplot as plt

class TrackHelper:

    def __init__(self, track, track_index):
        self.track = track
        self.track_index = track_index
        self.col = np.array((387*(track_index+1)%255,287*(track_index+1)%255,187*(track_index+1)%255))/255
        self.centroids = np.array(track)
        self.xpos = self.centroids[:,0]
        self.ypos = self.centroids[:,1]

        self.velx = np.diff(self.xpos)
        self.vely = np.diff(self.ypos)

        self.accx = np.diff(self.velx)
        self.accy = np.diff(self.vely)

    def __array__(self, dtype=None):
        return self.centroids

    def scatter_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.xpos, self.ypos, sizes=[0.2], c=[self.col])
        ax.axis([0, 1920, 1080, 0])
        ax.set_aspect('equal')
        ax.set_title(f'Track {self.track_index}')
        ax.set_xlabel('X Pixel (#)')
        ax.set_ylabel('Y Pixel (#)')
        plt.show()

    def pos_vs_time(self):
        plt.plot(self.xpos,c='r')
        plt.plot(self.ypos,c='y')
        plt.title(f'Position vs Time, Track {self.track_index}')
        plt.legend(['xpos','ypos'])
        plt.xlabel('Frames since start (#)')
        plt.ylabel('Pixel (#)')
        plt.show()

    def xy_plot(self):
        plt.plot(self.xpos, self.ypos, c=self.col)
        plt.axis([0, 1920, 1080, 0])
        plt.title(f'Track {self.track_index}')
        plt.xlabel('X Pixel (#)')
        plt.ylabel('Y Pixel (#)')
        plt.show()

    def vel_vs_time(self):
        plt.plot(self.velx,c='r')
        plt.plot(self.vely,c='y')
        plt.title(f'First Difference Signal, Track {self.track_index}')
        plt.legend(['xpos','ypos'])
        plt.xlabel('Frames since start (#)')
        plt.ylabel('Pixel Difference (#/dt)')
        plt.show()

    def acc_vs_time(self):
        plt.plot(self.accx,c='r')
        plt.plot(self.accy,c='y')
        plt.title(f'2nd Difference Signal, Track {self.track_index}')
        plt.legend(['xpos','ypos'])
        plt.xlabel('Frames since start (#)')
        plt.ylabel('Pixel Difference (#/dt^2)')
        plt.show()


def main():
    tracks = np.load('tracks.npy',allow_pickle=True)
    t = [TrackHelper(track, track_index) for track_index, track in enumerate(tracks)]

    view_tracks = [0, 1, 2, 30]

    plt.plot(t[0].velx,c='r')
    plt.plot(t[1].velx,c='g')
    plt.plot(t[2].velx,c='b')
    plt.plot(t[30].velx,c='y')
    plt.axis([10,160,-10,10])
    plt.title(f'Which ones are birds? Drones?')
    plt.legend(['Track 0 (Drone)','Track 1 (Drone)','Track 2 (Bird)','Track 30 (Bird)'])
    plt.xlabel('Frames since start (#)')
    plt.ylabel('Pixel Difference (#/dt)')
    plt.show()
    
    plt.plot(t[0].vely,c='r')
    plt.plot(t[1].vely,c='g')
    plt.plot(t[2].vely,c='b')
    plt.plot(t[30].vely,c='y')
    plt.axis([10,160,-10,10])
    plt.title(f'Which ones are birds? Drones?')
    plt.legend(['Track 0 (Drone)','Track 1 (Drone)','Track 2 (Bird)','Track 30 (Bird)'])
    plt.xlabel('Frames since start (#)')
    plt.ylabel('Pixel Difference (#/dt)')
    plt.show()

if __name__ == "__main__":
    main()