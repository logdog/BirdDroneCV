from json.encoder import INFINITY
import numpy as np
import math
import matplotlib.pyplot as plt

class AerialObject:
    def __init__(self,x,y,area,frameNum):
        self.xs = [x]
        self.ys = [y]
        self.areas = [area]
        self.startFrame = frameNum
        self.lifetime = 1

    def update(self,x,y,area):
        self.xs.append(x)
        self.ys.append(y)
        self.areas.append(area)
        self.lifetime += 1

    def radiusTo(self,x,y):
        return math.sqrt((x-self.xs[-1])**2 + (y-self.ys[-1])**2)

    def __repr__(self):
        return f'Start Frame: {self.startFrame}, Lifetime: {self.lifetime}, xs: {self.xs}, ys: {self.ys}'

frames = np.load('frames.npy',allow_pickle=True)
aerial_objects = []
MIN_RADIUS = 10

for frame_i, frame in enumerate(frames):
    for cx,cy,area,x,y,w,h in frame:

        # find closest object
        smallest_radius = math.inf
        smallest_radius_i = -1
        for i,obj in enumerate(aerial_objects):
            r = obj.radiusTo(cx,cy)
            if r < smallest_radius:
                smallest_radius = r
                smallest_radius_i = i

        if smallest_radius > MIN_RADIUS:
            obj = AerialObject(cx,cy,area,frame_i)
            aerial_objects.append(obj)
        else:
            aerial_objects[smallest_radius_i].update(cx,cy,area)

# only care about aerial objects that appear for at least 30 frames (1 second)
aerial_objects = filter(lambda o: o.lifetime > 30, aerial_objects)
for x in aerial_objects:
    print(x)
    

