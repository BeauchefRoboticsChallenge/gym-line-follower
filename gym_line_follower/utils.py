import cv2

class TrackImg:
    def __init__(self,image, ppm):
        self.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.ppm = ppm