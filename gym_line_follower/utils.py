import cv2
import numpy as np

class TrackRefImg:
    def __init__(self,image, ppm):
        # Transofrm the track texture to an inverse reflectance image for irsensor simulation
        self.img = np.array(255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        self.ppm = ppm