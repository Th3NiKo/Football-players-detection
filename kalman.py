import cv2
import numpy as np

class KalmanFilter:

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                            [0,1,0,0]],np.float32)

        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)

        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32) * 0.01

    def updatePosition(self,position):
        position = np.array(position, dtype=np.float32)
        self.kalman.correct(position)
    
    def predictPosition(self):
        return self.kalman.predict()

