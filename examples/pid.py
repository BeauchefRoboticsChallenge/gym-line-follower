#!/usr/bin/env python
import gym
from gym_line_follower.envs import LineFollowerEnv



import numpy as np
import time


P=0.79
I=0.001
D=0.2

class PidControl(object):
    """docstring for PidControl"""
    def __init__(self, P,I,D, vB=90 ,thresh=500):
        super(PidControl, self).__init__()
        self.P = P
        self.I = I
        self.D = D
        self.lError= 0
        self.thresh=thresh
        self.vB = vB
        self.sumErr = 0

    def linePos(self, sensor):
        numSensor=0
        i=0
        error=0
        for s in sensor:
            if s < self.thresh:
                error += 10 * (i + 1)
                numSensor+=1
            i+=1

        if numSensor > 0:
            error = int(error/numSensor)
        elif self.lError > 0:
            error = 70

        self.lError = error - 35
        return self.lError

    def getAction(self, sensor):
        err=self.linePos(sensor)
        self.sumErr += self.I*err
        if abs(self.sumErr) > 20: #Antiwindup
            self.sumErr -= self.I*err
        pid = self.P * err + self.sumErr #+ self.D * dErr
        velR= self.vB - pid
        velL = self.vB + pid
        action = np.array([velL,velR],dtype=np.float32)/255
        return action

def main():

    env = LineFollowerEnv(gui=True, nb_cam_pts=4, max_track_err=0.2, power_limit=0.2,
                            max_time=1000, randomize=True,obsv_type="ir_array",sub_steps=1,
                            sim_time_step=1 / 200, track_type="robotracer")
    control=PidControl(P,I,D)
    sensor=env.reset()
    #env.render("gui")#gui/human
    start=time.time()
    done=False
    i=0
    while not done:
        action=control.getAction(sensor)
        sensor, rew, done,info = env.step(action)
        #env.render("gui")#human/gui
        i+=1
    end=time.time()
    print("======================")
    print("IT/s ",i/(end - start))
    print("======================")
    env.close()

if __name__ == '__main__':
    main()