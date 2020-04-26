#!/usr/bin/env python
import gym
from gym_line_follower.envs import LineFollowerEnv



import numpy as np
import time

# PID gain
P=0.13
I=0.002
D=0.00001
# Sample time
Ts = 1/200

# as https://www.scilab.org/discrete-time-pid-controller-implementation trapezoidal form
class PidControl:
    """docstring for PidControl"""
    def __init__(self, P,I,D, Ts, sensorLen, vB=0.4, thresh=0.5):
        self.P = P
        self.I = I
        self.D = D
        # Aux constants
        self.b0 = (2*Ts*P + I*Ts*Ts + 4*D)/(2*Ts)
        self.b1 = (2*I*Ts - 8*D)/(2*Ts)
        self.b2 = (I*Ts*Ts - 2*Ts*P + 4*D)/(2*Ts)
        self.u1 = 0
        self.u2 = 0
        self.e1 = 0
        self.e2 = 0
        # motor
        self.vB = vB
        # sensor
        self.thresh = thresh
        self.sMax = sensorLen/2
        self.sensorPos = np.arange(sensorLen) - (sensorLen-1)/2

    def linePos(self, sensor):
        # Check if sensors are out of the line
        if np.all(sensor < self.thresh):
            # Return max value in last valid sensor position
            return self.sMax * np.sign(self.e1)
        else:
            return np.sum(self.sensorPos * sensor)

    def getAction(self, sensor):
        # Calculate PID
        e0=self.linePos(sensor)
        u0 = self.u2 + self.b0*e0 + self.b1*self.e1 + self.b2*self.e2
        # Calculate action
        velR = np.clip(self.vB - u0, -1, 1)
        velL = np.clip(self.vB + u0, -1, 1)
        action = np.array([velL,velR],dtype=np.float64)
        # Update variables
        self.u2 = self.u1
        self.u1 = u0
        self.e2 = self.e1
        self.e1 = e0
        return action

def main():

    env = LineFollowerEnv(gui=True, nb_cam_pts=4, max_track_err=0.2, power_limit=0.2,
                            max_time=1000, randomize=True,obsv_type="ir_array",sub_steps=10,
                            sim_time_step=Ts/10, track_type="robotracer")
    control=PidControl(P,I,D,Ts, sensorLen=6, vB=0.8)
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