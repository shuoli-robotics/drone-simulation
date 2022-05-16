import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math


class DroneControlSim:
    def __init__(self):
        self.sim_time = 10
        self.sim_step = 0.01
        self.drone_states = np.zeros((self.sim_time/self.sim_step, 12))
        self.pointer = 1

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

    def drone_dynamics(self,T,M):
        return dx 



    def rate_controller(self,cmd):
        pass


