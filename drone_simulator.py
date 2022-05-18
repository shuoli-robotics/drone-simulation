import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan


class DroneControlSim:
    def __init__(self):
        self.sim_time = 10
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

    def drone_dynamics(self,T,M):
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])/self.m
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        return dx 

    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            
            self.position_cmd[self.pointer] = [1,0,-1.5]
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])

            
            # self.velocity_cmd[self.pointer] = [0.0,0.0,-1.0]
            pitch_roll_cmd,thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            self.attitude_cmd[self.pointer] = np.append(pitch_roll_cmd,psi_cmd)

            #self.attitude_cmd[self.pointer] = [1,0,0]
            self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])

            # self.rate_cmd[self.pointer] = [1,0,0]
            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m

            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.016 
        kp_q = 0.016 
        kp_r = 0.028 
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

    def attitude_controller(self,cmd):
        kp_phi = 2.5 
        kp_theta = 2.5 
        kp_psi = 2.5
        error = cmd - self.drone_states[self.pointer,6:9]
        return np.array([kp_phi*error[0],kp_theta*error[1],kp_psi*error[2]])

    def velocity_controller(self,cmd):
        kp_vx = -0.2
        kp_vy = 0.2
        kp_vz = 2

        psi = self.drone_states[self.pointer,8]
        R = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        error = R@(cmd - self.drone_states[self.pointer,3:6])
        return np.array([kp_vy*error[1],kp_vx*error[0]]),kp_vz*error[2]-self.g*self.m

    def position_controller(self,cmd):
        kp_x = 0.7 
        kp_y = 0.7 
        kp_z = 0.7 

        error = cmd - self.drone_states[self.pointer,0:3]
        return np.array([kp_x*error[0],kp_y*error[1],kp_z*error[2]])


    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()
    
