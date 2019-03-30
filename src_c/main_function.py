#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:21:26 2019

@author: spikezz
"""

import numpy as np


def Calibration_Speed_Sensor(car_state):
    
    init_v=np.zeros(3)
    
    #calibration white noise of velocity
    init_v[0]=car_state.kinematics_estimated.linear_velocity.x_val
    init_v[1]=car_state.kinematics_estimated.linear_velocity.y_val
    init_v[2]=car_state.kinematics_estimated.linear_velocity.z_val
    #calibration white noise of velocity
    return init_v

def Set_Throttle(client,car_controls,data):
    
    car_controls.throttle=data.data
    client.setCarControls(car_controls)
    print("befehl:",data.data)
    
def Set_Brake(client,car_controls,data):
    
    car_controls.brake=data.data
    client.setCarControls(car_controls)
    print("befehl:",data.data)
    
def Set_Steering(client,car_controls,data):
    
    car_controls.steering=data.data
    client.setCarControls(car_controls)
    print("befehl:",data.data)