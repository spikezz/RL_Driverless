#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:57:10 2019

@author: spikezz
"""
import airsim
from geometry_msgs.msg import Point

class cone(Point):
    
    def __init__(self,color,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        client = airsim.CarClient()
        client.confirmConnection()
#        client.enableApiControl(True)
        if color==-1:
            
            print(client.simSpawnObject('/Game/cone_yellow.cone_yellow_C', airsim.Pose(position_val=airsim.Vector3r(self.x,self.y,self.z))))
        
        else:
            
            print(client.simSpawnObject('/Game/cone_blue.cone_blue_C', airsim.Pose(position_val=airsim.Vector3r(self.x,self.y,self.z))))
        