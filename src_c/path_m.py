#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:54:58 2019

@author: spikezz
"""
#import rospy
import airsim
from geometry_msgs.msg import Point


class path_m(Point):
    
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        client = airsim.VehicleClient()
        client.confirmConnection()
#        client.enableApiControl(True)       
        print(client.simSpawnObject('/Game/path_point.path_point_C', airsim.Pose(position_val=airsim.Vector3r(self.x,self.y,self.z))))
        