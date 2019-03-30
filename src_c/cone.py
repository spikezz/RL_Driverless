#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:57:10 2019

@author: spikezz
"""
import airsim
import path_m
from geometry_msgs.msg import Point
import calculate as cal
#
#def auto_spawn(auto_spawn,half_path_wide,delta_path,car_state):
#    
#    if auto_spawn==True:
#        
#        ##constant of path
#        half_path_wide=4
#        delta_path=5
#        ##constant of path
#        #create the Group contains track mittle point
#        list_path_point=[]
#        #create the Group contains track mittle point
#        #create the Group contains yellow cone
#        list_yellow_cone=[]
#        #create the Group contains yellow cone
#        #create the Group contains blue cone
#        list_cone_blue=[]
#        #create the Group contains blue cone
#        
#        #start point
#        startpoint=path_m.path_m(0,0,-5)
#        last_point=[startpoint.x,startpoint.y]
#        #start point
#        
#        cone_x, cone_y = startpoint.x-half_path_wide,startpoint.y
#        print("cone_x:",cone_x)
#        print("cone_y:",cone_y)
#        
#        cone_new=cone(1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
#        
#        list_yellow_cone.append(cone_new)
#        
#        print("cone new:",cone_new)
#        
#        
#        cone_x, cone_y = startpoint.x+half_path_wide,startpoint.y
#        print("cone_x:",cone_x)
#        print("cone_y:",cone_y)
#        
#        cone_new=cone(-1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
#        
#        list_yellow_cone.append(cone_new)
#        
#        print("cone new:",cone_new)
#        
#        corner=[]
#        corner.append(14)
#        
#        print("startpoint:",startpoint)
#        
#        for t in range (1,corner[0]):
#            print("corner:",t)
#            path_new= path_m.path_m(startpoint.x,startpoint.y+delta_path*t,-5)
#            list_path_point.append(path_new)
#            print("path new:",path_new)
#        
#        
#        for pa in list_path_point:
#            print("pa",pa)
#            line=[[last_point[0],last_point[1]],[pa.x,pa.y]]
#            print("line:",line)
#             
#            cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide)
#            print("cone_x:",cone_x)
#            print("cone_y:",cone_y)
#            cone_new=cone(1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
#            
#            list_yellow_cone.append(cone_new)
#            
#            print("cone new:",cone_new)
#        
#            cone_x, cone_y = cal.calculate_t(line,1,half_path_wide)
#            print("cone_x:",cone_x)
#            print("cone_y:",cone_y)
#            cone_new=cone(-1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
#            
#            list_cone_blue.append(cone_new)
#            
#            print("cone new:",cone_new)
#        
#            last_point=[pa.x,pa.y]

        
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
              