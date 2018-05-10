# -*- coding: utf-8 -*-
"""
Created on Sun May  6 03:15:52 2018

@author: Asgard
"""
import math




def calculate_r(point,CENTER):
    
    R=math.sqrt(pow(point[0]-CENTER[0],2)+pow(point[1]-CENTER[1],2))
    
    return R


def calculate_sita(symbol180,point,CENTER):
    
    if symbol180==1:
        
        sita=180+math.degrees(math.atan((point[0]-CENTER[0])/(point[1]-CENTER[1])))
        
    else:
        
        sita=math.degrees(math.atan((point[0]-CENTER[0])/(point[1]-CENTER[1])))
        
    return sita


def calculate_rotated_subpoint(CENTER,radius,angle,symbol):
    
    if symbol==1:
        
        point=(CENTER[0]+radius*math.sin(math.radians(angle)),CENTER[1]+radius*math.cos(math.radians(angle)))
    
    else:
        
        point=(CENTER[0]-radius*math.sin(math.radians(angle)),CENTER[1]-radius*math.cos(math.radians(angle)))
    
    return point
    
    
def calculate_rotated_point(CENTER,direction,R,sita):
    
    point=(CENTER[0]+R*math.cos(math.radians(270-direction-sita)),CENTER[1]+R*math.sin(math.radians(270-direction-sita)))
    
    return point















