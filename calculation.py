# -*- coding: utf-8 -*-
"""
Created on Sun May  6 03:15:52 2018

@author: Asgard
"""
import math




def R_point(point,CENTER_X,CENTER_Y):
    
    R=math.sqrt(pow(point[0]-CENTER_X,2)+pow(point[1]-CENTER_Y,2))
    return R