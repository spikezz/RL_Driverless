#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:59:09 2019

@author: spikezz
"""
import math

def calculate_sita_r(point1,point0):

    if point1[0]-point0[0] <0:

        sita=180+math.degrees(math.atan((point1[1]-point0[1])/(point1[0]-point0[0])))           
    
    elif point1[0]-point0[0] >0:
        
        sita=math.degrees(math.atan((point1[1]-point0[1])/(point1[0]-point0[0])))
        
    else:
        
        if point1[1]-point0[1]>=0:
            
            sita=90
            
        else:
            
            sita=-90

    return sita


def calculate_t(line,colour,distance,car_x,car_y):
    
    tpoint=[0,0]
    sita_l=0
    sita_t=0
    vek_l=[line[1][0]-line[0][0],line[1][1]-line[0][1]]
    
    if vek_l[0]!=0:
        
#        if vek_l[0]<0:
#            
#            sita_l=calculate_sita_r(vek_l,[0,0])
#            
#        elif vek_l[0]>0:
            
        sita_l=calculate_sita_r(vek_l,[0,0])

    else:

            
        if vek_l[1]>0:
            
            sita_l=90
        
        else:
            
            sita_l=-90
                
    if colour==1:
        
        sita_t=sita_l-90
        
    else:
        
        sita_t=sita_l+90
    
    tpoint[0]=math.cos(math.radians(sita_t))*distance+vek_l[0]+line[0][0]-car_x
    tpoint[1]=math.sin(math.radians(sita_t))*distance+vek_l[1]+line[0][1]-car_y
    
    return tpoint
