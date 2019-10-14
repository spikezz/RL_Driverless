# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:27 2019

@author: Asgard
"""

import pygame


class Mark(object):
    
    def __init__(self):
        
        pass
    
    class cross_mark(object):
        
        def __init__(self,x,y):
            
            self.x=x
            self.y=y
          
        def update(self,surface,top_down_camera):
            
            pygame.draw.line(surface,(255,255,0),(self.x-top_down_camera.x-10,self.y-top_down_camera.y),\
            (self.x-top_down_camera.x+10,self.y-top_down_camera.y),1)
            pygame.draw.line(surface,(255,255,0),(self.x-top_down_camera.x,self.y-top_down_camera.y-10),\
            (self.x-top_down_camera.x,self.y-top_down_camera.y+10),1)