# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 03:12:08 2019

@author: Asgard
"""

class Camera(object):
    
    def __init__(self):
        
        self.x = 0
        self.y = 0
    
    def set_position(self, x, y):
        
        self.x = x
        self.y = y
        