#The MIT License (MIT)

#Copyright (c) 2012 Robin Duda, (chilimannen)

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:55:09 2019

@author: Asgard
"""

import pygame
from tools import load_image


class Vehicle(pygame.sprite.Sprite):
    
    def __init__(self,center):
        
        pygame.sprite.Sprite.__init__(self)
        
        self.image = load_image('n19.png')
        self.rect = self.image.get_rect()
        self.image_copy = self.image
#        self.screen = pygame.display.get_surface()
#        self.area = self.screen.get_rect()
        self.x_canvas = center[0]
        self.y_canvas = center[1]
#        self.x_canvas = 0
#        self.y_canvas = 0
        self.rect.topleft = self.x_canvas-17, self.y_canvas-33
        self.map_block_side=1000
#        self.x, self.y = self.findspawn(center,12,4)
        self.x, self.y = self.findspawn(center,0,0)
        print(self.x, self.y)
        self.direction = 0.0
        self.speed = 0.0
        self.maxspeed = 5.0
        self.minspeed = 0.0
        self.acceleration = 0.2
        self.deceleration = 0.5
        self.softening = 0.04
        self.steering_rate = 3
        self.tracks = False
        self.steering_angle=0.0
#        self.rrl=0.0
#        self.rrr=0.0
#        self.rpl=(0,0)
#        self.rpr=(0,0)
#        self.rrm=0
        
    def findspawn(self,center,map_grid_x,map_grid_y):
        
        self.map_grid_x=map_grid_x#row
        self.map_grid_y=map_grid_y#line
#        self.offset_x=-100
#        self.offset_y=-250
        self.offset_x=-680-330
        self.offset_y=-384-280
#        self.offset_x=-680
#        self.offset_y=-384
#        self.offset_x=0
#        self.offset_y=0
        spawn_x=self.map_grid_x * self.map_block_side + center[0]+self.offset_x
        spawn_y=self.map_grid_y * self.map_block_side + center[1]+self.offset_y
    
        return spawn_x,spawn_y
    
    def rotate(self, image, rect, direction):
        
#        rotate an image while keeping its center
        rot_image = pygame.transform.rotate(image, direction)
        rot_rect = rot_image.get_rect(center=rect.center)
        
        return rot_image,rot_rect
    
    def set_inital_direction(self,direction):
        
        #self.dir is the direction of the car, car.dir=0 means face top,the positive direction is anticlockwise
        self.direction=direction
        self.image, self.rect = self.rotate(self.image_copy, self.rect, self.direction)
        
#    def update(self, last_x, last_y):
#        
#        if self.wheelangle>0:
#            x_old=self.x
#            y_old=self.y
#            self.rrm=math.sqrt(pow(x_old-self.rpl[0],2)+pow(y_old-self.rpl[1],2))#pixel/s
#            
#            if   x_old>self.rpl[0]:
#                
#                fi=math.asin((y_old-self.rpl[1])/self.rrm)
#                
#            elif x_old<=self.rpl[0]:  
#                
#                fi=math.pi-math.asin((y_old-self.rpl[1])/self.rrm)
#                
#            
#            fin=fi-(self.speed/self.rrl)
#            
#            self.x=self.rpl[0]+math.cos(fin)*self.rrm
#            self.y=self.rpl[1]+math.sin(fin)*self.rrm
#            
#        elif self.wheelangle<0:
#            
#            x_old=self.x
#            y_old=self.y
#            self.rrm=math.sqrt(pow(x_old-self.rpr[0],2)+pow(y_old-self.rpr[1],2))#pixel/s
#            
#            if   x_old>self.rpr[0]:
#                
#                fi=math.asin((y_old-self.rpr[1])/self.rrm)
#                
#            elif x_old<=self.rpr[0]:  
#                
#                fi=math.pi-math.asin((y_old-self.rpr[1])/self.rrm)
#                
#            
#            fin=fi+(self.speed/self.rrr)
#            
#            self.x=self.rpr[0]+math.cos(fin)*self.rrm
#            self.y=self.rpr[1]+math.sin(fin)*self.rrm
#            
#        else:  
#    
#            self.x = self.x + self.speed * math.cos(math.radians(270-self.dir))
#            self.y = self.y + self.speed * math.sin(math.radians(270-self.dir))
#            self.reset_tracks()
##        