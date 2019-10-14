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
    
    def __init__(self,center,offset):
        
        pygame.sprite.Sprite.__init__(self)
        
        self.image = load_image('test_map\golf.png')
        self.rect = self.image.get_rect()
        self.image_copy = load_image('test_map\golf.png')#save original pic
        self.rect_copy = self.image_copy.get_rect()
#        self.image_rect_copy_width=self.rect.width
#        self.image_rect_copy_height=self.rect.height
        self.map_block_side=1000
        self.screen_center=center
        self.offset=offset
        self.x, self.y = self.findspawn(0,0)#absolute coordinate
        self.direction = 0.0
        self.speed = 0.0
        self.maxspeed = 5.0
        self.minspeed = 0.0
        self.acceleration = 0.2
        self.deceleration = 0.5
        self.softening = 0.04
        self.steering_rate = 3
#        self.tracks = False
        self.steering_angle=0.0
#        self.rrl=0.0
#        self.rrr=0.0
#        self.rpl=(0,0)
#        self.rpr=(0,0)
#        self.rrm=0
        
    def findspawn(self,map_grid_x,map_grid_y):
        
        self.map_grid_x=map_grid_x#row
        self.map_grid_y=map_grid_y#line
        self.offset_x=self.offset[0]
        self.offset_y=self.offset[1]
        spawn_x=self.map_grid_x * self.map_block_side +self.offset_x
        spawn_y=self.map_grid_y * self.map_block_side +self.offset_y
    
        return spawn_x,spawn_y
    
    def center_camera(self):
        
        self.x_canvas = self.screen_center[0]
        self.y_canvas = self.screen_center[1]
        self.rect.topleft = self.x_canvas-self.rect.center[0], self.y_canvas-self.rect.center[1]
        print(self.x_canvas,self.rect.center[0],self.y_canvas,self.rect.center[1])
        print(self.rect.topleft)
    
    def rotate(self, image, rect, direction):
        
#        rotate an image while keeping its center
        rot_image = pygame.transform.rotate(image, direction)
        rot_rect = rot_image.get_rect(center=rect.center)
        
        return rot_image,rot_rect
    
    def scale(self,image,rect,w,h):
        
        scaled_image=pygame.transform.scale(image,(w,h))
        scaled_rect=scaled_image.get_rect(center=rect.center)
        
        return scaled_image,scaled_rect
        
    def set_direction(self,zoom):
        
        #self.dir is the direction of the car, car.dir=0 means face top,the positive direction is anticlockwise
        self.image, self.rect = self.rotate(self.image_copy, self.rect_copy, self.direction)
        
#        if not zoom:
#            
        self.image_rect_copy_width=self.rect.width
        self.image_rect_copy_height=self.rect.height
    
    def zoom(self):
        
        self.image, self.rect = self.scale(self.image,self.rect, int(self.image_rect_copy_width/2),\
                                           int(self.image_rect_copy_height/2))
        
    def update_self(self,cam_x,cam_y,center):
        
        self.rect.center = self.x - cam_x+center[0], self.y - cam_y+center[1] 
    
    def update(self,cam_x,cam_y,center):

        self.rect.center = self.x - cam_x+center[0], self.y - cam_y+center[1]

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