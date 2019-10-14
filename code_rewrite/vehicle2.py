# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:09:34 2019

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
        
        self.screen_center=center
        self.offset=offset
        
        self.map_block_side=1000
        
        self.x, self.y = self.findspawn(0,0)
    
        self.direction = 0.0
        
        self.maxspeed = 5.0
        self.minspeed = 0.0
        self.speed = 0.0
        
        self.acceleration = 0.2
        self.deceleration = 0.5
        self.softening = 0.04
        
        self.steering_rate = 3
        self.steering_angle=0.0
        
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
        self.rect.topleft = self.x-self.rect.center[0], self.y-self.rect.center[1]
        
    def set_direction(self,zoom):
        
        self.image, self.rect = self.rotate(self.image_copy, self.rect, self.direction)
        self.image_rect_copy_width=self.rect.width
        self.image_rect_copy_height=self.rect.height
        
    def rotate(self, image, rect, direction):
        
        rot_image = pygame.transform.rotate(image, direction)
        rot_rect = rot_image.get_rect(center=rect.center)
        
        return rot_image,rot_rect
    
    def update_self(self,cam_x,cam_y,center):
        
        self.rect.topleft = self.x - cam_x+center[0], self.y - cam_y+center[1]
    
    def update(self,cam_x,cam_y,center):
       
        self.rect.topleft = self.x - cam_x+center[0], self.y - cam_y+center[1]
