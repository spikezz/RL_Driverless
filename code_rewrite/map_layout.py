# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:10:15 2019

@author: Asgard
"""
import pygame
from tools import load_image

class Map(object):
    
    def __init__(self):
        
        self.map_tile_path=['test_map\intersection.png',
                            'test_map\intersection.png'
                           ]
        self.map_image_set = []
        self.map_plan=[\
                        [1]\
                      ]
        self.side=1000
        self.map_set= pygame.sprite.Group()
        
        for tile_idx in range (0,len(self.map_tile_path)):
        
            self.map_image_set.append(load_image(self.map_tile_path[tile_idx],False))
        
        for x in range (0,1):
            
            for y in range (0,1):
                
                self.temp_tile=self.Map_Tile(self, self.map_plan[x][y], x * self.side, y * self.side)
                self.map_set.add(self.temp_tile)
    
    def zoom_in(self):
        
        self.map_set= pygame.sprite.Group()
        
        for x in range (0,1):
            
            for y in range (0,1):
                
                self.temp_tile=self.Map_Tile(self, self.map_plan[x][y], x * self.side/2, y * self.side/2)
                self.temp_tile.zoom()
                self.map_set.add(self.temp_tile)
        
    class Map_Tile(pygame.sprite.Sprite):
    
        def __init__(self,whole_map, tile, y, x):
            
            pygame.sprite.Sprite.__init__(self)
            self.image = whole_map.map_image_set[tile]
            self.rect = self.image.get_rect()
            self.image_rect_copy_width=self.rect.width
            self.image_rect_copy_height=self.rect.height
            self.x = x
            self.y = y
            
        def update(self, cam_x, cam_y,center):
            
            self.rect.topleft = self.x - cam_x+center[0], self.y - cam_y+center[1]
            
        def scale(self,image,rect,w,h):
        
            scaled_image=pygame.transform.smoothscale(image,(w,h))
            scaled_rect=scaled_image.get_rect(center=rect.center)
            
            return scaled_image,scaled_rect
        
        def zoom(self):
        
            self.image, self.rect = self.scale(self.image,self.rect, int(self.image_rect_copy_width/2),\
                                               int(self.image_rect_copy_height/2))