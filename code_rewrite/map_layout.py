# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:10:15 2019

@author: Asgard
"""
import pygame
from tools import load_image

class Map(object):
    
    def __init__(self):
        
        self.map_tile_path=['test_map\intersection_scaled_1000.png',
                            'test_map\intersection_scaled.png'
                           ]
        self.map_image_set = []
        self.map_plan=[\
                        [1]\
                      ]
        self.side=1000
        self.map_set= pygame.sprite.Group()
        
        ##read the maps and draw
        for tile_idx in range (0,len(self.map_tile_path)):
        
            #add submap idx to array
            self.map_image_set.append(load_image(self.map_tile_path[tile_idx],False))
            #add submap idx to array

        for x in range (0,1):
    
            for y in range (0,1):
                
                temp_tile=self.Map_Tile(self, self.map_plan[x][y], x * self.side, y * self.side)
                print(temp_tile.rect.topleft)
                self.map_set.add(temp_tile)
#                #add submap to mapgroup
#                ##read the maps and draw
        
        self.mark_set= pygame.sprite.Group()
        self.mark_set.add(self.Land_Mark(x * self.side, y * self.side))
        
    class Land_Mark(pygame.sprite.Sprite):
        
        def __init__(self,x,y):
            
            pygame.sprite.Sprite.__init__(self)
            self.image=load_image('test_map\mark.png',False)
            self.rect = self.image.get_rect()
            self.x = x
            self.y = y
        
        def update(self, cam_x, cam_y):
            
            self.rect.topleft = self.x - cam_x, self.y - cam_y
            
    class Map_Tile(pygame.sprite.Sprite):
    
        def __init__(self,whole_map, tile, y, x):
            
            pygame.sprite.Sprite.__init__(self)
            self.image = whole_map.map_image_set[tile]
            self.rect = self.image.get_rect()
            self.x = x
            self.y = y
            
        def update(self, cam_x, cam_y):
            
            self.rect.topleft = self.x - cam_x, self.y - cam_y