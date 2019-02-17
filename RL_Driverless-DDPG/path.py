# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:40:55 2018

@author: Asgard
"""
import pygame, maps
from pygame.locals import *
from loader import load_image


FULL_TILE = 1000
# =============================================================================
# CENTER_X =  800
# CENTER_Y =  450
# 
# =============================================================================
CENTER_X =  680
CENTER_Y =  384

class path(pygame.sprite.Sprite):
    
    def __init__(self,x,y,cam_x,cam_y):
        pygame.sprite.Sprite.__init__(self)

        self.image = load_image('path.png', False)
        self.rect = self.image.get_rect()
        self.screen = pygame.display.get_surface()
        self.x = cam_x+x
        self.y = cam_y+y
        self.rect.center = self.x, self.y
        



    def update(self, cam_x, cam_y):
        
        self.rect.center = self.x - cam_x, self.y - cam_y
        

        
    