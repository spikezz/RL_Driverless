# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:14:37 2019

@author: Asgard
"""
import os
import pygame

def load_image(image_name,transparent=True):
    
    image_path=os.path.join('image',image_name)
    image = pygame.image.load(image_path)
    
    if transparent:
        
        image = image.convert()
        colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)
        
    else:
        
        image = image.convert_alpha()
    
    