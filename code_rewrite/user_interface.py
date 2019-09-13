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
Created on Sun Sep  8 19:55:42 2019

@author: Asgard
"""

import pygame


class User_Interface(object):
        
    
    def __init__(self):
        
        #init
        pygame.init()
            
        #window
        self.screen = pygame.display.set_mode(size =(1360,768))
#        self.screen = pygame.display.set_mode(size =(1360,768),flags=pygame.FULLSCREEN|\
#                                              pygame.HWSURFACE|pygame.DOUBLEBUF)
        
        #title
        pygame.display.set_caption('Karat Simulation')
        
        #font
        self.font = pygame.font.Font('font/times.ttf', 40)
        
        #background layer
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert_alpha()
        self.background.fill((0, 0, 0))
               
        #canvas layer
        self.vehicle_canvas = pygame.Surface(self.screen.get_size(),flags=pygame.SRCALPHA ,depth=32)
        self.vehicle_canvas = self.vehicle_canvas.convert_alpha()
        self.vehicle_canvas.set_alpha(0)
        
        self.sensor_canvas = pygame.Surface(self.screen.get_size(),flags=pygame.SRCALPHA ,depth=32)
        self.sensor_canvas = self.sensor_canvas.convert_alpha()
        self.sensor_canvas.set_alpha(0)
        
        ##center of screen
        center_x =  float(pygame.display.Info().current_w /2)
        center_y =  float(pygame.display.Info().current_h /2)
        self.center=(center_x,center_y)
#        self.center=(0,0)
        
        
        
    