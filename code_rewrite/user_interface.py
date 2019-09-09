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
        #screen = pygame.display.set_mode(size =(1360,768),flags=pygame.RESIZABLE)
        screen = pygame.display.set_mode(size =(1360,768),flags=pygame.FULLSCREEN|pygame.HWSURFACE)
        
        #title
        pygame.display.set_caption('Karat Simulation')
        
        #font
        font = pygame.font.Font('times.ttf', 40)
        
        #background layer
        background = pygame.Surface(screen.get_size())
        background = background.convert_alpha()
        background.fill((255, 1, 1))
        print(background.get_bitsize())
        
        #canvas layer
        canvas = pygame.Surface(screen.get_size(),flags=pygame.SRCALPHA ,depth=32)
        canvas = canvas.convert_alpha()
        canvas.set_alpha(0)
        
        ##center of screen
        CENTER_X =  float(pygame.display.Info().current_w /2)
        CENTER_Y =  float(pygame.display.Info().current_h /2)
        CENTER=(CENTER_X,CENTER_Y)
        
        
        
    