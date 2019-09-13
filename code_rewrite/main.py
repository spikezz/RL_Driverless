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
Created on Sun Sep  8 16:43:17 2019

@author: Asgard
"""

import pygame,sys
import user_interface,vehicle,camera,map_layout
from pygame.locals import *


COUNT_FREQUENZ=500

UI=user_interface.User_Interface()

clock = pygame.time.Clock()

camera = camera.Camera()

vehicle = vehicle.Vehicle(UI.center)

vehicle_set= pygame.sprite.Group()
vehicle_set.add(vehicle)
vehicle.set_inital_direction(182)

camera.set_position(vehicle.x, vehicle.y)

map_set=map_layout.Map().map_set
mark_set=map_layout.Map().mark_set

while True:
    
    for event in pygame.event.get():
#        print("event.type",dir(pygame.event))
#        print("event",event)
    # quit for windows
        if event.type == QUIT:
                    
            pygame.quit()
            sys.exit()
            
        elif event.type == KEYDOWN :
                
            # quit for esc key
            if event.key == K_ESCAPE:  
                            
                pygame.quit()
                sys.exit()
    
    camera.set_position(vehicle.x, vehicle.y)
    
    UI.screen.blit(UI.background, (0,0))
    
    map_set.update(camera.x, camera.y)

    map_set.draw(UI.screen)
    
    mark_set.update(camera.x, camera.y)
    
    mark_set.draw(UI.screen)
    
    vehicle_set.draw(UI.screen)
    
    UI.screen.blit(UI.vehicle_canvas, (0,0))
    
    UI.screen.blit(UI.sensor_canvas, (0,0))
    
#    clock.tick_busy_loop(COUNT_FREQUENZ)
    clock.tick(COUNT_FREQUENZ)
    pygame.display.update()

pygame.quit()
sys.exit(0)