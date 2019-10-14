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
import user_interface,vehicle,camera,map_layout,mark
from pygame.locals import *

COUNT_FREQUENZ=10

UI=user_interface.User_Interface()

clock = pygame.time.Clock()

top_down_camera = camera.Camera()

vehicle_1 = vehicle.Vehicle(UI.center,[2000,1000])
vehicle_1.direction=182
vehicle_1.set_direction(False)
vehicle_1.center_camera()
vehicle_2 = vehicle.Vehicle(UI.center,[2100,1100])
vehicle_3 = vehicle.Vehicle(UI.center,[2300,1300])

vehicle_set= pygame.sprite.Group()
vehicle_set.add(vehicle_1)
vehicle_set.add(vehicle_2)
vehicle_set.add(vehicle_3)

map_layout=map_layout.Map()
map_set=map_layout.map_set

cross_mark=mark.Mark.cross_mark(0,0)
cross_mark.update(UI.screen,top_down_camera)

def event_handler():
    
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
            
            if event.unicode == 'k':  
                
                if not UI.zoom_in:
                    
                    UI.zoom_in=True
                    
                else:
                    
                    UI.zoom_in=False
                
while True:

    event_handler()
    print("tick")
    top_down_camera.set_position(vehicle_1.x, vehicle_1.y)
#    top_down_camera.set_position(vehicle_2.x, vehicle_2.y)

    UI.update(vehicle_1,vehicle_set,map_set,top_down_camera,cross_mark)
#    vehicle_1.center_camera()
#    UI.update(vehicle_2,vehicle_set,map_set,top_down_camera,cross_mark)

#    map_layout.zoom_in()

#    for m in map_set:
#        
#        if UI.zoom_in:   
#            
#            m.zoom()

#    UI.screen.blit(UI.sensor_canvas, (0,0))

#    clock.tick_busy_loop(COUNT_FREQUENZ)
    clock.tick(COUNT_FREQUENZ)

    pygame.display.update()

pygame.quit()
sys.exit(0)