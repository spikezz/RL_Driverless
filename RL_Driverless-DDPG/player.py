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

#Camera module will keep track of sprite offset.

#Player module, the car.
import os, sys, pygame, math, maps
from pygame.locals import *
from random import randint
from loader import load_image

GRASS_SPEED = 0.715
GRASS_GREEN = 75
CENTER_X = -1
CENTER_Y = -1
FULL_TILE = 1000

#Rotate car.
def rot_center(image, rect, angle):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image,rot_rect

def findspawn():

    #while(maps.map_1[y][x] == 5):
            #x = randint(0,9)
            #y = randint(0,9)
    #return x * FULL_TILE, y * FULL_TILE 
     #x = randint(0,9)
     #y = randint(0,6)
     
     x=12#row
     y=4#line
# =============================================================================
#      fx=-100
#      fy=-320
#      
# =============================================================================
     fx=-100
     fy=-250
#     #while(maps.map_1[y][x] == 5):
#             #x = randint(0,9)
#             #y = randint(0,9)
     return x * FULL_TILE + CENTER_X+fx, y * FULL_TILE + CENTER_Y+fy

#define car as Player.
class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image('f1.png')
        
        self.rect = self.image.get_rect()
        self.image_orig = self.image
        self.screen = pygame.display.get_surface()
        self.area = self.screen.get_rect()
        CENTER_X =  int(pygame.display.Info().current_w /2)
        CENTER_Y =  int(pygame.display.Info().current_h /2)
        self.x = CENTER_X
        self.y = CENTER_Y
        self.rect.topleft = self.x-17, self.y-33#middle of the car
        self.x, self.y = findspawn()
        self.dir = 0
        self.speed = 0.0
        self.maxspeed = 5
        self.minspeed = 0
        self.acceleration = 0.5
        self.deacceleration = 1.5
        self.softening = 0.04
        self.steering = 1.60
        self.tracks = False
        self.wheelangle=0.0
        self.rrl=0.0
        self.rrr=0.0
        self.rpl=(0,0)
        self.rpr=(0,0)
        self.rrm=0
        
        
    def set_start_direction(self,direction):
        
        #self.dir is the direction of the car, car.dir=0 means face top,the positive direction is anticlockwise
        self.dir=direction
        self.image, self.rect = rot_center(self.image_orig, self.rect, self.dir)
    def reset(self):
        self.image = load_image('f1.png')
        self.rect = self.image.get_rect()
        self.area = self.screen.get_rect()
        CENTER_X =  int(pygame.display.Info().current_w /2)
        CENTER_Y =  int(pygame.display.Info().current_h /2)
        self.x = CENTER_X
        self.y = CENTER_Y
        self.rect.topleft = self.x-17, self.y-33#middle of the car
        self.x, self.y = findspawn()
        self.x=self.x-220
        self.y=self.y-15
        self.dir = 0
        
        self.speed = 0.0
        self.tracks = False
        self.wheelangle=0.0
        self.rrl=0.0
        self.rrr=0.0
        self.rpl=(0,0)
        self.rpr=(0,0)
        self.rrm=0
        
# =============================================================================
# #Reset the car.
#     def reset(self):
#         self.x =  int(pygame.display.Info().current_w /2)
#         self.y =  int(pygame.display.Info().current_h /2)
#         self.speed = 0.0
#         self.dir = 0
#         self.image, self.rect = rot_center(self.image_orig, self.rect, self.dir)
#         self.rect.topleft = self.x, self.y
#         self.x, self.y = findspawn()
#             
# #Emit tracks..
#     def emit_tracks(self):
#         self.tracks = True
#         
# =============================================================================
#Don't emit tracks..
    def reset_tracks(self):
        self.tracks = False
# =============================================================================
# 
# #If the car is on grass, decrease speed and emit tracks.
#     def grass(self, value):
#         if value > GRASS_GREEN:
#             if self.speed - self.deacceleration > GRASS_SPEED * 2:
#                 self.speed = self.speed - self.deacceleration * 2
#                 self.emit_tracks()
# 
# #Push back on impact
#     def impact(self):
#         if self.speed > 0:
#             self.speed = self.minspeed
# =============================================================================

    def soften(self):
        
        
        if self.speed > 0.04 :
            self.speed -= self.softening
        elif self.speed < -0.04:
            self.speed += self.softening
        else:
            self.speed=0
                

#Accelerate the vehicle
    def accelerate(self,a):
        if self.speed <=self.maxspeed and self.speed >=self.minspeed and self.speed + a>=self.minspeed and self.speed + a<=self.maxspeed:
            self.speed = self.speed + a
        elif self.speed + a<=self.minspeed and self.speed >=self.minspeed:
            self.speed = self.minspeed
        elif self.speed + a>=self.maxspeed and self.speed <=self.maxspeed:
            self.speed = self.maxspeed
# =============================================================================
#             if self.speed < self.maxspeed / 3:
#                 self.emit_tracks()
# =============================================================================

#Deaccelerate.
    def deaccelerate(self):
        if (self.speed >= self.minspeed) and (self.speed - self.deacceleration>self.minspeed):
            self.speed = self.speed - self.deacceleration
        else:
            self.speed = self.minspeed
            #self.emit_tracks()

#Steer.
    def steerleft(self):
    #def steerleft(self):    
        self.dir =self.dir+math.degrees(self.speed/self.rrl)
        #self.dir = self.dir+self.steering
        if self.dir > 360:
            self.dir = self.dir-360
# =============================================================================
#         if (self.speed > self.maxspeed / 2):
#             self.emit_tracks()
# =============================================================================
        self.image, self.rect = rot_center(self.image_orig, self.rect, self.dir)

#Steer.
    def steerright(self):
        self.dir =self.dir-math.degrees(self.speed/self.rrr)
        #self.dir = self.dir-self.steering
        if self.dir < 0:
            self.dir = self.dir+360
        #if (self.speed > self.maxspeed / 2):
           # self.emit_tracks()   
        self.image, self.rect = rot_center(self.image_orig, self.rect, self.dir)

#fix this function 
    def update(self, last_x, last_y):
        
        if self.wheelangle>0:
            x_old=self.x
            y_old=self.y
            self.rrm=math.sqrt(pow(x_old-self.rpl[0],2)+pow(y_old-self.rpl[1],2))#pixel/s
            
            if   x_old>self.rpl[0]:
                
                fi=math.asin((y_old-self.rpl[1])/self.rrm)
                
            elif x_old<=self.rpl[0]:  
                
                fi=math.pi-math.asin((y_old-self.rpl[1])/self.rrm)
                
            
            fin=fi-(self.speed/self.rrl)
            
            self.x=self.rpl[0]+math.cos(fin)*self.rrm
            self.y=self.rpl[1]+math.sin(fin)*self.rrm
            
        elif self.wheelangle<0:
            
            x_old=self.x
            y_old=self.y
            self.rrm=math.sqrt(pow(x_old-self.rpr[0],2)+pow(y_old-self.rpr[1],2))#pixel/s
            
            if   x_old>self.rpr[0]:
                
                fi=math.asin((y_old-self.rpr[1])/self.rrm)
                
            elif x_old<=self.rpr[0]:  
                
                fi=math.pi-math.asin((y_old-self.rpr[1])/self.rrm)
                
            
            fin=fi+(self.speed/self.rrr)
            
            self.x=self.rpr[0]+math.cos(fin)*self.rrm
            self.y=self.rpr[1]+math.sin(fin)*self.rrm
            
        else:  
    
            self.x = self.x + self.speed * math.cos(math.radians(270-self.dir))
            self.y = self.y + self.speed * math.sin(math.radians(270-self.dir))
            self.reset_tracks()
            

    def impact(self):
        if self.speed > 0:
            self.speed = -self.minspeed

#    def steering():
#        
#        return {
#            'a': 1,
#            'b': 2,
#        }[x]
        

