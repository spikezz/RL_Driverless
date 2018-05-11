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

#Map file.

import os, sys, pygame, math
from pygame.locals import *
from loader import load_image
from random import randrange

#Map filenames.

map_files = []
map_tile = ['track_origin_01.jpg', 
            'track_origin_02.jpg', 
            'track_origin_03.jpg', 
            'track_origin_04.jpg', 
            'track_origin_05.jpg', 
            'track_origin_06.jpg',
            'track_origin_07.jpg', 
            'track_origin_08.jpg', 
            'track_origin_09.jpg', 
            'track_origin_10.jpg', 
            'track_origin_11.jpg', 
            'track_origin_12.jpg',
            'track_origin_13.jpg', 
            'track_origin_14.jpg', 
            'track_origin_15.jpg', 
            'track_origin_16.jpg',  
            'track_origin_17.jpg', 
            'track_origin_18.jpg', 
            'track_origin_19.jpg', 
            'track_origin_20.jpg', 
            'track_origin_21.jpg', 
            'track_origin_22.jpg', 
            'track_origin_23.jpg', 
            'track_origin_24.jpg', 
            'track_origin_25.jpg', 
            'track_origin_26.jpg',
            'track_origin_27.jpg', 
            'track_origin_28.jpg', 
            'track_origin_29.jpg', 
            'track_origin_30.jpg',
            'track_origin_31.jpg', 
            'track_origin_32.jpg', 
            'track_origin_33.jpg', 
            'track_origin_34.jpg', 
            'track_origin_35.jpg', 
            'track_origin_36.jpg',
            'track_origin_37.jpg', 
            'track_origin_38.jpg', 
            'track_origin_39.jpg', 
            'track_origin_40.jpg',
            'track_origin_41.jpg', 
            'track_origin_42.jpg', 
            'track_origin_43.jpg', 
            'track_origin_44.jpg', 
            'track_origin_45.jpg', 
            'track_origin_46.jpg',
            'track_origin_47.jpg', 
            'track_origin_48.jpg', 
            'track_origin_49.jpg', 
            'track_origin_50.jpg',
            'track_origin_51.jpg', 
            'track_origin_52.jpg', 
            'track_origin_53.jpg', 
            'track_origin_54.jpg', 
            'track_origin_55.jpg', 
            'track_origin_56.jpg',
            'track_origin_57.jpg', 
            'track_origin_58.jpg',]
# =============================================================================
# 
# #Map to tile.
# crossing = 0
# straight = 1
# turn     = 2
# split    = 3
# deadend  = 4
# null     = 5
# 
# =============================================================================
#tilemap.
#map_1 = [
         # [2,1,3,1,1,3,1,1,1,4],
         # [1,5,1,5,4,0,1,2,5,4],
         # [1,4,3,1,3,3,1,3,2,1],
         # [3,1,3,1,3,5,4,5,1,1],
          #[3,2,1,5,1,5,3,1,0,3],
          #[1,2,0,1,0,3,0,4,1,1],
         # [1,5,1,4,2,1,1,2,3,1],
          #[1,2,0,1,3,3,0,0,2,1],
         # [1,1,4,2,2,5,1,2,1,3],
        #  [2,3,1,3,1,1,3,1,1,2]
        #]
map_1 = [
          [0,0,0,0,0,0 ,0 ,0 ,19,24,28,33,38,42,0 ,0 ,0 ,0 ,0 ,0 ],
          [0,0,0,0,0,0 ,0 ,0 ,20,25,29,34,0 ,43,45,47,0 ,0 ,0 ,0 ],
          [0,0,0,0,0,0 ,0 ,0 ,21,26,30,35,39,0 ,0 ,48,50,52,54,0 ],
          [0,0,0,0,0,0 ,12,16,22,27,31,36,40,0 ,0 ,0 ,0 ,0 ,55,57],
          [0,0,0,0,0,9 ,13,17,23,0 ,32,37,41,44,46,49,51,53,56,0 ],
          [0,1,3,5,7,10,14,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
          [0,2,4,6,8,11,15,18,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]
        ]
#tilemap rotation, x90ccw
# =============================================================================
# map_1_rot = [
#           [1,1,0,1,1,0,1,1,1,3],
#           [0,0,0,0,1,0,1,0,0,0],
#           [0,1,2,1,0,2,1,2,0,0],
#           [1,1,0,1,3,0,0,0,0,0],
#           [1,0,0,0,0,0,1,1,0,3],
#           [0,2,0,1,0,0,0,3,0,0],
#           [0,0,0,1,3,0,0,1,3,0],
#           [0,1,0,1,0,2,0,0,3,0],
#           [0,0,2,1,3,0,0,2,1,3],
#           [2,2,1,2,1,1,2,1,1,3]
#             ]
# =============================================================================


class Map(pygame.sprite.Sprite):
   # def __init__(self, tile_map, y, x, rot):
    def __init__(self, tile_map, y, x):
        pygame.sprite.Sprite.__init__(self)
        self.image = map_files[tile_map]
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y

#Realign the map
    def update(self, cam_x, cam_y):
        self.rect.topleft = self.x - cam_x, self.y - cam_y
        
     

