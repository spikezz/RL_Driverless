#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:10:19 2019

@author: spikezz
"""


import tensorflow as tf
import sys, pygame, math
import player,maps,tracks,camera,traffic_cone , path
import canvas as cv
import calculation as cal
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from loader import load_image
from pygame.locals import *
from tensorflow.python import pywrap_tensorflow

np.random.seed(1)
tf.set_random_seed(1)



#initial stuff
pygame.init()
screen = pygame.display.set_mode((1360,768),0)
pygame.display.set_caption('Karat Simulation')
font = pygame.font.Font(None, 40)
#initial stuff

##black background for render when car is out of map 
background = pygame.Surface(screen.get_size())
background = background.convert_alpha()
background.fill((1, 1, 1))
##black background for render when car is out of map 

##the transparent canvas for drawing the necessary geometric relationship.The Zero point is at (cam.x,cam.y)
canvas = pygame.Surface(screen.get_size(),SRCALPHA ,32)
canvas = canvas.convert_alpha()
canvas.set_alpha(0)
##the transparent canvas for drawing the necessary geometric relationship.The Zero point is at (cam.x,cam.y)

#testcode for shell
#CENTER_X = 800
#CENTER_Y = 450

##find the center of screen
CENTER_X =  float(pygame.display.Info().current_w /2)
CENTER_Y =  float(pygame.display.Info().current_h /2)
CENTER=(CENTER_X,CENTER_Y)
##find the center of screen

##constant of path
half_path_wide=80
##constant of path

##create some objects
clock = pygame.time.Clock()
car = player.Player()
cam = camera.Camera()
##create some objects

##create the spriteGroup contains objects
list_cone_yellow=[]
list_cone_blue=[]
list_path_point=[]
map_s= pygame.sprite.Group()
player_s= pygame.sprite.Group()
tracks_s= pygame.sprite.Group()
cone_s  = pygame.sprite.Group()
path_s  = pygame.sprite.Group()
cone_h  = pygame.sprite.Group()
##create the spriteGroup contains objects

##tracks  initalize. tracks are points left  while driving
tracks.initialize()
cam.set_pos(car.x, car.y)
##tracks  initalize. tracks are points left  while driving

##add objects
#car
player_s.add(car)
#car
#cone
coneb=traffic_cone.cone(CENTER[0],CENTER[1]-half_path_wide,-1,car.x,car.y)
coney=traffic_cone.cone(CENTER[0],CENTER[1]+half_path_wide,1,car.x,car.y)
#coneback=traffic_cone.cone(CENTER[0]+half_path_wide/2,CENTER[1],-1,car.x,car.y)
coneback=None
cone_s.add(coneb)
cone_s.add(coney)
#cone_h.add(coneback)
list_cone_blue.append(coneb)
list_cone_yellow.append(coney)
#cone
#middle point
startpoint=path.path(CENTER[0],CENTER[1],car.x,car.y)
path_s.add(startpoint)
list_path_point.append(startpoint)
#middle point
##add objects

##set orientation
car.set_start_direction(90)
#the turning angle of the wheel 
angle=0
#the turning angle of the wheel  
#saved angle
angle_old=0
#saved angle
##set orientation

###initial Model of the car
##specification of the car
half_Max_angle=45
half_middle_axis_length=20
half_horizontal_axis_length=17
radius_of_wheel=10
el_length=500
##specification of the car
##action relevant
turing_speed=1.0
##action relevant
model=cv.initialize_model(CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)
###initial Model of the car

##count/COUNT_FREQUENZ is the real time
episode_time=60
#every loop +1 for timer
count=0
#every loop +1 for timer
#FPS Frame(loop times) per second
COUNT_FREQUENZ=500
#FPS Frame(loop times) per second
#switch for timer
start_timer=False
#switch for timer
##count/COUNT_FREQUENZ is the real time

##map picture size
FULL_TILE = 1000
##map picture size

##offset of the start position of the car
xc0=car.x
yc0=car.y
##offset of the start position of the car

##the old (initial) position for speed measurements
x_old=car.x
y_old=car.y
##the old (initial) position for speed measurements

##read the maps and draw
for tile_num in range (0, len(maps.map_tile)):

    #add submap idx to array
    maps.map_files.append(load_image(maps.map_tile[tile_num],False))
    #add submap idx to array
    
for x in range (0,7):
    
    for y in range (0, 20):
        
        #add submap to mapgroup
        map_s.add(maps.Map(maps.map_1[x][y], x * FULL_TILE, y * FULL_TILE))
        #add submap to mapgroup
##read the maps and draw
       
##collision detection
collide=False
collide_finish=False
##collision detection   

##constant for cone
#cone_x=CENTER[0]
#cone_y=CENTER[1]
#position_sensor=[0,0]
#index
i=0
#index
#counter of yellow cone
p=0
#counter of yellow cone
#counter of blue cone
q=0#blue
#counter of blue cone
#Projection of point on the canvas of lidar data from blue cone
draw_blue_cone=[]
#Projection of point on the canvas of lidar data from blue cone
#Projection of point on the canvas of lidar data from yellow cone
draw_yellow_cone=[]
#Projection of point on the canvas of lidar data from yellow cone
#absolute value of distance from yellow cone
dis_yellow=[]
#absolute value of distance from yellow cone
#absolute value of distance from blue cone
dis_blue=[]
#absolute value of distance from blue cone
#absolute value of distance from back cone
#dis_back=0
#absolute value of distance from back cone
#input data form of RL algorithm of blue cone
vektor_blue=[]
#input data form of RL algorithm of blue cone
#input data form of RL algorithm of yellow cone
vektor_yellow=[]
#input data form of RL algorithm of yellow cone
#sum of square of blue cone
dis_blue_sqr_sum=0
#sum of square of blue cone
#sum of square of yellow cone
dis_yellow_sqr_sum=0
#sum of square of yellow cone
#positive deviation
#diff_sum_yb=0
#positive deviation
##constant for cone

##initial draw element
draw_blue_cone.append([0,0])
draw_yellow_cone.append([0,0])
dis_blue.append(0)
dis_yellow.append(0)
vektor_blue.append([0,0])
vektor_yellow.append([0,0])
##initial draw element

##constant for path
#index
j=0
#index
#temporary path coordinate for mouse and auto path build
path_x=0
path_y=0
#temporary path coordinate for mouse and auto path build
path_mirror=1
ctrl_pressed=False
#lidar effective distance
dis_path=0
#lidar effective distance
#temporary last path milestone
last_point=[startpoint.x,startpoint.y]
#temporary last path milestone
#path milestone
draw_path=[]
#path milestone
draw_path.append([0,0])#init
#find 2 path point
path_close_1=[0,0]
path_close_2=[0,0]
yellow_cone_close_1=[0,0]
yellow_cone_close_2=[0,0]
blue_cone_close_1=[0,0]
blue_cone_close_2=[0,0]
sin_projection_yellow=0
sin_projection_blue=0
#dis_close_path_1=0
#dis_close_path_2=0
dis_between_path=0
dis_between_yellow_cone=0
dis_between_blue_cone=0
dis_close_path_temp=0
base_path_mileage=0
tag=0
tag_2=1
path_tag=[]
path_tag_2=[]
ta=0
ta_2=0
projection=[0,0]
debug=False
swich_cal_projection=True
#find 2 path point

##constant for path

##constant for lidar
#number of sensored blue cone
k=0
#number of sensored blue cone
#number of sensored yellow cone
l=0
#number of sensored yellow cone
#range of lidar
bound_lidar=CENTER[0]*2/5
#range of lidar
##constant for lidar

##temporary state to form all the inputs
state=[[],[],0,[],0]
##temporary state to form all the inputs

##konstant of RL
#numer of completed process
ep_total=0
#numer of completed process
#switch of the summary
summary=False
#switch of the summary
#episode after max reward updated
ep_lr=0
#episode after max reward updated
#weight for the speed in reward
speed_faktor=10
#weight for the speed in reward
#speed_faktor_enhance=1
#angle_faktor_enhance=1
#weight for the distance in reward
distance_faktor=0
#weight for the distance in reward
#minimum distance before impact
safty_distance_impact=50
#minimum distance before impact
#minimum distance before turning
safty_distance_turning=65
punish_turning=False
idx_punish=0
punished_reward=0
punish_batch_size=3
#minimum distance before turning
#distance which means impact
collision_distance=40
#distance which means impact
#total moved distance
distance=0
#total moved distance
#total moved distance on middle line
distance_projection=0
distance_projection_old=0
speed_projection=0
v_distance_projection=0
#total moved distance on middle line
#distance every episode
distance_set=[]
#distance every episode
#reward of the current state and action
reward=1
#reward of the current state and action
#temporary reward of the whole episode 
reward_sum=0
#temporary reward of the whole episode 
#average whole episode reward of the whole training  process
reward_mean=[]
#average whole episode reward of the whole training  process
#set of runing reward
rr=[]
#set of runing reward
#index of runing reward set
rr_idx=0
#index of runing reward set
#reward of the whole episode 
running_reward =0
#reward of the whole episode 
#max running reward until now
running_reward_max=0
#max running reward until now
#ratio of the max max running reward and the average running reward.1 is the goal
reward_mean_max_rate=[]
#ratio of the max max running reward and the average running reward.1 is the goal
#vt=0
#learning start
start_action=False
#learning start
#Rendering start
Render=True
#Rendering star
# learning rate for actor
LR_A = 1e-6
# learning rate for actor
# learning rate for critic
LR_C = 1e-6
# learning rate for critic
# reward discount
rd = 0.9
# reward discount
#after this learning number of main net update the target net of actor
REPLACE_ITER_A = 2048
#after this learning number of main net update the target net of actor
#after this learning number of main net update the target net of Critic
REPLACE_ITER_C = 2048
#after this learning number of main net update the target net of Critic
#occupied memory
MEMORY_CAPACITY = 131072
#occupied memory
#size of memory slice
BATCH_SIZE =128
#size of memory slice
#minimal exploration wide of action
VAR_MIN_updated = 0.01
VAR_MIN = 0.1
#minimal exploration wide of action
#initial exploration wide of action
var1 = 1
var2 = 1
#var1 = 0.1
#var2 = 0.1
#initial exploration wide of action
#dimension of action
ACTION_DIM = 2
#dimension of action
#action boundary
ACTION_BOUND0 = np.array([-0.5,0.5])
ACTION_BOUND1 = np.array([-45,45])
#action boundary
#action boundary a[0]*ACTION_BOUND[0],a[1]*ACTION_BOUND[1]
ACTION_BOUND=np.array([0.5,5])
#action boundary a[0]*ACTION_BOUND[0],a[1]*ACTION_BOUND[1]
#max reward reset
max_reward_reset=0
#max reward reset
#set of manual changed learning rate 
lr_set=[]
#set of manual changed learning rate 
#dimension of inputs for RL Agent

#dimension of inputs for RL Agent
#inputs state of RL Agent

#inputs state of RL Agent
#copy the state

#copy the state

##konstant of RL
#
path_man=[]
corner=[]
corner.append(32)
corner.append(36)
corner.append(41)
corner.append(43)
corner.append(44)
corner.append(45)
corner.append(46)
corner.append(47)
corner.append(48)
corner.append(49)
corner.append(54)
for t in range (1,corner[0]):
    path_man.append([CENTER[0]-49*t,path_mirror*(CENTER[1]-3*t)])

for t in range (1,5):
    path_man.append([path_man[corner[0]-2][0]-49.74*t,path_mirror*(path_man[corner[0]-2][1]-5*t)])
    
for t in range (1,6):
    path_man.append([path_man[corner[1]-2][0]-49*t,path_mirror*(path_man[corner[1]-2][1]-10*t)])

for t in range (1,3):
    path_man.append([path_man[corner[2]-2][0]-45.82*t,path_mirror*(path_man[corner[2]-2][1]-20*t)])

path_man.append([path_man[corner[3]-2][0]-40,path_mirror*(path_man[corner[3]-2][1]-30)])
path_man.append([path_man[corner[4]-2][0]-31,path_mirror*(path_man[corner[4]-2][1]-39.23)])
path_man.append([path_man[corner[5]-2][0]-19.6,path_mirror*(path_man[corner[5]-2][1]-46)])
path_man.append([path_man[corner[6]-2][0]+15,path_mirror*(path_man[corner[6]-2][1]-47.7)])
path_man.append([path_man[corner[7]-2][0]+30,path_mirror*(path_man[corner[7]-2][1]-40)])
path_man.append([path_man[corner[8]-2][0]+35,path_mirror*(path_man[corner[8]-2][1]-35.71)])
for t in range (1,6):
    path_man.append([path_man[corner[9]-2][0]+43.5*t,path_mirror*(path_man[corner[9]-2][1]-24.65*t)])
    
for t in range (1,23):
    path_man.append([path_man[corner[10]-2][0]+47.37*t,path_mirror*(path_man[corner[10]-2][1]-16*t)])
    
coneback=traffic_cone.cone(path_man[corner[10]-2][0]+47.37*18,path_mirror*(path_man[corner[10]-2][1]-16*18),1,car.x,car.y)
cone_h.add(coneback)
#for t in range (1,5):
    #path_man.append([path_man[8][0]-50*t,path_man[8][1]+10*math.sqrt(t)])

#path_man.append([path_man[30][0]-48.29,path_man[30][1]-12.94])
#path_man.append([path_man[31][0]-25*math.sqrt(3),path_man[31][1]-25])
    
# =============================================================================
# coneb_back=traffic_cone.cone(CENTER[0]+100,CENTER[1]-20,-1,car.x,car.y)
# coney_back=traffic_cone.cone(CENTER[0]+100,CENTER[1]+20,1,car.x,car.y)
# 
# cone_s.add(coneb_back)
# cone_s.add(coney_back)
# 
# list_cone_blue.append(coneb_back)
# list_cone_yellow.append(coney_back)
# 
# draw_yellow_cone.append([0,0])
# dis_yellow.append(0)
# vektor_yellow.append([0,0])
# p=p+1
# 
# draw_blue_cone.append([0,0])
# dis_blue.append(0)
# vektor_blue.append([0,0])
# q=q+1
# =============================================================================


    
for pa in path_man:
    path_x=pa[0]
    path_y=pa[1]
     
    path_new=path.path(path_x,path_y,car.x,car.y)
    list_path_point.append(path_new)  
    path_s.add(path_new)  
     
     
     
    line=[last_point,[path_new.x,path_new.y]]
     
    cone_x, cone_y = cal.calculate_t(line,1,half_path_wide,car.x,car.y)
    cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
    list_cone_yellow.append(cone_new)
    cone_s.add(cone_new)
     
    draw_yellow_cone.append([0,0])
    dis_yellow.append(0)
    vektor_yellow.append([0,0])
    p=p+1
    
    cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide,car.x,car.y)
    cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
    list_cone_blue.append(cone_new)
    cone_s.add(cone_new)
   
    draw_blue_cone.append([0,0])
    dis_blue.append(0)
    vektor_blue.append([0,0])
    q=q+1

    
    last_point=[path_x+car.x,path_y+car.y]
    draw_path.append([0,0])
    j=j+1
# =============================================================================
##
        
while True:


    #show1=len(list_cone_yellow)
    ##key event continually
    
    
    keys = pygame.key.get_pressed()
    

    if keys[K_1]:

        angle=half_Max_angle
        
    if keys[K_2]:
  
        angle=30
        
    if keys[K_3]:
       
        angle=15    
        
    if keys[K_4]:

        angle=2.3
    
    if keys[K_6]:
        
        angle=0  
        
    if keys[K_5]:
        
        car.speed=10
        
    if keys[K_7]:

        angle=-2.3
        
    if keys[K_8]:
  
        angle=-15
        
    if keys[K_9]:
     
        angle=-30   
        
    if keys[K_0]:
      
        angle=-half_Max_angle
        
    if keys[K_LEFT]:
        
        turing_speed=1
        
        if angle<0:
            
            angle=-1
            
        if angle<half_Max_angle+1:
            
            angle=angle+turing_speed

        #if angle==0:
           
    if keys[K_RIGHT]:
        
        turing_speed=-1
        
        if angle>0:
            
            angle=1
        
        if angle>-half_Max_angle-1:
            
            angle=angle+turing_speed
   
        #if angle==0:
           
    if keys[K_UP]:
        
        car.accelerate(car.acceleration)
        

        
    if keys[K_DOWN]:
        
        car.deaccelerate()
        
    if keys[K_BACKSPACE]:
        
        car.speed=0
    
    if keys[K_LCTRL] : 
        
        ctrl_pressed=True
        
        if pygame.mouse.get_pressed()==(True,False,False):
            
            path_x, path_y = pygame.mouse.get_pos()
            dis_path=cal.calculate_r((path_x+car.x,path_y+car.y),last_point)
            
            if dis_path>50:
                #pass
                
                path_new=path.path(path_x,path_y,car.x,car.y)
                list_path_point.append(path_new)  
                path_s.add(path_new)  
                
                line=[last_point,[path_new.x,path_new.y]]
                
                cone_x, cone_y = cal.calculate_t(line,1,half_path_wide,car.x,car.y)
                cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
                list_cone_yellow.append(cone_new)
                cone_s.add(cone_new)
                
                draw_yellow_cone.append([0,0])
                dis_yellow.append(0)
                vektor_yellow.append([0,0])
                p=p+1
                
                cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide,car.x,car.y)
                cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
                list_cone_blue.append(cone_new)
                cone_s.add(cone_new)
               
                draw_blue_cone.append([0,0])
                dis_blue.append(0)
                vektor_blue.append([0,0])
                q=q+1
                
                last_point=[path_x+car.x,path_y+car.y]
                draw_path.append([0,0])
                j=j+1
    else:
        
        ctrl_pressed=False
            
    ##system event
    for event in pygame.event.get():
#            print(event)
        # quit for windows
        if event.type == QUIT:
                    
            pygame.quit()
            
            sys.exit()

                        
        if event.type == KEYDOWN :
            
            # quit for esc key
#                if event.key == K_ESCAPE:  
#                                
#                    pygame.quit()
#                    
#                    sys.exit()
                
            #timer
            if event.unicode == ' ':  
                
                if start_timer==False: 
                    
                    start_timer=True
                    
                else: 
                    
                    start_timer=False
                    
            if event.unicode == '\r':
                
                start_action=True
                
                
            if event.unicode == '\x08':
                
                car.reset()
                car.set_start_direction(90)
            
            if event.unicode == 'r':
                
                if  Render==False: 
                    
                    Render=True
                    
                else: 
                    
                    Render=False
                
            if event.key == K_e:
                
                #ep_lr=0
                max_reward_reset=max_reward_reset+1
                
            if event.unicode == 'd':
                
                if  debug==False: 
                    
                    debug=True
                    
                else: 
                    
                    debug=False

#                if event.unicode == 'd':
#                    pass
##                    lr=lr/10
##                    RL.learning_rate=lr
##                    print("max lr:",lr)
#
#                if event.unicode == 'm':
#                    pass
##                    lr=lr*10
##                    RL.learning_rate=lr
##                    print("max lr:",lr)
#                
            if event.unicode == 'o':

                episode_time=episode_time+0.1
                print("episode_time:",episode_time)
            
            if event.unicode == 'p':

                episode_time=episode_time-0.1
                print("episode_time:",episode_time)
                
#                if event.key == K_t:
#                    pass
#
#                    #if os.path.isdir(path): shutil.rmtree(path)
#                    #os.mkdir(path)
#                    #ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
#                    #save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
#                    #print("\nSave Model %s\n" % save_path)
#                                        
#
#                    
        if event.type == KEYUP :    
            
            if event.key == K_LEFT or event.key == K_RIGHT:
                
                angle=0

                
        if event.type == MOUSEBUTTONDOWN and ctrl_pressed==False:
            

                
            if pygame.mouse.get_pressed()==(True,False,False):
                
                cone_x, cone_y = pygame.mouse.get_pos()
                cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
                list_cone_yellow.append(cone_new)
                cone_s.add(cone_new)
                
                draw_yellow_cone.append([0,0])
                dis_yellow.append(0)
                vektor_yellow.append([0,0])
                p=p+1
                
            elif pygame.mouse.get_pressed()==(False,False,True) :
                
                cone_x, cone_y = pygame.mouse.get_pos()
                cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
                list_cone_blue.append(cone_new)
                cone_s.add(cone_new)
               
                draw_blue_cone.append([0,0])
                dis_blue.append(0)
                vektor_blue.append([0,0])
                q=q+1
                
    ##system event
# =============================================================================
#     if keys[K_DELETE]:
#         
#         cone_s.empty()
#         list_cone_blue=[]
#         list_cone_yellow=[]
# =============================================================================
    ##key event 

                
        
# =============================================================================
#             if action[1]> angle+var:
# 
#                 angle=angle+car.steering
#                 
#             if action[1]< angle-var:
# 
#                 angle=angle-car.steering
# =============================================================================
        #print("angle:",angle)
        #print("action[1]:",action[1])

    ##reward
    
    distance=distance+car.speed
    #reward=speed_faktor*car.speed
        
    car.soften()
    #camera position reset
    cam.set_pos(car.x, car.y)
    
    
    #speed calculation for center of the car
#        speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s
    speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))
#        print("speed",speed)
    #update the old position
    x_old=car.x
    y_old=car.y
    
    ##
    
        
    #postion_sensor=[(model[7][0][0]-CENTER[0]+car.x),(model[7][0][1]-CENTER[1]+car.y)]
    ##
       
    ##text setting
    
    text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (0, 0, 102))
    textpos_fps = text_fps.get_rect(centery=25, left=20)
    
    #print("FPS:",clock.get_fps())
    
    text_timer = font.render('Timer: ' + str(round(float(count/COUNT_FREQUENZ),2)) +'s', 1, (0, 0, 102))
    textpos_timer = text_timer.get_rect(centery=65, left=20)
     
    text_loop = font.render('Loop: ' + str(int(clock.get_time())) +'ms', 1, (0, 0, 102))
    textpos_loop = text_loop.get_rect(centery=105, left=20)
    
    text_pos= font.render('POS: ' + '( '+str(round(float(car.x-xc0),2))+' , '+str(round(float(car.y-yc0),2))+' )', 1, (0, 0, 102))   
    textpos_pos = text_pos.get_rect(centery=145, left=20)
    
    text_posr= font.render('POS_RAW: ' +  '( '+str(round(float(car.x),2))+' , '+str(round(float(car.y),2))+' )', 1, (0, 0, 102))   
    textpos_posr = text_posr.get_rect(centery=185, left=20)
    
    text_dir= font.render('Direction: ' + str(round(float(car.dir),2)), 1, (0, 0, 102))   
    textpos_dir = text_dir.get_rect(centery=225, left=20)
    
    text_speed= font.render('speed: ' + str(round(float(speed),2))+'pixel/s'+'|'+str(round(float(car.speed),2))+'pixel/loop', 1, (0, 0, 102))   
    textpos_speed = text_speed.get_rect(centery=265, left=20)
    
    text_colour= font.render('colour: ' + str(round(screen.get_at(((int(CENTER[0]-50), int(CENTER[1]-50)))).g,2)), 1, (0, 0, 102))   
    textpos_colour = text_colour.get_rect(centery=305, left=20)
    
#        text_posr= font.render('POS_RAW: ' +  '( '+str(round(float(car.x),2))+' , '+str(round(float(car.y),2))+' )', 1, (0, 0, 102))   
#        textpos_posr = text_posr.get_rect(centery=345, left=20)
    #text_dis_yellow= font.render('distance to yellow cone: ' + str(round(float(dis_yellow[0]),2)), 1, (0, 0, 102))   
    #textpos_dis_yellow = text_dis_yellow.get_rect(centery=345, left=20)
    
    #text_dis_blue= font.render('distance to blue cone: ' + str(round(float(dis_blue[0]),2)), 1, (0, 0, 102))   
    #textpos_dis_blue =text_dis_blue.get_rect(centery=385, left=20)
    
    
    
    
    ##text setting
    
    
    #model of car
    #35*41 rect
    #middle axis 40
    #axis 35
    #wheel 20
    
    #angle signal give to the object car
    
    if start_action==True:
        
        start_timer=True
        
    car.wheelangle=angle
    model=cv.turning(model,angle,CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)

    
    if angle>0 :
    
        car.rrl=model[19]
        car.steerleft()
        #vektor_speed=[car.speed*,car.speed*]
       
    elif angle<0  :

        car.rrr=model[21]
        car.steerright()
    

    model=cv.rotate(model,CENTER,car.dir)
    
    dis_close_yellow_cone_1=1000
    
    for i in range (0, p+1):
        
        dis_yellow[i]=cal.calculate_r((car.x,car.y),(list_cone_yellow[i].x-model[7][0][0],list_cone_yellow[i].y-model[7][0][1]))
        draw_yellow_cone[i]=[list_cone_yellow[i].x-cam.x,list_cone_yellow[i].y-cam.y]
        
        if dis_yellow[i]<dis_close_yellow_cone_1:
            
            yellow_cone_close_1=draw_yellow_cone[i]
            dis_close_yellow_cone_1=dis_yellow[i]

    
    dis_close_yellow_cone_2=1000
    
    for i in range (0, p+1):
        
        if dis_yellow[i]<dis_close_yellow_cone_2 and dis_yellow[i]>dis_close_yellow_cone_1:
            yellow_cone_close_2=draw_yellow_cone[i]
            dis_close_yellow_cone_2=dis_yellow[i]
    
    dis_between_yellow_cone=cal.calculate_r(yellow_cone_close_1,yellow_cone_close_2)
    
    sin_projection_yellow=cal.calculate_projection(True,dis_close_yellow_cone_1,dis_close_yellow_cone_2,dis_between_yellow_cone)[1]
#        print("sin_projection_yellow:",sin_projection_yellow)
    
    if dis_between_yellow_cone<20:
        sin_projection_yellow=dis_close_yellow_cone_1
#            print("broke",dis_between_yellow_cone)
        
    dis_close_blue_cone_1=1000
    
    for i in range (0, q+1):
        
        dis_blue[i]=cal.calculate_r((list_cone_blue[i].x-model[7][0][0],list_cone_blue[i].y-model[7][0][1]),(car.x,car.y))
        draw_blue_cone[i]=[list_cone_blue[i].x-cam.x,list_cone_blue[i].y-cam.y]         
        if dis_blue[i]<dis_close_blue_cone_1:
            
            blue_cone_close_1=draw_blue_cone[i]
            dis_close_blue_cone_1=dis_blue[i]

    
    dis_close_blue_cone_2=1000
    
    for i in range (0, q+1):
        
        if dis_blue[i]<dis_close_blue_cone_2 and dis_blue[i]>dis_close_blue_cone_1:
            blue_cone_close_2=draw_blue_cone[i]
            dis_close_blue_cone_2=dis_blue[i]
            
    dis_between_blue_cone=cal.calculate_r(blue_cone_close_1,blue_cone_close_2)
    
    sin_projection_blue=cal.calculate_projection(True,dis_close_blue_cone_1,dis_close_blue_cone_2,dis_between_blue_cone)[1]
#        print("sin_projection_blue:",sin_projection_blue)
    
    if dis_between_blue_cone<20:
        
        sin_projection_blue=dis_close_blue_cone_1
#            print("broke",dis_between_blue_cone)
        
    for i in range (0, j+1):
        
        draw_path[i]=[list_path_point[i].x-cam.x,list_path_point[i].y-cam.y]
   
    dis_close_path_1=1000
    
    for i in range (0, j+1):
        
        dis_close_path_temp=cal.calculate_r((list_path_point[i].x-model[7][0][0],list_path_point[i].y-model[7][0][1]),(car.x,car.y))
        
        if dis_close_path_temp<dis_close_path_1:
            path_close_1=draw_path[i]
            dis_close_path_1=dis_close_path_temp
            tag=i
        
    dis_close_path_2=1000
   
    for i in range (0, j+1):
        
        dis_close_path_temp=cal.calculate_r((list_path_point[i].x-model[7][0][0],list_path_point[i].y-model[7][0][1]),(car.x,car.y))

        if dis_close_path_temp<dis_close_path_2 and dis_close_path_temp>dis_close_path_1:
            path_close_2=draw_path[i]
            dis_close_path_2=dis_close_path_temp
            tag_2=i
    #init             
    if path_tag==[]:
        
        path_close_1=draw_path[0]
        path_tag.append(tag)

    #init         
            
    if path_tag_2==[]:
        
        path_close_2=draw_path[1]
        path_tag_2.append(tag_2)
        
    dis_between_path=cal.calculate_r(path_close_1,path_close_2)
    
    
        
    if tag!=path_tag[ta]:
        
        path_tag.append(tag)
        ta=ta+1
        swich_cal_projection=False
#            print("ta",ta) 

    if tag_2!=path_tag_2[ta_2]:
        
        path_tag_2.append(tag_2)
        if tag_2>path_tag_2[ta_2]:
            swich_cal_projection=True 
            base_path_mileage=dis_between_path*ta
            
        ta_2=ta_2+1
        
    elif count==0:
        
        base_path_mileage=0

#        print("dis_between_path",dis_between_path)
    projection=cal.calculate_projection(swich_cal_projection,dis_close_path_1,dis_close_path_2,dis_between_path)
    projektion=[0,0]

    distance_projection=base_path_mileage+projection[0]
    v_distance_projection=projection[1]
#        print("count:",count)
    if count==0:
        
        distance_projection_old=distance_projection

    speed_projection=distance_projection-distance_projection_old

    if speed_projection>car.maxspeed:
#            print("speed_projection:",speed_projection)
        speed_projection=car.maxspeed

        
    if speed_projection<0:
        
        speed_projection=-speed_projection

    distance_projection_old=distance_projection

    if coneback:
        
        dis_back=cal.calculate_r((car.x,car.y),(coneback.x-model[7][0][0],coneback.y-model[7][0][1]))

    ##draw background
    if Render==True:
        
        screen.blit(background, (0,0))
    ##
    
    
    ##update map
    map_s.update(cam.x, cam.y)
    if Render==True:
        map_s.draw(screen)
    ##update map
    
    ##draw cones
    cone_s.update(cam.x, cam.y)
    cone_h.update(cam.x, cam.y)
    if Render==True:
        cone_s.draw(screen)
        cone_h.draw(screen)
    ##draw cones
    
    ##draw path point
    path_s.update(cam.x,cam.y)
    if Render==True:
        
        path_s.draw(screen)
    
    
    ##draw path point
    
    ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
    car.rpl=(model[18][0][0]-CENTER[0]+car.x,model[18][0][1]-CENTER[1]+car.y)
    car.rpr=(model[20][0][0]-CENTER[0]+car.x,model[20][0][1]-CENTER[1]+car.y)
    ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
  
    

    
    
    ##
    for i in range (0, q+1):
        
        vektor_blue[i]=cv.input_vektor_position(model,draw_blue_cone[i],CENTER,car.dir)
    
    for i in range (0, p+1):
        
        vektor_yellow[i]=cv.input_vektor_position(model,draw_yellow_cone[i],CENTER,car.dir)

    vektor_speed=[car.speed*math.cos(math.radians(270-angle)),car.speed*math.sin(math.radians(270-angle))]
#        print("vektor_speed",vektor_speed)
    ##
    
    
    ##
    text_episode_time= font.render('episode_time: ' +str(round(float(episode_time),2)), 1, (0, 0, 102))   
    text_pos_episode_time = text_episode_time.get_rect(centery=425, left=20)
    
    text_blue= font.render('vektor_blue: '+ '( ' + str(round(float(vektor_blue[0][0]),2))+' , '+str(round(float(vektor_blue[0][1]),2))+' )', 1, (0, 0, 102))   
    textpos_blue = text_blue.get_rect(centery=465, left=20)
    
    text_speed_v= font.render('vektor_speed: ' +  '( ' + str(round(float(vektor_speed[0]),2))+' , '+str(round(float(vektor_speed[1]),2))+' )', 1, (0, 0, 102))   
    textpos_speed_v = text_speed_v.get_rect(centery=505, left=20)
    ##
    
    
    #anything want to show
    text_show1= font.render('distance: ' + str(round(float(distance),2)), 1, (0, 0, 102))   
    textpos_show1 = text_dir.get_rect(centery=545, left=20)
    
    text_show2= font.render('reward: ' + str(round(float(running_reward),2)), 1, (0, 0, 102))   
    textpos_show2 = text_dir.get_rect(centery=585, left=20)
    #anything want to show
    
    
    
    ##
    player_s.update(cam.x, cam.y)
    if Render==True:
        
        player_s.draw(screen)
    ##
    
    
    ##
    tracks_s.add(tracks.Track(cam.x + CENTER[0] , cam.y + CENTER[1], car.dir))
    tracks_s.update(cam.x, cam.y)
    if Render==True:
        
        tracks_s.draw(screen)
    ##
    
    
    ##
    canvas.fill((255, 255, 255,0))
    ##
    
    ##draw lines to find center of the car2378.525163
    pygame.draw.line(canvas, (0,100,0), (100,CENTER[1]), (CENTER[0],CENTER[1]),2)
    pygame.draw.line(canvas, (0,100,0), (CENTER[0],100), (CENTER[0],CENTER[1]),2)
    ##draw lines to find center of the car
    
    
    ##draw model of the car
    pygame.draw.line(canvas, (255,255,255), model[0][0], model[1][0],4)
    pygame.draw.line(canvas, (255,255,255), model[2][0], model[3][0],4)#front wheel
    pygame.draw.line(canvas, (255,255,255), model[4][0], model[5][0],4)#front axis
    pygame.draw.line(canvas, (255,255,255), model[6][0], model[7][0],4)#middle axis
    pygame.draw.line(canvas, (255,255,255), model[10][0], model[11][0],4)
    pygame.draw.line(canvas, (255,255,255), model[12][0], model[13][0],4)#back wheel
    pygame.draw.line(canvas, (255,255,255), model[8][0], model[9][0],4)#back axis
    ##draw model of the car
       

    ##draw back axis extension
    pygame.draw.line(canvas, (255,255,102), model[16][0],model[17][0],2)
    ##draw back axis extension
    
    ##draw distance to cones
    #print("model[7][0]:",model[7][0])
    for i in range (0, q+1):
        
        if dis_blue[i]<bound_lidar:
            if draw_blue_cone[i]==blue_cone_close_1 or draw_blue_cone[i]==blue_cone_close_2:
                
                pygame.draw.line(canvas, (255,255,0), model[7][0],draw_blue_cone[i],5)
                
            else:
                
                pygame.draw.line(canvas, (0,255,255), model[7][0],draw_blue_cone[i],2)
    
    for i in range (0, p+1):
        
        if dis_yellow[i]<bound_lidar:
            
            if draw_yellow_cone[i]==yellow_cone_close_1 or draw_yellow_cone[i]==yellow_cone_close_2:
                
                pygame.draw.line(canvas, (0,255,255), model[7][0],draw_yellow_cone[i],5)
                
            else:  
                
                pygame.draw.line(canvas, (255,255,0), model[7][0],draw_yellow_cone[i],2)
    ##draw distance to cones
    pygame.draw.line(canvas, (0,255,255), yellow_cone_close_1,yellow_cone_close_2,5)
    pygame.draw.line(canvas, (255,255,0), blue_cone_close_1,blue_cone_close_2,5)
    
    ##draw path
    for i in range (1, j+1):
        
        pygame.draw.line(canvas, (0,255,0), draw_path[i-1],draw_path[i],2)
        
    ##draw path

    pygame.draw.line(canvas, (220,20,60), model[7][0],path_close_1,3)
    pygame.draw.line(canvas, (220,20,60), model[7][0],path_close_2,3)  
    ##draw front axis extension
    if angle >= 2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), model[5][0],model[14][0],2)#frontwheel turing axis
   
    if angle <= -2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), model[4][0],model[15][0],2)#frontwheel turing axis
    
    ##draw front axis extension
    
    
    ##draw  the circle which car moving along in canvas
    if  angle>0:
        pass
        #pygame.draw.arc(canvas, (255,255,102), (model[18][0][0]-model[19],model[18][0][1]-model[19],2*model[19],2*model[19]), 0, 360, 0)
    
    if  angle<0:
        pass
        #pygame.draw.arc(canvas, (255,255,102), (model[20][0][0]-model[21],model[20][0][1]-model[21],2*model[21],2*model[21]), 0, 360, 0)
    
    ##draw  the car moving circle in canvas
    
    
    ##show canvas
    if Render==True:
        
        screen.blit(canvas, (0,0))
    ##show canvas
    
    
    ##show text
    if Render==True:
        
        screen.blit(text_fps, textpos_fps)
        screen.blit(text_timer, textpos_timer)
        screen.blit(text_loop, textpos_loop)
        screen.blit(text_pos, textpos_pos)
        screen.blit(text_posr, textpos_posr)
        screen.blit(text_dir, textpos_dir)
        screen.blit(text_speed, textpos_speed)
        screen.blit(text_colour, textpos_colour)
        #screen.blit(text_dis_yellow, textpos_dis_yellow)
        #screen.blit(text_dis_blue, textpos_dis_blue)
        screen.blit(text_episode_time, text_pos_episode_time)
        screen.blit(text_blue, textpos_blue)
        screen.blit(text_speed_v, textpos_speed_v)
        screen.blit(text_show1, textpos_show1)
        screen.blit(text_show2, textpos_show2)
    ##show text
    
    
    ##start drawing
    
    #vektor_blue Within radar range
    vektor_blue_temp=[]
    #vektor_blue Within radar range
    #vektor_yellow Within radar range
    vektor_yellow_temp=[]
    #vektor_yellow Within radar range
#        for v in vektor_blue[i]
#        
    for i in range (0, q+1):
        
        if dis_blue[i]<bound_lidar:
            
            vektor_blue_temp.append(vektor_blue[i])
            dis_blue_sqr_sum=dis_blue_sqr_sum+pow(dis_blue[i],2)
            
    for i in range (0, p+1):
        
        if dis_yellow[i]<bound_lidar:
            
            vektor_yellow_temp.append(vektor_yellow[i])
            dis_yellow_sqr_sum=dis_yellow_sqr_sum+pow(dis_yellow[i],2)
            
            
    if start_timer==True:

        count=count+1
    ##timer 
    #reward=distance_faktor*distance
    if start_action==True:
            
            
#            print("reward",reward)
#            print("v_distance_projection:",reward/speed_projection-speed_faktor)
#            print("speed_projection",speed_projection)
#            print("old reward:",-speed_projection*20*(car.maxspeed/car.acceleration)/v_distance_projection)
        
        for i in range (0, q+1):
        
            if dis_blue[i]<collision_distance:
                
                collide=True
                
        for i in range (0, p+1):
        
            if dis_yellow[i]<collision_distance:
                
                collide=True
                
        if coneback:
            
            if dis_back<collision_distance*2:
                
                collide_finish=True

#            if punish_turning==True:
#                reward=-math.sqrt(reward**2)
#                idx_punish = M.pointer % M.capacity
#                punished_reward = M.read(idx_punish,punish_batch_size)[:, -input_dim - 1]
#                for t in range(0,len(punished_reward)):
#                    
#                    if punished_reward[t]>0:
#                        
#                        reward_sum=reward_sum-2*math.sqrt(punished_reward[t]**2)
#                        punished_reward[t]=punished_reward[t]-2*math.sqrt(punished_reward[t]**2)
##                print("punished_reward",punished_reward)
##                punished_reward =-2*car.maxspeed/punished_reward
##                print("punished_reward_new",punished_reward)
#                M.write(idx_punish,punish_batch_size,punished_reward)
#                punished_reward = M.read(idx_punish,punish_batch_size)[:, -input_dim - 1]
##                print("punished_reward_new",punished_reward)
#                punish_turning=False
##            print("idx_punish:",idx_punish)   
        reward_sum=reward_sum+reward
#            print("reward_sum:",reward_sum)    
        if collide==True or count/COUNT_FREQUENZ>episode_time or collide_finish==True: 
            
            if collide==True :
                
                reward=-pow(car.speed,4)

                
            car.impact()
            car.reset()
            car.set_start_direction(90)

            #print("neg_reward:",reward)
            reward_sum=reward_sum+reward
            running_reward=reward_sum
            #print("reward:",reward)
            #print("episode:",episode)
            
            print("FPS:",clock.get_fps())
            reward_sum=0
            dis_blue_sqr_sum=0
            dis_yellow_sqr_sum=0
            distance_set.append(distance)
            distance=0
            path_tag=[]
            path_tag_2=[]
            ta=0
            ta_2=0
            angle=0
            count=0
            collide=False
            collide_finish=False
            summary=True
#            print("reward:",reward)
        #RL.store_transition(observation, action, reward)
        
#            print("reward_sum:",reward_sum)
  
        observation_old=observation
    #print(episode)

    ##clock tick
    clock.tick_busy_loop(COUNT_FREQUENZ)
    ##clock tick
    
    ##update screen
    if Render==True:
        pygame.display.update()
    ##update screen