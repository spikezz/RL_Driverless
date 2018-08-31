# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:14:08 2018

@author: Asgard
"""
import tensorflow as tf
import os
import shutil
import sys, pygame, math
import player,maps,tracks,camera,traffic_cone , path
import canvas as cv
import calculation as cal
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from pygame.locals import *
from loader import load_image
from operator import itemgetter, attrgetter
from RL import PolicyGradient


LOAD = True
MODE = ['online', 'cycle']
n_model = 0



pygame.init()
screen = pygame.display.set_mode((1360,768),0)
pygame.display.set_caption('Karat Simulation')
font = pygame.font.Font(None, 40)


##black background for render when car is out of map 
background = pygame.Surface(screen.get_size())
background = background.convert_alpha()
background.fill((1, 1, 1))
##black background for render when car is out of map 


##the transparent canvas for drawing the necessary geometric relationship.
#the Zero point is at (cam.x,cam.y)
canvas = pygame.Surface(screen.get_size(),SRCALPHA ,32)
canvas = canvas.convert_alpha()
canvas.set_alpha(0)
##the transparent canvas for drawing the necessary geometric relationship.

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
# =============================================================================
# coneb=traffic_cone.cone(800,380,-1,car.x,car.y)
# coney=traffic_cone.cone(800,510,1,car.x,car.y)
# =============================================================================
coneb=traffic_cone.cone(CENTER[0],CENTER[1]-half_path_wide,-1,car.x,car.y)
coney=traffic_cone.cone(CENTER[0],CENTER[1]+half_path_wide,1,car.x,car.y)
startpoint=path.path(CENTER[0],CENTER[1],car.x,car.y)
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
##create the spriteGroup contains objects


##some initalize tracks are points left  while driving
tracks.initialize()
cam.set_pos(car.x, car.y)
##some initalize tracks are points left  while driving


##add car
player_s.add(car)
cone_s.add(coneb)
cone_s.add(coney)
path_s.add(startpoint)
list_cone_blue.append(coneb)
list_cone_yellow.append(coney)

list_path_point.append(startpoint)
##add car


##start angle from car
car.set_start_direction(90)
angle=0#the turning angle of the wheel 
##start angle from car





###initial Model of the car


##specification of the car
half_middle_axis_length=20
half_horizontal_axis_length=17
radius_of_wheel=10
el_length=500
##specification of the car


model=cv.initialize_model(CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)
###initial Model of the car


##count/COUNT_FREQUENZ is the real time
count=0 #every loop +1 for timer
COUNT_FREQUENZ=500#FPS Frame(loop times) per second
start_timer=False# switch for timer
##count/COUNT_FREQUENZ is the real time


##map picture size
FULL_TILE = 1000
##map picture size


##offset of the start position of the car
xc0=car.x
yc0=car.y
##offset of the start position of the car


##the old position for speed measurements
x_old=car.x
y_old=car.y
##the old position for speed measurements


##read the maps and draw
for tile_num in range (0, len(maps.map_tile)):
    
    #add submap idx to array
    maps.map_files.append(load_image(maps.map_tile[tile_num],False))

for x in range (0,7):
    
    for y in range (0, 20):
        
        #add submap to mapgroup
        map_s.add(maps.Map(maps.map_1[x][y], x * FULL_TILE, y * FULL_TILE))
        
##read the maps and draw
collide=False
        
##constant for cone
#cone_x=CENTER[0]
#cone_y=CENTER[1]
#position_sensor=[0,0]
i=0
p=0#yellow
q=0#blue
draw_blue_cone=[]
draw_yellow_cone=[]
dis_yellow=[]
dis_blue=[]
vektor_blue=[]
vektor_yellow=[]
dis_blue_sqr_sum=0
dis_yellow_sqr_sum=0
diff_sum_yb=0
##constant for cone

##constant for path
j=0
path_x=0
path_y=0
ctrl_pressed=False
dis_path=0
last_point=[startpoint.x,startpoint.y]
draw_path=[]
draw_path.append([0,0])
##constant for path

##constant for lidar
k=0
l=0
bound_lidar=CENTER[0]*2/5
##constant for lidar
##append draw element
draw_blue_cone.append([0,0])
draw_yellow_cone.append([0,0])
dis_blue.append(0)
dis_yellow.append(0)
vektor_blue.append([0,0])
vektor_yellow.append([0,0])
##append draw element

##state
state=[[],0,0,0]
##state

##konstant of RL
episode=0
ep_total=0
ep_use=0

speed_faktor=1
speed_faktor_enhance=1
angle_faktor_enhance=1
SD_input=0
MN_input=0
distance_faktor=0
safty_distance_impact=70
safty_distance_turning=75
collision_distance=40
distance=0
distance_set=[]
reward=1
reward_show=0
reward_sum=0
reward_mean=[]
rr=[]
rr_idx=0
running_reward =0
running_reward_max=0
reward_mean_max_rate=[]
vt=0

start_action=False
Render=False

input_max=70
action_n=7
rd= 0.9

lr =0.00001
lr_use=1
lr_reset=0
lr_set=[]
features_n=input_max

observation=np.zeros(input_max)
for t in range (0,input_max):
    observation[t]=0

action = 0
##konstant of RL


##PG init
RL = PolicyGradient(
    n_a=action_n,
    n_f=features_n,
    LR=lr,
    RD=rd,
    OG=False,
)
##PG init

##


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
    path_man.append([CENTER[0]-49*t,CENTER[1]-3*t])

for t in range (1,5):
    path_man.append([path_man[corner[0]-2][0]-49.74*t,path_man[corner[0]-2][1]-5*t])
    
for t in range (1,6):
    path_man.append([path_man[corner[1]-2][0]-49*t,path_man[corner[1]-2][1]-10*t])

for t in range (1,3):
    path_man.append([path_man[corner[2]-2][0]-45.82*t,path_man[corner[2]-2][1]-20*t])

path_man.append([path_man[corner[3]-2][0]-40,path_man[corner[3]-2][1]-30])
path_man.append([path_man[corner[4]-2][0]-31,path_man[corner[4]-2][1]-39.23])
path_man.append([path_man[corner[5]-2][0]-19.6,path_man[corner[5]-2][1]-46])
path_man.append([path_man[corner[6]-2][0]+15,path_man[corner[6]-2][1]-47.7])
path_man.append([path_man[corner[7]-2][0]+30,path_man[corner[7]-2][1]-40])
path_man.append([path_man[corner[8]-2][0]+35,path_man[corner[8]-2][1]-35.71])
for t in range (1,6):
    path_man.append([path_man[corner[9]-2][0]+43.5*t,path_man[corner[9]-2][1]-24.65*t])
    
for t in range (1,23):
    path_man.append([path_man[corner[10]-2][0]+47.37*t,path_man[corner[10]-2][1]-16*t])
    

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
    
saver = tf.train.Saver()
di = './'+MODE[n_model]
###main loop process
if LOAD:
    saver.restore(RL.sess, tf.train.latest_checkpoint(di))
else:
    RL.sess.run(tf.global_variables_initializer())

###main loop process
        
while True:

    if episode<1:
        #show1=len(list_cone_yellow)
        ##key event continually
        
        
        keys = pygame.key.get_pressed()
        
    
        if keys[K_1]:
    
            angle=45
            
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
          
            angle=-45
            
        if keys[K_LEFT]:
            
            if angle<0:
                
                angle=-1
                
            if angle<46:
                
                angle=angle+car.steering
    
            #if angle==0:
               
        if keys[K_RIGHT]:
            
            if angle>0:
                
                angle=1
            
            if angle>-46:
                
                angle=angle-car.steering
       
            #if angle==0:
               
        if keys[K_UP]:
            
            car.accelerate()
            
    
            
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
            
            # quit for windows
            if event.type == QUIT:
                        
                pygame.quit()
                
                sys.exit()
    
                            
            if event.type == KEYDOWN :
                
                # quit for esc key
                if event.key == K_ESCAPE:  
                                
                    pygame.quit()
                    
                    sys.exit()
                    
                #timer
                if event.key == K_SPACE :  
                    
                    if start_timer==False: 
                        
                        start_timer=True
                        
                    else: 
                        
                        start_timer=False
                        
                if event.key ==K_RETURN:
                    
                    start_action=True
                    
                    
                if event.key ==K_BACKSPACE:
                    
                    car.reset()
                    car.set_start_direction(90)
                
                if event.key == K_r:
                    
                    if  Render==False: 
                        
                        Render=True
                        
                    else: 
                        
                        Render=False
                    
                if event.key == K_e:
                    
                    ep_use=0
                    lr_reset=lr_reset+1
                    
                if event.key == K_d:
                    
                    lr=lr/10
                    print("max lr:",lr)

                if event.key == K_m:

                    lr=lr*10
                    print("max lr:",lr)
                
                if event.key == K_t:

                    if  RL.deterministic==False: 
                        
                        RL.deterministic=True
                        
                    else: 
                        
                        RL.deterministic=False
                        

                    
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
#             elif action==3:
#                 
#                 if angle<0:
#                     
#                     angle=-1
#                     
#                 if angle<46:
#                     
#                     angle=angle+car.steering
#                     
#                 car.deaccelerate()
#                 
#             elif action==4:
#                 
#                 if angle>0:
#                     
#                     angle=1
#                 
#                 if angle>-46:
#                     
#                     angle=angle-car.steering
#                 car.deaccelerate()
#                 
#             elif action==5:
#                 angle=0
#                 car.deaccelerate()
# =============================================================================
        ##reward
        
                ##
        
                
        distance=distance+car.speed
        #reward=speed_faktor*car.speed
            
        car.soften()
        #camera position reset
        cam.set_pos(car.x, car.y)
        
        
        #speed calculation for center of the car
        speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s
        
    
        #update the old position
        x_old=car.x
        y_old=car.y
        

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
        car.wheelangle=angle
        model=cv.turning(model,angle,CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)
    
        
        if angle>0 :
        
            car.rrl=model[19]
            car.steerleft(angle)
            #vektor_speed=[car.speed*,car.speed*]
           
        elif angle<0  :
    
            car.rrr=model[21]
            car.steerright(angle)
        
    
        model=cv.rotate(model,CENTER,car.dir)
        
        for i in range (0, p+1):
            
            dis_yellow[i]=cal.calculate_r((car.x,car.y),(list_cone_yellow[i].x-model[7][0][0],list_cone_yellow[i].y-model[7][0][1]))
            draw_yellow_cone[i]=[list_cone_yellow[i].x-cam.x,list_cone_yellow[i].y-cam.y]
        
        for i in range (0, q+1):
            
            dis_blue[i]=cal.calculate_r((list_cone_blue[i].x-model[7][0][0],list_cone_blue[i].y-model[7][0][1]),(car.x,car.y))
            draw_blue_cone[i]=[list_cone_blue[i].x-cam.x,list_cone_blue[i].y-cam.y]         
    
        for i in range (0, j+1):
            
            draw_path[i]=[list_path_point[i].x-cam.x,list_path_point[i].y-cam.y]
            
        if start_action==True:
            
        #if Render==True:
           
            
            start_timer=True
            #print("observation:",observation)
            action=RL. choose_action(observation)
            #print("action:",action)
            
            if action==0:
                
                #angle=0
                car.accelerate()
                

            elif action==1:    
                 
                if angle<0:
                     
                    angle=-1
                     
                if angle<46:
                     
                    angle=angle+car.steering
                    
                if car.speed<=0:
                    
                    car.accelerate()
                    
            elif action==2:
                 
                if angle>0:
                     
                    angle=1
                 
                if angle>-46:
                     
                    angle=angle-car.steering
                
                if car.speed<=0:
                    
                    car.accelerate()
                    
            elif action==3:
                
                if angle<0:
                    
                    angle=-1
                    
                if angle<46:
                    
                    angle=angle+car.steering
                    
                car.accelerate()
                
            elif action==4:
                
                if angle>0:
                    
                    angle=1
                
                if angle>-46:
                    
                    angle=angle-car.steering
                    
                car.accelerate()
            
            elif action==5:     

                car.deaccelerate()

            elif action==6:
                
                pass
            
            for i in range (0, q+1):
            
                if dis_blue[i]<safty_distance_turning:
                    
                    if dis_blue[i]<safty_distance_impact:
                        angle=45
                        #car.deaccelerate()
                        action=1
                        break
                  
                    if angle<0:
                         
                        angle=-1
                         
                    if angle<46:
                         
                        angle=angle+car.steering
                        
                    if car.speed<=0:
                        
                        car.accelerate()
                
                    action=1
                    break
            
            for i in range (0, p+1):
            
                if dis_yellow[i]<safty_distance_turning:
                    
                    if dis_yellow[i]<safty_distance_impact:
                        angle=-45
                        #car.deaccelerate()
                        action=2
                        break
                     
                    if angle>0:
                         
                        angle=1
                     
                    if angle>-46:
                         
                        angle=angle-car.steering
                    
                    if car.speed<=0:
                        
                        car.accelerate()
                        
                    #car.deaccelerate()
                    action=2
                    break

        ##start drawing
        
        
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
        if Render==True:
            
            cone_s.draw(screen)
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
    
        vektor_speed=[speed*math.cos(math.radians(270-angle)),car.speed*COUNT_FREQUENZ*math.sin(math.radians(270-angle))]
        ##
        
        
        ##
        text_yellow= font.render('vektor_yellow: ' + '( '+str(round(float(vektor_yellow[0][0]),2))+' , '+str(round(float(vektor_yellow[0][1]),2))+' )', 1, (0, 0, 102))   
        textpos_yellow = text_yellow.get_rect(centery=425, left=20)
        
        text_blue= font.render('vektor_blue: '+ '( ' + str(round(float(vektor_blue[0][0]),2))+' , '+str(round(float(vektor_blue[0][1]),2))+' )', 1, (0, 0, 102))   
        textpos_blue = text_blue.get_rect(centery=465, left=20)
        
        text_speed_v= font.render('vektor_speed: ' +  '( ' + str(round(float(vektor_speed[0]),2))+' , '+str(round(float(vektor_speed[1]),2))+' )', 1, (0, 0, 102))   
        textpos_speed_v = text_speed_v.get_rect(centery=505, left=20)
        ##
        
        
        #anything want to show
        text_show1= font.render('distance: ' + str(round(float(distance),2)), 1, (0, 0, 102))   
        textpos_show1 = text_dir.get_rect(centery=545, left=20)
        
        text_show2= font.render('reward: ' + str(round(float(reward_show),2)), 1, (0, 0, 102))   
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
        
        ##draw lines to find center of the car
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
        for i in range (0, q+1):
            
            if dis_blue[i]<bound_lidar:
                
                pygame.draw.line(canvas, (0,255,255), model[7][0],draw_blue_cone[i],2)
            
        for i in range (0, p+1):
            
            if dis_yellow[i]<bound_lidar:
                
                pygame.draw.line(canvas, (255,255,0), model[7][0],draw_yellow_cone[i],2)
        ##draw distance to cones
        
        
        ##draw path
        for i in range (1, j+1):
            
            pygame.draw.line(canvas, (0,255,0), draw_path[i-1],draw_path[i],2)
        ##draw path
        
        ##draw front axis extension
        if angle >= 2.3 or angle==0:
            
            pygame.draw.line(canvas, (255,255,102), model[5][0],model[14][0],2)#frontwheel turing axis
       
        if angle <= -2.3 or angle==0:
            
            pygame.draw.line(canvas, (255,255,102), model[4][0],model[15][0],2)#frontwheel turing axis
        
        ##draw front axis extension
        
        
        ##draw  the circle which car moving along in canvas
        if  angle>0:
            
            pygame.draw.arc(canvas, (255,255,102), (model[18][0][0]-model[19],model[18][0][1]-model[19],2*model[19],2*model[19]), 0, 360, 3)
        
        if  angle<0:
            
            pygame.draw.arc(canvas, (255,255,102), (model[20][0][0]-model[21],model[20][0][1]-model[21],2*model[21],2*model[21]), 0, 360, 3)
        
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
            screen.blit(text_yellow, textpos_yellow)
            screen.blit(text_blue, textpos_blue)
            screen.blit(text_speed_v, textpos_speed_v)
            screen.blit(text_show1, textpos_show1)
            screen.blit(text_show2, textpos_show2)
        ##show text
        
        
        ##start drawing
        
        ##interface for RL 
        vektor_blue_temp=[]
        vektor_yellow_temp=[]
    
        for i in range (0, q+1):
            
            if dis_blue[i]<bound_lidar:
                
                vektor_blue_temp.append(vektor_blue[i])
                dis_blue_sqr_sum=dis_blue_sqr_sum+pow(dis_blue[i],2)
                
        for i in range (0, p+1):
            
            if dis_yellow[i]<bound_lidar:
                
                vektor_yellow_temp.append(vektor_yellow[i])
                dis_yellow_sqr_sum=dis_yellow_sqr_sum+pow(dis_yellow[i],2)
                

        
        k=len(vektor_blue_temp)
        l=len(vektor_yellow_temp)
        if k>0 and l>0:
            diff_sum_yb=math.sqrt(pow((math.sqrt(dis_blue_sqr_sum/k)-math.sqrt(dis_yellow_sqr_sum/l)),2))
            #print("diff:",diff_sum_yb)
            state_sort=np.vstack((np.vstack(vektor_blue_temp),np.vstack(vektor_yellow_temp)))
            state_sort_temp=[]
            state_sort_end=[]
            
            for i in range (0, k+l):
              
               state_sort_temp.append([cal.calculate_sita_r(state_sort[i],[0,0]),state_sort[i]])
        
            state_sort=sorted(state_sort_temp)
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #print (state_sort)
            
            for i in range (0, k+l):
                
                state_sort_end.append(state_sort[i][1])
                
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            state[0]=state_sort_end    
            state[3]=np.vstack(state[0]).ravel()
            state[1]=np.vstack(vektor_speed).ravel()
            state[2]=angle
            state_input=np.hstack((state[1]*speed_faktor_enhance,state[2]*angle_faktor_enhance,state[3]))
            #print (state_input)
            #print ('size:',state_input.size)
            SD_input=np.std(state_input)
            MN_input=np.mean(state_input)
            state_input=(state_input-MN_input)/SD_input
            for t in range(len(state_input)):
                observation[t]=state_input[t]
    
        #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')      
        
        ##timer 
        if start_timer==True:
    
            count=count+1
        ##timer 
        #reward=distance_faktor*distance
        if start_action==True:
            
            if math.sqrt(diff_sum_yb)>1:
                
                reward=car.speed*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/math.sqrt(diff_sum_yb))
                
            else:
            
                reward=car.speed*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/1)
            #reward=car.speed*speed_faktor+distance_faktor*distance
            
            reward_sum=reward_sum+reward

            
            #if pygame.sprite.spritecollide(car, cone_s, False) or count/COUNT_FREQUENZ>1.8 :
            for i in range (0, q+1):
            
                if dis_blue[i]<collision_distance:
                    collide=True
                    
            for i in range (0, p+1):
            
                if dis_yellow[i]<collision_distance:
                    collide=True
                    
            if collide==True or count/COUNT_FREQUENZ>1.5: 
            #if pygame.sprite.spritecollide(car, cone_s, False) : 
                if collide==True:
                    reward=-pow(car.speed,3)
                car.impact()
                car.reset()
                car.set_start_direction(90)

                #print("neg_reward:",reward)
                reward_sum=reward_sum+reward
                reward_show=reward_sum
                #print("reward:",reward)
                #print("episode:",episode)
                
                print("FPS:",clock.get_fps())
                reward_sum=0
                dis_blue_sqr_sum=0
                dis_yellow_sqr_sum=0
                distance_set.append(distance)
                distance=0
                angle=0
                count=0
                episode=episode+1
                
            elif count/COUNT_FREQUENZ>4 :
                
                car.reset()
                car.set_start_direction(90)
                 #print("neg_reward:",reward)
                reward_sum=reward_sum+reward
                reward_show=reward_sum
                #print("reward:",reward)
                #print("episode:",episode)
                
                print("FPS:",clock.get_fps())
                reward_sum=0
                dis_blue_sqr_sum=0
                dis_yellow_sqr_sum=0
                distance_set.append(distance)
                distance=0
                angle=0
                count=0
                episode=episode+1
                
            RL.store_transition(observation, action, reward)

        #print(episode)

        ##clock tick
        clock.tick_busy_loop(COUNT_FREQUENZ)
        ##clock tick
        
        ##update screen
        if Render==True:
            pygame.display.update()
        ##update screen
        
    else:
        #print("RL.r_set:",RL.r_set)
        rs_sum = sum(RL.r_set)
        
        #if 'running_reward' not in globals():
            
            #running_reward = rs_sum

        #else:
            #running_reward = running_reward * 0.01 + rs_sum * 0.99
            
        running_reward = rs_sum
        
        if running_reward_max<running_reward and ep_total>1:
            
            running_reward_max=running_reward
            ep_use=0
            lr_reset=lr_reset+1
            #RL.learning_rate=lr*10
            
        #else:
            #RL.learning_rate=lr
            #lr=1
            #deterministic_count=0
            #RL.learning_rate=0.1
            #ep_use=0
            #lr_reset=lr_reset+1
            
        #if deterministic_count<0:
        
            #RL.deterministic=True
            #deterministic_count=deterministic_count+1
        #else:
            
            #RL.deterministic=False
            #RL.learning_rate=0.01
        print("running_reward:",running_reward)
        print("max_running_reward:",running_reward_max)
        
        rr.append(running_reward)   
        rr_idx=rr_idx+1
        reward_mean.append(sum(rr)/rr_idx)
        
        print("reward_mean:",reward_mean[rr_idx-1])
        if rr_idx>5:
            
            reward_mean_max_rate.append(running_reward_max/reward_mean[rr_idx-1])
        #if RL.learning_rate>0.001:

        print("lr_reset:",lr_reset)
        #RL.learning_rate=0.3/(ep_use+3000)
        lr_set.append(RL.learning_rate)
        print("learning rate:",RL.learning_rate)
        print("max lr:",lr)
        print("deterministic:",RL.deterministic)
        #print("deterministic_count:",deterministic_count)
        #print("rr:",rr)
        if Render==False:
            
            vt=RL.learn(car.maxspeed,car.acceleration)
            
        else:
            
            RL.ob_set,RL.a_set,RL.r_set=[],[],[]  
        #print("RL.learn:",vt)

        plt.subplot(431)
        plt.plot(rr)  
        plt.xlabel('episode steps')
        plt.ylabel('runing reward')

        plt.subplot(432)
        plt.plot(vt)    # plot the episode vt
        plt.xlabel('episode steps')
        plt.ylabel('normalized state-action value')

        
     
        plt.subplot(433)
        plt.plot(reward_mean)  
        plt.xlabel('episode steps')
        plt.ylabel('reward_mean')
        
        plt.subplot(434)
        plt.plot(RL.prob0)  
        plt.xlabel('episode steps')
        plt.ylabel('prob0')
        
        plt.subplot(435)
        plt.plot(RL.prob1)  
        plt.xlabel('episode steps')
        plt.ylabel('prob1')
        
        plt.subplot(436)
        plt.plot(RL.prob2)  
        plt.xlabel('episode steps')
        plt.ylabel('prob2')
        
        plt.subplot(437)
        plt.plot(RL.prob3)  
        plt.xlabel('episode steps')
        plt.ylabel('prob3')
        
        plt.subplot(438)
        plt.plot(RL.prob4)  
        plt.xlabel('episode steps')
        plt.ylabel('prob4')
        
        plt.subplot(439)
        plt.plot(RL.prob5)  
        plt.xlabel('episode steps')
        plt.ylabel('prob5')
        
        plt.subplot(4,3,10)
        plt.plot(RL.prob6)  
        plt.xlabel('episode steps')
        plt.ylabel('prob6')
        
                
        plt.subplot(4,3,11)
        plt.plot(lr_set)  
        plt.xlabel('episode steps')
        plt.ylabel('learning rate')
        
        plt.subplot(4,3,12)
        plt.plot(reward_mean_max_rate)  
        plt.xlabel('episode steps')
        plt.ylabel('reward Max/mean')
# =============================================================================
#         plt.subplot(4,3,11)
#         plt.plot(distance_set)  
#         plt.xlabel('episode steps')
#         plt.ylabel('distance_set')
#       
# =============================================================================
        plt.show()
        
        RL.prob0=[]
        RL.prob1=[]
        RL.prob2=[]
        RL.prob3=[]
        RL.prob4=[]
        RL.prob5=[]
        RL.prob6=[]
        
        
        ep_total=ep_total+1
        print("totaol train:",ep_total)
        ep_use=ep_use+1
        print("lr ep :",ep_use)
        episode=0
        
        if os.path.isdir(di): shutil.rmtree(di)
        os.mkdir(di)
        ckpt_path = os.path.join('./'+MODE[n_model], 'PG.ckpt')
        save_path = saver.save(RL.sess, ckpt_path, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)
###main loop process