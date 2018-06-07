# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:14:08 2018

@author: Asgard
"""
import sys, pygame, math
import player,maps,tracks,camera,traffic_cone
import canvas as cv
import calculation as cal
from pygame.locals import *
from loader import load_image


pygame.init()
screen = pygame.display.set_mode((pygame.display.Info().current_w,pygame.display.Info().current_h),pygame.FULLSCREEN)
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


##create some objects
clock = pygame.time.Clock()
car = player.Player()
cam = camera.Camera()
# =============================================================================
# coneb=traffic_cone.cone(800,380,-1,car.x,car.y)
# coney=traffic_cone.cone(800,510,1,car.x,car.y)
# =============================================================================
coneb=traffic_cone.cone(680,314,-1,car.x,car.y)
coney=traffic_cone.cone(680,444,1,car.x,car.y)
##create some objects


##create the spriteGroup contains objects
list_cone_yellow=[]
list_cone_blue=[]
map_s= pygame.sprite.Group()
player_s= pygame.sprite.Group()
tracks_s= pygame.sprite.Group()
cone_s  = pygame.sprite.Group()
##create the spriteGroup contains objects


##some initalize tracks are points left  while driving
tracks.initialize()
cam.set_pos(car.x, car.y)
##some initalize tracks are points left  while driving


##add car
player_s.add(car)
cone_s.add(coneb)
cone_s.add(coney)
list_cone_blue.append(coneb)
list_cone_yellow.append(coney)
##add car


##start angle from car
car.set_start_direction(90)
angle=0#the turning angle of the wheel 
##start angle from car


#testcode for shell
#CENTER_X = 800
#CENTER_Y = 450


##find the center of screen
CENTER_X =  float(pygame.display.Info().current_w /2)
CENTER_Y =  float(pygame.display.Info().current_h /2)
CENTER=(CENTER_X,CENTER_Y)
##find the center of screen


###initial Model of the car


##specification of the car
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
count=0 #every loop +1 for timer
COUNT_FREQUENZ=200#FPS Frame(loop times) per second
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
       
        
##constant for cone
cone_x=CENTER[0]
cone_y=CENTER[1]
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
##constant for cone

##append draw element
draw_blue_cone.append([0,0])
draw_yellow_cone.append([0,0])
dis_blue.append(0)
dis_yellow.append(0)
vektor_blue.append([0,0])
vektor_yellow.append([0,0])
##append draw element

##state
state=[0,0,0,0]
##state

###main loop process
        
while True:
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
            
           
                
        if event.type == KEYUP :    
            
            if event.key == K_LEFT or event.key == K_RIGHT:
                
                angle=0
# =============================================================================
#             
#             if event.key == K_a : 
#                 
#                     i=i+1
#                     
#             if event.key == K_s : 
#                 
#                 if (i<(len(list_cone_yellow)) and i<(len(list_cone_blue))):
#                     
#                     show2=cal.calculate_r((list_cone_blue[i].x,list_cone_blue[i].y),(list_cone_yellow[i].x,list_cone_yellow[i].y))
#                     dis_yellow=cal.calculate_r((car.x,car.y),(list_cone_yellow[i].x,list_cone_yellow[i].y))
#                     dis_blue=cal.calculate_r((list_cone_blue[i].x,list_cone_blue[i].y),(car.x,car.y))
#                     
#                 
# =============================================================================
                
        if event.type == MOUSEBUTTONDOWN:
            
            if start_timer==False: 
                    
                start_timer=True
                    
            else: 
                    
                start_timer=False
                
            if pygame.mouse.get_pressed()==(True,False,False):
                
                cone_x, cone_y = pygame.mouse.get_pos()
                cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
                list_cone_yellow.append(cone_new)
                cone_s.add(cone_new)
                
                draw_yellow_cone.append([0,0])
                dis_yellow.append(0)
                vektor_yellow.append([0,0])
                p=p+1
                
            elif pygame.mouse.get_pressed()==(False,False,True):
                
                cone_x, cone_y = pygame.mouse.get_pos()
                cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
                list_cone_blue.append(cone_new)
                cone_s.add(cone_new)
               
                draw_blue_cone.append([0,0])
                dis_blue.append(0)
                vektor_blue.append([0,0])
                q=q+1
                
    ##system event
    
    show1=len(list_cone_yellow)
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
        
        turing_speed=1
        
        if angle<0:
            
            angle=-1
            
        if angle<46:
            
            angle=angle+turing_speed

        #if angle==0:
           
    if keys[K_RIGHT]:
        
        turing_speed=-1
        
        if angle>0:
            
            angle=1
        
        if angle>-46:
            
            angle=angle+turing_speed
   
        #if angle==0:
           
    if keys[K_UP]:
        
        car.accelerate()
        

        
    if keys[K_DOWN]:
        
        car.deaccelerate()
        
    if keys[K_BACKSPACE]:
        
        car.speed=0
        
# =============================================================================
#     if keys[K_DELETE]:
#         
#         cone_s.empty()
#         list_cone_blue=[]
#         list_cone_yellow=[]
# =============================================================================
    ##key event 
    

    
    car.soften()
    #camera position reset
    cam.set_pos(car.x, car.y)
    
    
    #speed calculation for center of the car
    speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s

    #update the old position
    x_old=car.x
    y_old=car.y
    
    ##
    for i in range (0, p+1):
        
        dis_yellow[i]=cal.calculate_r((car.x,car.y),(list_cone_yellow[i].x-CENTER[0],list_cone_yellow[i].y-CENTER[1]))
        draw_yellow_cone[i]=[list_cone_yellow[i].x-cam.x,list_cone_yellow[i].y-cam.y]
    
    for i in range (0, q+1):
        
        dis_blue[i]=cal.calculate_r((list_cone_blue[i].x-CENTER[0],list_cone_blue[i].y-CENTER[1]),(car.x,car.y))
        draw_blue_cone[i]=[list_cone_blue[i].x-cam.x,list_cone_blue[i].y-cam.y]         

    #postion_sensor=[(model[7][0][0]-CENTER[0]+car.x),(model[7][0][1]-CENTER[1]+car.y)]
    ##
       
    ##text setting
    
    text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (0, 0, 102))
    textpos_fps = text_fps.get_rect(centery=25, left=20)
    
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
    
    text_dis_yellow= font.render('distance to yellow cone: ' + str(round(float(dis_yellow[0]),2)), 1, (0, 0, 102))   
    textpos_dis_yellow = text_dis_yellow.get_rect(centery=345, left=20)
    
    text_dis_blue= font.render('distance to blue cone: ' + str(round(float(dis_blue[0]),2)), 1, (0, 0, 102))   
    textpos_dis_blue =text_dis_blue.get_rect(centery=385, left=20)
    
    
    
    
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
    

    ##start drawing
    
    
    ##draw background
    screen.blit(background, (0,0))
    ##
    
    
    ##
    map_s.update(cam.x, cam.y)
    map_s.draw(screen)
    ##
    
        
    cone_s.update(cam.x, cam.y)
    cone_s.draw(screen)
    
    
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
    text_show1= font.render('vektor_speed_x: ' + str(round(float(vektor_speed[0]),2)), 1, (0, 0, 102))   
    textpos_show1 = text_dir.get_rect(centery=545, left=20)
    
    text_show2= font.render('vektor_speed_y: ' + str(round(float(vektor_speed[1]),2)), 1, (0, 0, 102))   
    textpos_show2 = text_dir.get_rect(centery=585, left=20)
    #anything want to show
    
    
    
    ##
    player_s.update(cam.x, cam.y)
    player_s.draw(screen)
    ##
    
    
    ##
    tracks_s.add(tracks.Track(cam.x + CENTER[0] , cam.y + CENTER[1], car.dir))
    tracks_s.update(cam.x, cam.y)
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
    
    for i in range (0, q+1):
    
        pygame.draw.line(canvas, (0,255,255), model[7][0],draw_blue_cone[i],2)
        
    for i in range (0, p+1):
        
        pygame.draw.line(canvas, (255,255,0), model[7][0],draw_yellow_cone[i],2)
    
    ##draw front axis extension
    if angle >= 2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), model[5][0],model[14][0],2)#frontwheel turing axis
   
    if angle <= -2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), model[4][0],model[15][0],2)#frontwheel turing axis
    
    ##draw front axis extension
    
    
    ##draw  the car moving circle in canvas
    if  angle>0:
        
        pygame.draw.arc(canvas, (255,255,102), (model[18][0][0]-model[19],model[18][0][1]-model[19],2*model[19],2*model[19]), 0, 360, 3)
    
    if  angle<0:
        
        pygame.draw.arc(canvas, (255,255,102), (model[20][0][0]-model[21],model[20][0][1]-model[21],2*model[21],2*model[21]), 0, 360, 3)
    
    ##draw  the car moving circle in canvas
    
    
    ##show canvas
    screen.blit(canvas, (0,0))
    ##show canvas
    
    
    ##show text
    screen.blit(text_fps, textpos_fps)
    screen.blit(text_timer, textpos_timer)
    screen.blit(text_loop, textpos_loop)
    screen.blit(text_pos, textpos_pos)
    screen.blit(text_posr, textpos_posr)
    screen.blit(text_dir, textpos_dir)
    screen.blit(text_speed, textpos_speed)
    screen.blit(text_colour, textpos_colour)
    screen.blit(text_dis_yellow, textpos_dis_yellow)
    screen.blit(text_dis_blue, textpos_dis_blue)
    screen.blit(text_yellow, textpos_yellow)
    screen.blit(text_blue, textpos_blue)
    screen.blit(text_speed_v, textpos_speed_v)
    screen.blit(text_show1, textpos_show1)
    screen.blit(text_show2, textpos_show2)
    ##show text
    
    
    ##start drawing
    
    ##interface for RL 
    state[0]=vektor_blue
    state[1]=vektor_yellow
    state[2]=vektor_speed
# =============================================================================
#     action=(action_turning,action_accelerate)
#     turing_speed=action[0]
#     car.acceleration=action[1]
# =============================================================================
    ##interface for RL 
    
    
    
    ##timer 
    if start_timer==True:

        count=count+1
    ##timer 
    
    ##clock tick
    clock.tick_busy_loop(COUNT_FREQUENZ)
    ##clock tick
    
    ##update screen
    pygame.display.update()
    ##update screen
    
    
###main loop process