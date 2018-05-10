# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:14:08 2018

@author: Asgard
"""
import sys, pygame, math
import player,maps,tracks,camera
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
##create some objects


##create the spriteGroup contains objects
map_s= pygame.sprite.Group()
player_s= pygame.sprite.Group()
tracks_s  = pygame.sprite.Group()
##create the spriteGroup contains objects


##some initalize tracks are points left  while driving
tracks.initialize()
cam.set_pos(car.x, car.y)
##some initalize tracks are points left  while driving


##add car
player_s.add(car)
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
##constant of the car


##bwel/r:back,wheel,extension,left/right end point
#the extension line(end point) for finding the center of the circle kurve, which the car obey when turning
bwel=(CENTER[0]-el_length ,CENTER[1]+half_middle_axis_length)
bwer=(CENTER[0]+el_length ,CENTER[1]+half_middle_axis_length)
##bwel/r:back,wheel,extension,left/right end point


##R_bwel:R_ radius from the point to the center of the screen
R_bwel=cal.calculate_r(bwel,CENTER)
R_bwer=cal.calculate_r(bwer,CENTER)
##R_bwel:R_ radius from the point to the center of the screen

##sita_bwel:the angle between car.dir=0 and the point
sita_bwel=cal.calculate_sita(1,bwel,CENTER)
sita_bwer=cal.calculate_sita(1,bwer,CENTER)
##sita_bwel:the angle between car.dir=0 and the point


##f:front
#the extension line(end point) of the front wheel.fwel is for front right wheel,fwer is for front left wheel
fwel=(CENTER[0]-el_length ,CENTER[1]-half_middle_axis_length)
fwer=(CENTER[0]+el_length ,CENTER[1]-half_middle_axis_length)
##f:front


##
R_fwel=cal.calculate_r(fwel,CENTER)
R_fwer=cal.calculate_r(fwer,CENTER)
##


##
sita_fwel=cal.calculate_sita(0,fwel,CENTER)
sita_fwer=cal.calculate_sita(0,fwer,CENTER)
##


##rpl:round(circle) posion left(center); rrl:round(circle) radius left(radius)
#find the center/radius of the circle kurve, which the car obey when turning
rpl=(CENTER[0]+half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(5))),CENTER[1]+half_middle_axis_length)
R_rpl=cal.calculate_r(rpl,CENTER)
sita_rpl=cal.calculate_sita(1,rpl,CENTER)
rrl=2*half_middle_axis_length/math.sin(math.radians(5))
##


##rpr:round(circle) posion right(center); rrr:round(circle) radius right(radius)
#find the center/radius of the circle kurve, which the car obey when turning
rpr=(CENTER[0]-half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(-5))),CENTER[1]+half_middle_axis_length)
R_rpr=cal.calculate_r(rpr,CENTER)
sita_rpr=cal.calculate_sita(1,rpr,CENTER)     
rrr=-2*half_middle_axis_length/math.sin(math.radians(-5))
##


##fl/rb/t:front left/right bottom/top(point);
flb=(CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length+radius_of_wheel)
flt=(CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length-radius_of_wheel)
frb=(CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length+radius_of_wheel)
frt=(CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length-radius_of_wheel)
##fl/rb/t:front left/right bottom/top(point);


##
R_flb=cal.calculate_r(flb,CENTER)
R_flt=cal.calculate_r(flt,CENTER)
R_frb=cal.calculate_r(frb,CENTER)
R_frt=cal.calculate_r(frt,CENTER)
##


##
sita_flb=cal.calculate_sita(1,flb,CENTER) 
sita_flt=cal.calculate_sita(1,flt,CENTER) 
sita_frb=cal.calculate_sita(1,frb,CENTER) 
sita_frt=cal.calculate_sita(1,frt,CENTER) 
##


##fal/r:front axis left/right
fal=(CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length)
far=(CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length)
##fal/r:front axis left/right


##
R_fal=cal.calculate_r(fal,CENTER)
R_far=cal.calculate_r(far,CENTER)
##


##
sita_fal=cal.calculate_sita(0,fal,CENTER) 
sita_far=cal.calculate_sita(0,far,CENTER) 
##


##mab/t:middle axis bottom/top
mab=(CENTER[0],CENTER[1]+half_middle_axis_length)
mat=(CENTER[0],CENTER[1]-half_middle_axis_length)
##mab/t:middle axis bottom/top


##
R_mab=cal.calculate_r(mab,CENTER)
R_mat=cal.calculate_r(mat,CENTER)
##


##
sita_mab=cal.calculate_sita(1,mab,CENTER) 
sita_mat=cal.calculate_sita(0,mat,CENTER)
##


##bal/r:back axis left/right
bal=(CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length)
bar=(CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length)
##bal/r:back axis left/right


##
R_bal=cal.calculate_r(bal,CENTER)
R_bar=cal.calculate_r(bar,CENTER)
##

##
sita_bal=cal.calculate_sita(1,bal,CENTER) 
sita_bar=cal.calculate_sita(1,bar,CENTER) 
##


##bl/rb/t:back left/right bottom/top(point);
blb=(CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length+radius_of_wheel)
blt=(CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length-radius_of_wheel)
brb=(CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length+radius_of_wheel)
brt=(CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length-radius_of_wheel)
##bl/rb/t:back left/right bottom/top(point);


##
R_blb=cal.calculate_r(blb,CENTER)
R_blt=cal.calculate_r(blt,CENTER)
R_brb=cal.calculate_r(brb,CENTER)
R_brt=cal.calculate_r(brt,CENTER)
##


##
sita_blb=cal.calculate_sita(1,blb,CENTER) 
sita_blt=cal.calculate_sita(1,blt,CENTER) 
sita_brb=cal.calculate_sita(1,brb,CENTER) 
sita_brt=cal.calculate_sita(1,brt,CENTER)    
##


##fl/rwx/y:x/y of front left/right wheel
CENTER_flwx=CENTER[0]-half_horizontal_axis_length
CENTER_flwy=CENTER[1]-half_middle_axis_length
CENTER_frwx=CENTER[0]+half_horizontal_axis_length
CENTER_frwy=CENTER[1]-half_middle_axis_length
CENTER_flw=(CENTER_flwx,CENTER_flwy)
CENTER_frw=(CENTER_frwx,CENTER_frwy)
##fl/rwx/y:x/y of front left/right wheel

###initial Model of the car


##count/COUNT_FREQUENZ is the real time
count=0 #every loop +1 for timer
COUNT_FREQUENZ=10#FPS Frame(loop times) per second
start_timer=False# switch for timer
##count/COUNT_FREQUENZ is the real time


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
        map_s.add(maps.Map(maps.map_1[x][y], x * 1000, y * 1000))
        
##read the maps and draw
        
        
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
    ##system event
    

    ##key event     
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
            
            angle=angle+1

        #if angle==0:
           
    if keys[K_RIGHT]:
       
        if angle>0:
            
            angle=1
        
        if angle>-46:
            
            angle=angle-1
   
        #if angle==0:
           
    if keys[K_UP]:
        
        car.accelerate()
        
    else:
        
        car.soften()
        
    if keys[K_DOWN]:
        
        car.deaccelerate()
        
    if keys[K_BACKSPACE]:
        
        car.speed=0
        
    if keys[K_DELETE]:
        
        pass
    
    ##key event 
    
    #camera position reset
    cam.set_pos(car.x, car.y)
    
    
    #speed calculation for center of the car
    speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s

    #update the old position
    x_old=car.x
    y_old=car.y
    
    
    #car.grass(screen.get_at(((int(CENTER_W-5), int(CENTER_H-5)))).g)
    
    ##text setting
    
    text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (0, 0, 102))
    textpos_fps = text_fps.get_rect(centery=25, left=20)
    
    text_timer = font.render('Timer: ' + str(round(float(count/COUNT_FREQUENZ),2)) +'s', 1, (0, 0, 102))
    textpos_timer = text_fps.get_rect(centery=65, left=20)
     
    text_loop = font.render('Loop: ' + str(int(clock.get_time())) +'ms', 1, (0, 0, 102))
    textpos_loop = text_fps.get_rect(centery=105, left=20)
    
    text_posx= font.render('POS_X: ' + str(round(float(car.x-xc0),2)), 1, (0, 0, 102))   
    textpos_posx = text_posx.get_rect(centery=145, left=20)
   
    text_posy= font.render('POS_Y: ' + str(round(float(car.y-yc0),2)), 1, (0, 0, 102))   
    textpos_posy = text_posy.get_rect(centery=185, left=20)
    
    text_posxr= font.render('POS_X_RAW: ' + str(round(float(car.x),2)), 1, (0, 0, 102))   
    textpos_posxr = text_posxr.get_rect(centery=225, left=20)
   
    text_posyr= font.render('POS_Y_RAW: ' + str(round(float(car.y),2)), 1, (0, 0, 102))   
    textpos_posyr = text_posyr.get_rect(centery=265, left=20)
    
    text_dir= font.render('Direction: ' + str(round(float(car.dir),2)), 1, (0, 0, 102))   
    textpos_dir = text_dir.get_rect(centery=305, left=20)
    
    text_speed= font.render('speed: ' + str(round(float(speed),2))+'|'+str(round(float(car.speed),2))+'pixel/s', 1, (0, 0, 102))   
    textpos_speed = text_dir.get_rect(centery=345, left=20)
    
    text_colour= font.render('colour: ' + str(round(screen.get_at(((int(CENTER[0]-50), int(CENTER[1]-50)))).g,2)), 1, (0, 0, 102))   
    textpos_colour = text_dir.get_rect(centery=385, left=20)
    
    #anything want to show
    text_show1= font.render('show1: ' + str(round(screen.get_at(((int(CENTER[0]-50), int(CENTER[1]-50)))).g,2)), 1, (0, 0, 102))   
    textpos_show1 = text_dir.get_rect(centery=425, left=20)
    
    text_show2= font.render('show2: ' + str(round(float(car.rpr[1]),2)), 1, (0, 0, 102))   
    textpos_show2 = text_dir.get_rect(centery=465, left=20)
    #anything want to show
    
    ##text setting
    
    
    #model of car
    #35*41 rect
    #middle axis 40
    #axis 35
    #wheel 20
    
    #angle signal give to the object car
    car.wheelangle=angle
    
    ##the new position of the point of wheel after turing
    flb=cal.calculate_rotated_subpoint(CENTER_flw,radius_of_wheel,angle,1)
    flt=cal.calculate_rotated_subpoint(CENTER_flw,radius_of_wheel,angle,-1)
    frb=cal.calculate_rotated_subpoint(CENTER_frw,radius_of_wheel,angle,1)
    frt=cal.calculate_rotated_subpoint(CENTER_frw,radius_of_wheel,angle,-1)  
    ##the new position of the point of wheel after turing
    
    
    ##
    R_flb=cal.calculate_r(flb,CENTER)
    R_flt=cal.calculate_r(flt,CENTER)
    R_frb=cal.calculate_r(frb,CENTER)
    R_frt=cal.calculate_r(frt,CENTER)
    ##
    
    
    ##
    sita_flb=cal.calculate_sita(0,flb,CENTER)
    sita_flt=cal.calculate_sita(0,flt,CENTER)
    sita_frb=cal.calculate_sita(0,frb,CENTER)
    sita_frt=cal.calculate_sita(0,frt,CENTER)
    ##
    
    
    if angle>0 :
        
        fwel=(CENTER_frw[0]-(el_length+half_horizontal_axis_length)*math.cos(math.radians(angle)),CENTER_frw[1]+(el_length+half_horizontal_axis_length)*math.sin(math.radians(angle)))
        R_fwel=cal.calculate_r(fwel,CENTER)
        sita_fwel=cal.calculate_sita(1,fwel,CENTER)
        
        #determine the center of the car moving circle in the coordinate of the canvas layer
        rpl=(CENTER[0]+half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(angle))),CENTER[1]+half_middle_axis_length)
        R_rpl=cal.calculate_r(rpl,CENTER)
        sita_rpl=cal.calculate_sita(1,rpl,CENTER)
        rrl=2*half_middle_axis_length/math.sin(math.radians(angle))
        
        car.rrl=rrl
        car.steerleft(angle)


     
        
        
    elif angle<0  :
        
        
        fwer=(CENTER_flw[0]+(el_length+half_horizontal_axis_length)*math.cos(math.radians(angle)),CENTER_flw[1]-(el_length+half_horizontal_axis_length)*math.sin(math.radians(angle)))
        R_fwer=cal.calculate_r(fwer,CENTER)
        sita_fwer=cal.calculate_sita(1,fwer,CENTER)
        
        #determine the center of the car moving circle in the coordinate of the canvas layer
        rpr=(CENTER[0]-half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(angle))),CENTER[1]+half_middle_axis_length)
        R_rpr=cal.calculate_r(rpr,CENTER)
        sita_rpr=cal.calculate_sita(1,rpr,CENTER)     
        rrr=-2*half_middle_axis_length/math.sin(math.radians(angle))

        car.rrr=rrr
        car.steerright(angle)
    
    
    else:
        
        ##f:front
        #the extension line(end point) of the front wheel.fwel is for front right wheel,fwer is for front left wheel
        fwel=(CENTER[0]-el_length ,CENTER[1]-half_middle_axis_length)
        fwer=(CENTER[0]+el_length ,CENTER[1]-half_middle_axis_length)
        ##f:front
        
        
        ##
        R_fwel=cal.calculate_r(fwel,CENTER)
        R_fwer=cal.calculate_r(fwer,CENTER)
        ##
        
        
        ##
        sita_fwel=cal.calculate_sita(0,fwel,CENTER)
        sita_fwer=cal.calculate_sita(0,fwer,CENTER)
        ##
    
    
    ##
    rpl=cal.calculate_rotated_point(CENTER,car.dir,R_rpl,sita_rpl)
    rpr=cal.calculate_rotated_point(CENTER,car.dir,R_rpr,sita_rpr)
    ##
    
      
    
    ##
    fwel=cal.calculate_rotated_point(CENTER,car.dir,R_fwel,sita_fwel)
    fwer=cal.calculate_rotated_point(CENTER,car.dir,R_fwer,sita_fwer)
    ##
    
    
    ##
    flb=cal.calculate_rotated_point(CENTER,car.dir,R_flb,sita_flb)
    flt=cal.calculate_rotated_point(CENTER,car.dir,R_flt,sita_flt)
    frb=cal.calculate_rotated_point(CENTER,car.dir,R_frb,sita_frb)
    frt=cal.calculate_rotated_point(CENTER,car.dir,R_frt,sita_frt)
    ##
    
    
    ##
    fal=cal.calculate_rotated_point(CENTER,car.dir,R_fal,sita_fal)
    far=cal.calculate_rotated_point(CENTER,car.dir,R_far,sita_far)
    ##
    
    
    ##
    mat=cal.calculate_rotated_point(CENTER,car.dir,R_mat,sita_mat)
    mab=cal.calculate_rotated_point(CENTER,car.dir,R_mab,sita_mab)
    ##
    
    ##
    bal=cal.calculate_rotated_point(CENTER,car.dir,R_bal,sita_bal)
    bar=cal.calculate_rotated_point(CENTER,car.dir,R_bar,sita_bar)
    ##
    
    
    ##
    blb=cal.calculate_rotated_point(CENTER,car.dir,R_blb,sita_blb)
    blt=cal.calculate_rotated_point(CENTER,car.dir,R_blt,sita_blt)
    brb=cal.calculate_rotated_point(CENTER,car.dir,R_brb,sita_brb)
    brt=cal.calculate_rotated_point(CENTER,car.dir,R_brt,sita_brt)
    ##
    
    
    ##
    bwel=cal.calculate_rotated_point(CENTER,car.dir,R_bwel,sita_bwel)
    bwer=cal.calculate_rotated_point(CENTER,car.dir,R_bwer,sita_bwer)
    ##
    
    ##start drawing
    
    
    ##draw background
    screen.blit(background, (0,0))
    ##
    
    
    ##
    map_s.update(cam.x, cam.y)
    map_s.draw(screen)
    ##
    
        
    ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
    car.rpl=(rpl[0]-CENTER[0]+car.x,rpl[1]-CENTER[1]+car.y)
    car.rpr=(rpr[0]-CENTER[0]+car.x,rpr[1]-CENTER[1]+car.y)
    ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
  
    
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
    pygame.draw.line(canvas, (255,255,255), flb, flt,4)
    pygame.draw.line(canvas, (255,255,255), frb, frt,4)#front wheel
    pygame.draw.line(canvas, (255,255,255), fal, far,4)#front axis
    pygame.draw.line(canvas, (255,255,255), mab, mat,4)#middle axis
    pygame.draw.line(canvas, (255,255,255), blt, blb,4)
    pygame.draw.line(canvas, (255,255,255), brt, brb,4)#back wheel
    pygame.draw.line(canvas, (255,255,255), bar, bal,4)#back axis
    ##draw model of the car
    
    
    ##draw back axis extension
    pygame.draw.line(canvas, (255,255,102), bwel,bwer,2)
    ##draw back axis extension
    
    
    ##draw front axis extension
    if angle >= 2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), far,fwel,2)#frontwheel turing axis
   
    if angle <= -2.3 or angle==0:
        
        pygame.draw.line(canvas, (255,255,102), fal,fwer,2)#frontwheel turing axis
    
    ##draw front axis extension
    
    
    ##draw  the car moving circle in canvas
    if  angle>0:
        
        pygame.draw.arc(canvas, (255,255,102), (rpl[0]-rrl,rpl[1]-rrl,2*rrl,2*rrl), 0, 360, 3)
    
    if  angle<0:
        
        pygame.draw.arc(canvas, (255,255,102), (rpr[0]-rrr,rpr[1]-rrr,2*rrr,2*rrr), 0, 360, 3)
    
    ##show canvas
    screen.blit(canvas, (0,0))
    ##show canvas
    
    ##show text
    screen.blit(text_fps, textpos_fps)
    screen.blit(text_timer, textpos_timer)
    screen.blit(text_loop, textpos_loop)
    screen.blit(text_posx, textpos_posx)
    screen.blit(text_posy, textpos_posy)
    screen.blit(text_posxr, textpos_posxr)
    screen.blit(text_posyr, textpos_posyr)
    screen.blit(text_dir, textpos_dir)
    screen.blit(text_speed, textpos_speed)
    screen.blit(text_colour, textpos_colour)
    screen.blit(text_show1, textpos_show1)
    screen.blit(text_show2, textpos_show2)
    ##show text
    
    
    ##start drawing
    
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