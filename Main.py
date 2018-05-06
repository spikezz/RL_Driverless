# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:14:08 2018

@author: Asgard
"""
import sys, pygame,math
import player,maps,tracks,calculation
import camera
from pygame.locals import *
from loader import load_image

start_timer=False
count=0 
COUNT_FREQUENZ=30
angle=0
cd=0#circle direction

pygame.init()

screen = pygame.display.set_mode((pygame.display.Info().current_w,pygame.display.Info().current_h),pygame.FULLSCREEN)

pygame.display.set_caption('Karat Simulation')

font = pygame.font.Font(None, 40)

background = pygame.Surface(screen.get_size())
background = background.convert_alpha()
background.fill((1, 1, 1))

canvas = pygame.Surface(screen.get_size(),SRCALPHA ,32)
canvas = canvas.convert_alpha()
canvas.set_alpha(0)

clock = pygame.time.Clock()
car = player.Player()
cam = camera.Camera()
map_s= pygame.sprite.Group()
player_s= pygame.sprite.Group()
tracks_s  = pygame.sprite.Group()

tracks.initialize()

player_s.add(car)

cam.set_pos(car.x, car.y)
#car.dir=90
#car.image, car.rect = player.rot_center(car.image_orig, car.rect, car.dir)

xc0=car.x
yc0=car.y
x_old=car.x
y_old=car.y
#CENTER_X = 800
#CENTER_Y = 450
CENTER_X =  float(pygame.display.Info().current_w /2)
CENTER_Y =  float(pygame.display.Info().current_h /2)

bwel=(300 ,CENTER_Y+20)
bwer=(1300,CENTER_Y+20)
#R_bwel=calculation.R_point(bwel,CENTER_X,CENTER_Y)
R_bwel=math.sqrt(pow(bwel[0]-CENTER_X,2)+pow(bwel[1]-CENTER_Y,2))
R_bwer=math.sqrt(pow(bwer[0]-CENTER_X,2)+pow(bwer[1]-CENTER_Y,2))

sita_bwel=180-math.degrees(math.atan(25))
sita_bwer=180+math.degrees(math.atan(25))


fwel=(300 ,CENTER_Y-20)
fwer=(1300,CENTER_Y-20)

R_fwel=math.sqrt(pow(fwel[0]-CENTER_X,2)+pow(fwel[1]-CENTER_Y,2))
R_fwer=math.sqrt(pow(fwer[0]-CENTER_X,2)+pow(fwer[1]-CENTER_Y,2))

sita_fwel=math.degrees(math.atan(25))
sita_fwer=-math.degrees(math.atan(25))


rpl=(CENTER_X+17-(40/math.tan(math.radians(5))),CENTER_Y+20)
R_rpl=math.sqrt(pow(rpl[0]-CENTER_X,2)+pow(rpl[1]-CENTER_Y,2))
sita_rpl=180+math.degrees(math.atan((rpl[0]-CENTER_X)/(rpl[1]-CENTER_Y)))
        
rrl=40/math.sin(math.radians(5))

rpr=(CENTER_X-17-(40/math.tan(math.radians(-5))),CENTER_Y+20)
R_rpr=math.sqrt(pow(rpr[0]-CENTER_X,2)+pow(rpr[1]-CENTER_Y,2))
sita_rpr=180+math.degrees(math.atan((rpr[0]-CENTER_X)/(rpr[1]-CENTER_Y)))
        
rrr=-40/math.sin(math.radians(-5))

flb=(CENTER_X-17,CENTER_Y-10)
flt=(CENTER_X-17,CENTER_Y-30)
frb=(CENTER_X+17,CENTER_Y-10)
frt=(CENTER_X+17,CENTER_Y-30)

fal=(CENTER_X-17,CENTER_Y-20)
far=(CENTER_X+17,CENTER_Y-20)

mab=(CENTER_X,CENTER_Y+20)
mat=(CENTER_X,CENTER_Y-20)

bal=(CENTER_X-17,CENTER_Y+20)
bar=(CENTER_X+17,CENTER_Y+20)

blb=(CENTER_X-17,CENTER_Y+30)
blt=(CENTER_X-17,CENTER_Y+10)
brb=(CENTER_X+17,CENTER_Y+30)
brt=(CENTER_X+17,CENTER_Y+10)

R_flb=math.sqrt(pow(flb[0]-CENTER_X,2)+pow(flb[1]-CENTER_Y,2))
R_flt=math.sqrt(pow(flt[0]-CENTER_X,2)+pow(flt[1]-CENTER_Y,2))
R_frb=math.sqrt(pow(frb[0]-CENTER_X,2)+pow(frb[1]-CENTER_Y,2))
R_frt=math.sqrt(pow(frt[0]-CENTER_X,2)+pow(frt[1]-CENTER_Y,2))

R_fal=math.sqrt(pow(fal[0]-CENTER_X,2)+pow(fal[1]-CENTER_Y,2))
R_far=math.sqrt(pow(far[0]-CENTER_X,2)+pow(far[1]-CENTER_Y,2))

R_mab=math.sqrt(pow(mab[0]-CENTER_X,2)+pow(mab[1]-CENTER_Y,2))
R_mat=math.sqrt(pow(mat[0]-CENTER_X,2)+pow(mat[1]-CENTER_Y,2))

R_bal=math.sqrt(pow(bal[0]-CENTER_X,2)+pow(bal[1]-CENTER_Y,2))
R_bar=math.sqrt(pow(bar[0]-CENTER_X,2)+pow(bar[1]-CENTER_Y,2))

R_blb=math.sqrt(pow(blb[0]-CENTER_X,2)+pow(blb[1]-CENTER_Y,2))
R_blt=math.sqrt(pow(blt[0]-CENTER_X,2)+pow(blt[1]-CENTER_Y,2))
R_brb=math.sqrt(pow(brb[0]-CENTER_X,2)+pow(brb[1]-CENTER_Y,2))
R_brt=math.sqrt(pow(brt[0]-CENTER_X,2)+pow(brt[1]-CENTER_Y,2))

sita_flb=math.degrees(math.atan(17/10))
sita_flt=math.degrees(math.atan(17/30))
sita_frb=math.degrees(math.atan(-17/10))
sita_frt=math.degrees(math.atan(-17/30))

sita_fal=math.degrees(math.atan(17/20))
sita_far=math.degrees(math.atan(-17/20))

sita_mab=math.degrees(math.atan(0/-20))+180
sita_mat=math.degrees(math.atan(0/20))

sita_bal=math.degrees(math.atan(17/-20))+180
sita_bar=math.degrees(math.atan(-17/-20))+180

sita_blb=math.degrees(math.atan(17/-30))+180
sita_blt=math.degrees(math.atan(17/-10))+180
sita_brb=math.degrees(math.atan(-17/-30))+180
sita_brt=math.degrees(math.atan(-17/-10))+180    

CENTER_flwx=CENTER_X-17
CENTER_flwy=CENTER_Y-20
CENTER_frwx=CENTER_X+17
CENTER_frwy=CENTER_Y-20



for tile_num in range (0, len(maps.map_tile)):
    
    maps.map_files.append(load_image(maps.map_tile[tile_num],False))
    
for x in range (0,7):
    
    for y in range (0, 20):
        
        map_s.add(maps.Map(maps.map_1[x][y], x * 1000, y * 1000))

while True:
    
    for event in pygame.event.get():
        
        if event.type == QUIT:# 判断事件是否为退出事件
                    
            pygame.quit()# 退出pygame
            
            sys.exit()# 退出系统

            #接收到退出事件后退出程序
        if event.type == KEYDOWN :# 判断事件是否为退出事件            
            if event.key == K_ESCAPE:                              
                pygame.quit()# 退出pygame
                
                sys.exit()# 退出系统
   
            if event.key == K_SPACE :                
                if start_timer==False:                    
                    start_timer=True
                    
                else:                    
                    start_timer=False
               
    keys = pygame.key.get_pressed()
    
    if keys[K_LEFT]:
        car.steerleft()

    if keys[K_1]:
        cd=-1#circle direction
        angle=45
        
    if keys[K_2]:
        cd=-1#circle direction
        angle=30
        
    if keys[K_3]:
        cd=-1#circle direction
        angle=15    
        
    if keys[K_4]:
        cd=-1#circle direction
        angle=2.3
        
    if keys[K_5] or keys[K_6]:
        cd=0#circle direction
        angle=0  
        
        
    if keys[K_7]:
        cd=1#circle direction
        angle=-2.3
        
    if keys[K_8]:
        cd=1#circle direction
        angle=-15
        
    if keys[K_9]:
        cd=1#circle direction
        angle=-30   
        
    if keys[K_0]:
        cd=1#circle direction
        angle=-45
        
        
    if keys[K_RIGHT]:
        
        car.steerright()    
        
    if keys[K_UP]:
        
        #car.speed=10
        car.accelerate()
    else:
        
        car.soften()
        
    if keys[K_DOWN]:
        
        car.deaccelerate()
        
    if keys[K_BACKSPACE]:
        
        car.speed=0
        
    if keys[K_DELETE]:
        
        pass
        
    cam.set_pos(car.x, car.y)
    
    speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s
    
    x_old=car.x
    y_old=car.y
    
    text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (0, 0, 102))
    textpos_fps = text_fps.get_rect(centery=25, left=20)
    
    text_timer = font.render('Timer: ' + str(float(count/COUNT_FREQUENZ)) +'s', 1, (0, 0, 102))
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
    
    text_show= font.render('show: ' + str(round(float(pygame.display.Info().current_w),2)), 1, (0, 0, 102))   
    textpos_show = text_dir.get_rect(centery=385, left=20)
    
    screen.blit(background, (0,0))
    
    map_s.update(cam.x, cam.y)
    map_s.draw(screen)
    
    player_s.update(cam.x, cam.y)
    player_s.draw(screen)

    tracks_s.add(tracks.Track(cam.x + CENTER_X , cam.y + CENTER_Y, car.dir))
    
    tracks_s.update(cam.x, cam.y)
    tracks_s.draw(screen)
    
    canvas.fill((255, 255, 255,0))

    #model of car
    #35*41 rect
    #middle axis 41
    #axis 35
    #wheel 20
       
    car.wheelangle=angle
    
    flb=(CENTER_flwx+10*math.sin(math.radians(angle)),CENTER_flwy+10*math.cos(math.radians(angle)))
    flt=(CENTER_flwx-10*math.sin(math.radians(angle)),CENTER_flwy-10*math.cos(math.radians(angle)))
    frb=(CENTER_frwx+10*math.sin(math.radians(angle)),CENTER_frwy+10*math.cos(math.radians(angle)))
    frt=(CENTER_frwx-10*math.sin(math.radians(angle)),CENTER_frwy-10*math.cos(math.radians(angle)))
    
    R_flb=math.sqrt(pow(flb[0]-CENTER_X,2)+pow(flb[1]-CENTER_Y,2))
    R_flt=math.sqrt(pow(flt[0]-CENTER_X,2)+pow(flt[1]-CENTER_Y,2))
    R_frb=math.sqrt(pow(frb[0]-CENTER_X,2)+pow(frb[1]-CENTER_Y,2))
    R_frt=math.sqrt(pow(frt[0]-CENTER_X,2)+pow(frt[1]-CENTER_Y,2))
    
    sita_flb=math.degrees(math.atan((flb[0]-CENTER_X)/(flb[1]-CENTER_Y)))
    sita_flt=math.degrees(math.atan((flt[0]-CENTER_X)/(flt[1]-CENTER_Y)))
    sita_frb=math.degrees(math.atan((frb[0]-CENTER_X)/(frb[1]-CENTER_Y)))
    sita_frt=math.degrees(math.atan((frt[0]-CENTER_X)/(frt[1]-CENTER_Y)))
    
    
    if cd==-1 :
        
        fwel=(CENTER_X+17-517*math.cos(math.radians(angle)),CENTER_Y-20+517*math.sin(math.radians(angle)))
        R_fwel=math.sqrt(pow(fwel[0]-CENTER_X,2)+pow(fwel[1]-CENTER_Y,2))

        sita_fwel=180+math.degrees(math.atan((fwel[0]-CENTER_X)/(fwel[1]-CENTER_Y)))
        
        rpl=(CENTER_X+17-(40/math.tan(math.radians(angle))),CENTER_Y+20)
        R_rpl=math.sqrt(pow(rpl[0]-CENTER_X,2)+pow(rpl[1]-CENTER_Y,2))
        sita_rpl=180+math.degrees(math.atan((rpl[0]-CENTER_X)/(rpl[1]-CENTER_Y)))
        rrl=40/math.sin(math.radians(angle))
        
     
        
        
    elif cd==1 :
        
        fwer=(CENTER_X-17+517*math.cos(math.radians(angle)),CENTER_Y-20-517*math.sin(math.radians(angle)))
        R_fwer=math.sqrt(pow(fwer[0]-CENTER_X,2)+pow(fwer[1]-CENTER_Y,2))

        sita_fwer=180+math.degrees(math.atan((fwer[0]-CENTER_X)/(fwer[1]-CENTER_Y)))
        
        rpr=(CENTER_X-17-(40/math.tan(math.radians(angle))),CENTER_Y+20)
        R_rpr=math.sqrt(pow(rpr[0]-CENTER_X,2)+pow(rpr[1]-CENTER_Y,2))
        sita_rpr=180+math.degrees(math.atan((rpr[0]-CENTER_X)/(rpr[1]-CENTER_Y)))
        rrr=-40/math.sin(math.radians(angle))
        
    
    
    else:
        
        
        fwel=(300 ,CENTER_Y-20)
        fwer=(1300,CENTER_Y-20)
        
        R_fwel=math.sqrt(pow(fwel[0]-CENTER_X,2)+pow(fwel[1]-CENTER_Y,2))
        R_fwer=math.sqrt(pow(fwer[0]-CENTER_X,2)+pow(fwer[1]-CENTER_Y,2))
        
        sita_fwel=math.degrees(math.atan(25))
        sita_fwer=-math.degrees(math.atan(25))
            
    
    
    
    
    rpl=(CENTER_X+R_rpl*math.cos(math.radians(270-car.dir-sita_rpl)),CENTER_Y+R_rpl* math.sin(math.radians(270-car.dir-sita_rpl)))
    rpr=(CENTER_X+R_rpr*math.cos(math.radians(270-car.dir-sita_rpr)),CENTER_Y+R_rpr* math.sin(math.radians(270-car.dir-sita_rpr)))
    
    fwel=(CENTER_X+R_fwel*math.cos(math.radians(270-car.dir-sita_fwel)),CENTER_Y+R_fwel* math.sin(math.radians(270-car.dir-sita_fwel)))
    fwer=(CENTER_X+R_fwer*math.cos(math.radians(270-car.dir-sita_fwer)),CENTER_Y+R_fwer* math.sin(math.radians(270-car.dir-sita_fwer)))
    
    
    flb=(CENTER_X+R_flb*math.cos(math.radians(270-car.dir-sita_flb)),CENTER_Y+R_flb* math.sin(math.radians(270-car.dir-sita_flb)))
    flt=(CENTER_X+R_flt*math.cos(math.radians(270-car.dir-sita_flt)),CENTER_Y+R_flt* math.sin(math.radians(270-car.dir-sita_flt)))
    
    frb=(CENTER_X+R_frb*math.cos(math.radians(270-car.dir-sita_frb)),CENTER_Y+R_frb* math.sin(math.radians(270-car.dir-sita_frb)))
    frt=(CENTER_X+R_frt*math.cos(math.radians(270-car.dir-sita_frt)),CENTER_Y+R_frt* math.sin(math.radians(270-car.dir-sita_frt)))
    
    fal=(CENTER_X+R_fal*math.cos(math.radians(270-car.dir-sita_fal)),CENTER_Y+R_fal* math.sin(math.radians(270-car.dir-sita_fal)))
    far=(CENTER_X+R_far*math.cos(math.radians(270-car.dir-sita_far)),CENTER_Y+R_far* math.sin(math.radians(270-car.dir-sita_far)))
    
    mat=(CENTER_X+R_mat*math.cos(math.radians(270-car.dir-sita_mat)),CENTER_Y+R_mat* math.sin(math.radians(270-car.dir-sita_mat)))
    mab=(CENTER_X+R_mab*math.cos(math.radians(270-car.dir-sita_mab)),CENTER_Y+R_mab* math.sin(math.radians(270-car.dir-sita_mab)))
    
    bal=(CENTER_X+R_bal*math.cos(math.radians(270-car.dir-sita_bal)),CENTER_Y+R_bal* math.sin(math.radians(270-car.dir-sita_bal)))
    bar=(CENTER_X+R_bar*math.cos(math.radians(270-car.dir-sita_bar)),CENTER_Y+R_bar* math.sin(math.radians(270-car.dir-sita_bar)))
    
    blb=(CENTER_X+R_blb*math.cos(math.radians(270-car.dir-sita_blb)),CENTER_Y+R_blb* math.sin(math.radians(270-car.dir-sita_blb)))
    blt=(CENTER_X+R_blt*math.cos(math.radians(270-car.dir-sita_blt)),CENTER_Y+R_blt* math.sin(math.radians(270-car.dir-sita_blt)))
    brb=(CENTER_X+R_brb*math.cos(math.radians(270-car.dir-sita_brb)),CENTER_Y+R_brb* math.sin(math.radians(270-car.dir-sita_brb)))
    brt=(CENTER_X+R_brt*math.cos(math.radians(270-car.dir-sita_brt)),CENTER_Y+R_brt* math.sin(math.radians(270-car.dir-sita_brt)))
    
    bwel=(CENTER_X+R_bwel*math.cos(math.radians(270-car.dir-sita_bwel)),CENTER_Y+R_bwel* math.sin(math.radians(270-car.dir-sita_bwel)))
    bwer=(CENTER_X+R_bwer*math.cos(math.radians(270-car.dir-sita_bwer)),CENTER_Y+R_bwer* math.sin(math.radians(270-car.dir-sita_bwer)))
    
    
    
    pygame.draw.line(canvas, (0,100,0), (100,CENTER_Y), (CENTER_X,CENTER_Y),2)
    pygame.draw.line(canvas, (0,100,0), (CENTER_X,100), (CENTER_X,CENTER_Y),2)#front wheel
    
    pygame.draw.line(canvas, (0,1,0), flb, flt,2)
    pygame.draw.line(canvas, (0,1,0), frb, frt,2)#front wheel
    pygame.draw.line(canvas, (0,1,0), fal, far,2)#front axis
    pygame.draw.line(canvas, (0,1,0), mab, mat,2)#middle axis
    pygame.draw.line(canvas, (0,1,0), blt, blb,2)
    pygame.draw.line(canvas, (0,1,0), brt, brb,2)#back wheel
    pygame.draw.line(canvas, (0,1,0), bar, bal,2)#back axis
    
    pygame.draw.line(canvas, (255,255,102), bwel,bwer,2)#backwheel axis
    if angle >= 2.3 or angle==0:
        pygame.draw.line(canvas, (255,255,102), far,fwel,2)#frontwheel turing axis
    if angle <= -2.3 or angle==0:
        pygame.draw.line(canvas, (255,255,102), fal,fwer,2)#frontwheel turing axis
    
    
    
    
    
    

    if cd==-1 :
        
        pygame.draw.arc(screen, (255,255,102), (rpl[0]-rrl,rpl[1]-rrl,2*rrl,2*rrl), 0, 360, 3)
    
    if cd== 1 :
        
        pygame.draw.arc(screen, (255,255,102), (rpr[0]-rrr,rpr[1]-rrr,2*rrr,2*rrr), 0, 360, 3)
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    screen.blit(canvas, (0,0))
    
    screen.blit(text_fps, textpos_fps)
    screen.blit(text_timer, textpos_timer)
    screen.blit(text_loop, textpos_loop)
    screen.blit(text_posx, textpos_posx)
    screen.blit(text_posy, textpos_posy)
    screen.blit(text_posxr, textpos_posxr)
    screen.blit(text_posyr, textpos_posyr)
    screen.blit(text_dir, textpos_dir)
    screen.blit(text_speed, textpos_speed)
    screen.blit(text_show, textpos_show)
    
    if start_timer==True:

        count=count+1
    
    clock.tick_busy_loop(COUNT_FREQUENZ)

    pygame.display.update()