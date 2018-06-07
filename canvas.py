# -*- coding: utf-8 -*-
"""
Created on Thu May 10 18:09:08 2018

@author: Asgard
"""

import pygame, math
import calculation as cal
from pygame.locals import *


def initialize_model(CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length):
    
    
    ##bwel/r:back,wheel,extension,left/right end point
    #the extension line(end point) for finding the center of the circle kurve, which the car obey when turning
    bwel=[CENTER[0]-el_length ,CENTER[1]+half_middle_axis_length]
    bwer=[CENTER[0]+el_length ,CENTER[1]+half_middle_axis_length]
    ##bwel/r:back,wheel,extension,left/right end point
        
    ##R_bwel:R_ radius from the point to the center of the screen
    R_bwel=cal.calculate_r(bwel,CENTER)
    R_bwer=cal.calculate_r(bwer,CENTER)
    ##R_bwel:R_ radius from the point to the center of the screen
    
    ##sita_bwel:the angle between car.dir=0 and the point
    sita_bwel=cal.calculate_sita(1,bwel,CENTER)
    sita_bwer=cal.calculate_sita(1,bwer,CENTER)
    ##sita_bwel:the angle between car.dir=0 and the point
    BWEL=[bwel,R_bwel,sita_bwel]
    BWER=[bwer,R_bwer,sita_bwer]
    
    
    ##f:front
    #the extension line(end point) of the front wheel.fwel is for front right wheel,fwer is for front left wheel
    fwel=[CENTER[0]-el_length ,CENTER[1]-half_middle_axis_length]
    fwer=[CENTER[0]+el_length ,CENTER[1]-half_middle_axis_length]
    ##f:front
    
    
    ##
    R_fwel=cal.calculate_r(fwel,CENTER)
    R_fwer=cal.calculate_r(fwer,CENTER)
    ##
    
    
    ##
    sita_fwel=cal.calculate_sita(0,fwel,CENTER)
    sita_fwer=cal.calculate_sita(0,fwer,CENTER)
    ##
    FWEL=[fwel,R_fwel,sita_fwel]
    FWER=[fwer,R_fwer,sita_fwer]
    
    
    ##rpl:round(circle) posion left(center); rrl:round(circle) radius left(radius)
    #find the center/radius of the circle kurve, which the car obey when turning
    rpl=[CENTER[0]+half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(5))),CENTER[1]+half_middle_axis_length]
    R_rpl=cal.calculate_r(rpl,CENTER)
    sita_rpl=cal.calculate_sita(1,rpl,CENTER)
    rrl=2*half_middle_axis_length/math.sin(math.radians(5))
    ##
    RPL=[rpl,R_rpl,sita_rpl]
    
    ##rpr:round(circle) posion right(center); rrr:round(circle) radius right(radius)
    #find the center/radius of the circle kurve, which the car obey when turning
    rpr=[CENTER[0]-half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(-5))),CENTER[1]+half_middle_axis_length]
    R_rpr=cal.calculate_r(rpr,CENTER)
    sita_rpr=cal.calculate_sita(1,rpr,CENTER)     
    rrr=-2*half_middle_axis_length/math.sin(math.radians(-5))
    ##
    RPR=[rpr,R_rpr,sita_rpr]
    
    
    ##fl/rb/t:front left/right bottom/top(point);
    flb=[CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length+radius_of_wheel]
    flt=[CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length-radius_of_wheel]
    frb=[CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length+radius_of_wheel]
    frt=[CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length-radius_of_wheel]
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
    FLB=[flb,R_flb,sita_flb]
    FLT=[flt,R_flt,sita_flt]
    FRB=[frb,R_frb,sita_frb]
    FRT=[frt,R_frt,sita_frt]
    
    
    ##fal/r:front axis left/right
    fal=[CENTER[0]-half_horizontal_axis_length,CENTER[1]-half_middle_axis_length]
    far=[CENTER[0]+half_horizontal_axis_length,CENTER[1]-half_middle_axis_length]
    ##fal/r:front axis left/right
    
    
    ##
    R_fal=cal.calculate_r(fal,CENTER)
    R_far=cal.calculate_r(far,CENTER)
    ##
    
    
    ##
    sita_fal=cal.calculate_sita(0,fal,CENTER) 
    sita_far=cal.calculate_sita(0,far,CENTER) 
    ##
    FAL=[fal,R_fal,sita_fal]
    FAR=[far,R_far,sita_far]
    
    
    ##mab/t:middle axis bottom/top
    mab=[CENTER[0],CENTER[1]+half_middle_axis_length]
    mat=[CENTER[0],CENTER[1]-half_middle_axis_length]
    ##mab/t:middle axis bottom/top
    
    
    ##
    R_mab=cal.calculate_r(mab,CENTER)
    R_mat=cal.calculate_r(mat,CENTER)
    ##
    
    
    ##
    sita_mab=cal.calculate_sita(1,mab,CENTER) 
    sita_mat=cal.calculate_sita(0,mat,CENTER)
    ##
    MAB=[mab,R_mab,sita_mab]
    MAT=[mat,R_mat,sita_mat]
    
    
    ##bal/r:back axis left/right
    bal=[CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length]
    bar=[CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length]
    ##bal/r:back axis left/right
    
    
    ##
    R_bal=cal.calculate_r(bal,CENTER)
    R_bar=cal.calculate_r(bar,CENTER)
    ##
    
    ##
    sita_bal=cal.calculate_sita(1,bal,CENTER) 
    sita_bar=cal.calculate_sita(1,bar,CENTER) 
    ##
    BAL=[bal,R_bal,sita_bal]
    BAR=[bar,R_bar,sita_bar]
    
    
    ##bl/rb/t:back left/right bottom/top(point);
    blb=[CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length+radius_of_wheel]
    blt=[CENTER[0]-half_horizontal_axis_length,CENTER[1]+half_middle_axis_length-radius_of_wheel]
    brb=[CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length+radius_of_wheel]
    brt=[CENTER[0]+half_horizontal_axis_length,CENTER[1]+half_middle_axis_length-radius_of_wheel]
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
    BLB=[blb,R_blb,sita_blb]
    BLT=[blt,R_blt,sita_blt]
    BRB=[brb,R_brb,sita_brb]
    BRT=[brt,R_brt,sita_brt]
    
    
    ##fl/rwx/y:x/y of front left/right wheel
    CENTER_flwx=CENTER[0]-half_horizontal_axis_length
    CENTER_flwy=CENTER[1]-half_middle_axis_length
    CENTER_frwx=CENTER[0]+half_horizontal_axis_length
    CENTER_frwy=CENTER[1]-half_middle_axis_length
    CENTER_flw=[CENTER_flwx,CENTER_flwy]
    CENTER_frw=[CENTER_frwx,CENTER_frwy]
    ##fl/rwx/y:x/y of front left/right wheel
    
    ###initial Model of the car
    model=[FLB,FLT,FRB,FRT,FAL,FAR,MAB,MAT,BAL,BAR,BLB,BLT,BRB,BRT,FWEL,FWER,BWEL,BWER,RPL,rrl,RPR,rrr,CENTER_flw,CENTER_frw]    
    
    return model

def turning(model,angle,CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length):
    
    
     ##the new position of the point of wheel after turing
    model[0][0]=cal.calculate_rotated_subpoint(model[22],radius_of_wheel,angle,1)
    model[1][0]=cal.calculate_rotated_subpoint(model[22],radius_of_wheel,angle,-1)
    model[2][0]=cal.calculate_rotated_subpoint(model[23],radius_of_wheel,angle,1)
    model[3][0]=cal.calculate_rotated_subpoint(model[23],radius_of_wheel,angle,-1)  
    ##the new position of the point of wheel after turing
    
    
    ##
    model[0][1]=cal.calculate_r(model[0][0],CENTER)
    model[1][1]=cal.calculate_r(model[1][0],CENTER)
    model[2][1]=cal.calculate_r(model[2][0],CENTER)
    model[3][1]=cal.calculate_r(model[3][0],CENTER)
    ##
    
    
    ##
    model[0][2]=cal.calculate_sita(0,model[0][0],CENTER)
    model[1][2]=cal.calculate_sita(0,model[1][0],CENTER)
    model[2][2]=cal.calculate_sita(0,model[2][0],CENTER)
    model[3][2]=cal.calculate_sita(0,model[3][0],CENTER)
    ##
    
    
    if angle>0 :
        
        model[14][0]=(model[23][0]-(el_length+half_horizontal_axis_length)*math.cos(math.radians(angle)),model[23][1]+(el_length+half_horizontal_axis_length)*math.sin(math.radians(angle)))
        model[14][1]=cal.calculate_r(model[14][0],CENTER)
        model[14][2]=cal.calculate_sita(1,model[14][0],CENTER)
        
        #determine the center of the car moving circle in the coordinate of the canvas layer
        model[18][0]=(CENTER[0]+half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(angle))),CENTER[1]+half_middle_axis_length)
        model[18][1]=cal.calculate_r(model[18][0],CENTER)
        model[18][2]=cal.calculate_sita(1,model[18][0],CENTER)
        model[19]=2*half_middle_axis_length/math.sin(math.radians(angle))
        
        #car.rrl=rrl
        #car.steerleft(angle)


     
        
        
    elif angle<0  :
        
        
        model[15][0]=(model[22][0]+(el_length+half_horizontal_axis_length)*math.cos(math.radians(angle)),model[22][1]-(el_length+half_horizontal_axis_length)*math.sin(math.radians(angle)))
        model[15][1]=cal.calculate_r(model[15][0],CENTER)
        model[15][2]=cal.calculate_sita(1,model[15][0],CENTER)
        
        #determine the center of the car moving circle in the coordinate of the canvas layer
        model[20][0]=(CENTER[0]-half_horizontal_axis_length-(2*half_middle_axis_length/math.tan(math.radians(angle))),CENTER[1]+half_middle_axis_length)
        model[20][1]=cal.calculate_r(model[20][0],CENTER)
        model[20][2]=cal.calculate_sita(1,model[20][0],CENTER)     
        model[21]=-2*half_middle_axis_length/math.sin(math.radians(angle))

        #car.rrr=rrr
        #car.steerright(angle)
    
    
    else:
        
        ##f:front
        #the extension line(end point) of the front wheel.fwel is for front right wheel,fwer is for front left wheel
        model[14][0]=(CENTER[0]-el_length ,CENTER[1]-half_middle_axis_length)
        model[15][0]=(CENTER[0]+el_length ,CENTER[1]-half_middle_axis_length)
        ##f:front
        
        
        ##
        model[14][1]=cal.calculate_r(model[14][0],CENTER)
        model[15][1]=cal.calculate_r(model[15][0],CENTER)
        ##
        
        
        ##
        model[14][2]=cal.calculate_sita(0,model[14][0],CENTER)
        model[15][2]=cal.calculate_sita(0,model[15][0],CENTER)
        ##
    
    
    return model

def rotate(model,CENTER,direction):
    
    
  
    ##
    model[18][0]=cal.calculate_rotated_point(CENTER,direction,model[18][1],model[18][2])
    model[20][0]=cal.calculate_rotated_point(CENTER,direction,model[20][1],model[20][2])
    ##
    
      
    
    ##
    model[14][0]=cal.calculate_rotated_point(CENTER,direction,model[14][1],model[14][2])
    model[15][0]=cal.calculate_rotated_point(CENTER,direction,model[15][1],model[15][2])
    ##
    
    
    ##
    model[0][0]=cal.calculate_rotated_point(CENTER,direction,model[0][1],model[0][2])
    model[1][0]=cal.calculate_rotated_point(CENTER,direction,model[1][1],model[1][2])
    model[2][0]=cal.calculate_rotated_point(CENTER,direction,model[2][1],model[2][2])
    model[3][0]=cal.calculate_rotated_point(CENTER,direction,model[3][1],model[3][2])
    ##
    
    
    ##
    model[4][0]=cal.calculate_rotated_point(CENTER,direction,model[4][1],model[4][2])
    model[5][0]=cal.calculate_rotated_point(CENTER,direction,model[5][1],model[5][2])
    ##
    
    
    ##
    model[6][0]=cal.calculate_rotated_point(CENTER,direction,model[6][1],model[6][2])
    model[7][0]=cal.calculate_rotated_point(CENTER,direction,model[7][1],model[7][2])
    ##
    
    ##
    model[8][0]=cal.calculate_rotated_point(CENTER,direction,model[8][1],model[8][2])
    model[9][0]=cal.calculate_rotated_point(CENTER,direction,model[9][1],model[9][2])
    ##
    
    
    ##
    model[10][0]=cal.calculate_rotated_point(CENTER,direction,model[10][1],model[10][2])
    model[11][0]=cal.calculate_rotated_point(CENTER,direction,model[11][1],model[11][2])
    model[12][0]=cal.calculate_rotated_point(CENTER,direction,model[12][1],model[12][2])
    model[13][0]=cal.calculate_rotated_point(CENTER,direction,model[13][1],model[13][2])
    ##
    
    
    ##
    model[16][0]=cal.calculate_rotated_point(CENTER,direction,model[16][1],model[16][2])
    model[17][0]=cal.calculate_rotated_point(CENTER,direction,model[17][1],model[17][2])
    ##
    
    
    return model

def input_vektor_position(model,draw_cone,CENTER,direction):
    
    vektor=[draw_cone[0]-model[7][0][0]+CENTER[0],draw_cone[1]-model[7][0][1]+CENTER[1]]
    R_vektor=cal.calculate_r(vektor,CENTER)
    sita_vektor=cal.calculate_sita(0,vektor,CENTER)
    vektor=cal.calculate_rotated_point(CENTER,-direction,R_vektor,sita_vektor)

    if draw_cone[1]<=model[7][0][1]:
         
         vektor=[vektor[0]-CENTER[0],vektor[1]-CENTER[1]]
         
    else:
        
         vektor=[-(vektor[0]-CENTER[0]),-(vektor[1]-CENTER[1])]
         
    return vektor