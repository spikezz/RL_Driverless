#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:54:24 2019

@author: spikezz
"""
from airsim import ImageRequest
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64


import numpy as np
import calculate as cal
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.special as ss

import airsim
import rospy
import os
import tools
import RL
import time

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

def set_throttle(data):
    
    car_controls.throttle=data.data
#    client.setCarControls(car_controls)
    print("befehl:",data.data)
    
def set_brake(data):
    
    car_controls.brake=data.data
#    client.setCarControls(car_controls)
    print("befehl:",data.data)
    
def set_steering(data):
    
    car_controls.steering=data.data
#    client.setCarControls(car_controls)
    print("befehl:",data.data)

#def set_control_vechicle():
    
#    action = actor.choose_action(observation,critic,ep_total)
#    probability=softmax(np.array([action[3],action[4],action[5]]))
#
#    action_ori[0].append(action[0])
#    action_ori[1].append(action[1])
#    action_ori[2].append(action[2])
#    action_ori[3].append(probability[0])
#    action_ori[4].append(probability[1])
#    action_ori[5].append(probability[2])
#    #
#    action[0] = np.clip(np.random.normal(action[0], var[0]), *ACTION_BOUND0)
#    action[1] = np.clip(np.random.normal(action[1], var[1]), *ACTION_BOUND1)
# 
#    noise_factor_blue=-np.exp(2*(sin_projection_blue-2.5))+1
#    noise_factor_yellow=-np.exp(2*(sin_projection_yellow-2.5))+1
#    
#    if sin_projection_blue<noise_bound:
#        
#        action[2]=action[2]-noise_factor_blue
#
#    if sin_projection_yellow<noise_bound:
#        
#        action[2]=action[2]+noise_factor_yellow
#    
#    action[2] = np.clip(np.random.normal(action[2], var[2]), *ACTION_BOUND2)
#    
#    action_corr=[action[0],action[1]]
#    action_corr[0] = action_corr[0] +(ACTION_BOUND0[1]-ACTION_BOUND0[0])*1.1
#    action_corr[1] = action_corr[1] +(ACTION_BOUND1[1]-ACTION_BOUND1[0])/2
#    
#    car_controls.steering=float(action[2])
#         
#    actor.angle.append(action[2])
#   
#    choice=np.random.choice(range(len(probability)),p=probability)
#
#    if choice==0:
#        
#        car_controls.brake=0
#        action_corr[1]=car_controls.brake
#        action[1]=action_corr[1]-(ACTION_BOUND1[1]-ACTION_BOUND1[0])/2
#  
#        if  car_state.speed<=velocity_max:
#            
#            car_controls.throttle=float(action_corr[0]) #beschleunigen
#            action_lock=True
#            
#        else:
#            
#            car_controls.throttle=0
#            action_corr[0]=car_controls.throttle
#            action[0]=action_corr[0]-(ACTION_BOUND0[1]-ACTION_BOUND0[0])/2
#
#        actor.accelerate.append(action_corr[0])
#        actor.brake.append(action_corr[1])
#        
#    elif choice==1:
#        
#        car_controls.throttle=0
#        action_corr[0]=0
#        action[0]=action_corr[0]-(ACTION_BOUND0[1]-ACTION_BOUND0[0])/2
#
#        if  car_state.speed>=velocity_min:
#            car_controls.brake=float(action_corr[1]) #bremsen
#            actor.brake.append(action_corr[1])
#        else:
#            car_controls.brake=0#bremsen
#            action_corr[1]=car_controls.brake
#            action[1]=action_corr[1]-(ACTION_BOUND1[1]-ACTION_BOUND1[0])/2
#            actor.brake.append(action_corr[1])
#        actor.accelerate.append(action_corr[0])
#     
#        
#    elif choice==2:
#        
#        action_corr[0]=car_controls.throttle
#        action[0]=action_corr[0]-(ACTION_BOUND0[1]-ACTION_BOUND0[0])/2
#        action_corr[1]=car_controls.brake
#        action[1]=action_corr[1]-(ACTION_BOUND1[1]-ACTION_BOUND1[0])/2
#        actor.accelerate.append(action_corr[0])
#        actor.brake.append(action_corr[1])
#
#    client.setCarControls(car_controls)

def get_memory(bound_lidar):
    
    list_cone_sensored=[]
    speed_projection=[0,0]
    dis_close_blue_cone_1=1000
    blue_cone_close_1=[]
    
    for c in list_blue_cone:

        try:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
        
        except:
            
            print("odo_msg",odo_msg.pose.pose.position)
            print("cone",c)
            
        if distance_cone< bound_lidar:
            
            list_cone_sensored.append([c.position.x_val,c.position.y_val])
            
        if distance_cone<dis_close_blue_cone_1:
            
            blue_cone_close_1=[c.position.x_val,c.position.y_val]
            dis_close_blue_cone_1=distance_cone
            
    dis_close_blue_cone_2=1000
    blue_cone_close_2=[]
    
    for c in list_blue_cone:
        
        try:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
        
        except:
            
            print("odo_msg",odo_msg.pose.pose.position)
            print("cone",c)
            
        if distance_cone<dis_close_blue_cone_2 and distance_cone>dis_close_blue_cone_1:
            
            blue_cone_close_2=[c.position.x_val,c.position.y_val]
            dis_close_blue_cone_2=distance_cone
    try:   
        
        dis_between_blue_cone=cal.calculate_r(blue_cone_close_1,blue_cone_close_2)
        v_cone_b=np.array(blue_cone_close_1)-np.array(blue_cone_close_2)
        r_cone_b=cal.calculate_r(v_cone_b,[0,0])
        
    except:
        
        print(blue_cone_close_1,blue_cone_close_2)
        
    sin_projection_blue=cal.calculate_projection(True,dis_close_blue_cone_1,dis_close_blue_cone_2,dis_between_blue_cone)[1]
     
    dis_close_yellow_cone_1=1000
    yellow_cone_close_1=[]
    
    for c in list_yellow_cone:
        
        try:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
        
        except:
            
            print("odo_msg",odo_msg.pose.pose.position)
            print("cone",c)
            
        if distance_cone< bound_lidar:
            
            list_cone_sensored.append([c.position.x_val,c.position.y_val])
    
        if distance_cone<dis_close_yellow_cone_1:
            
            yellow_cone_close_1=[c.position.x_val,c.position.y_val]
            dis_close_yellow_cone_1=distance_cone
            
    dis_close_yellow_cone_2=1000
    yellow_cone_close_2=[]
    
    for c in list_yellow_cone:
        
        try:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
        
        except:
            
            print("odo_msg",odo_msg.pose.pose.position)
            print("cone",c)
            
        if distance_cone<dis_close_yellow_cone_2 and distance_cone>dis_close_yellow_cone_1:
            
            yellow_cone_close_2=[c.position.x_val,c.position.y_val]
            dis_close_yellow_cone_2=distance_cone
    try:   
        
        dis_between_yellow_cone=cal.calculate_r(yellow_cone_close_1,yellow_cone_close_2)
        v_cone_y=np.array(yellow_cone_close_1)-np.array(yellow_cone_close_2)
        r_cone_y=cal.calculate_r(v_cone_y,[0,0])
        
    except:
        
        print(yellow_cone_close_1,yellow_cone_close_2)     
     
    
    sin_projection_yellow=cal.calculate_projection(True,dis_close_yellow_cone_1,dis_close_yellow_cone_2,dis_between_yellow_cone)[1]
    
    
    v_speed=np.array([vel_msg.x,vel_msg.y])
    speed_projection[0]=np.absolute(np.dot(v_speed,v_cone_b)/r_cone_b)
    speed_projection[1]=np.absolute(np.dot(v_speed,v_cone_y)/r_cone_y)
    reward_projection=(speed_projection[0]+speed_projection[1])/2

    try:
        
        distance_coneback=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[coneback[0].position.x_val,coneback[0].position.y_val])       
     
    except:
        
        print("odo_msg",odo_msg.pose.pose.position)
        print("coneback",coneback)
     
    cone_sort_temp=[]
    cone_sort_end=[]
            
    for i in range (len(list_cone_sensored)):
              
        cone_sort_temp.append([cal.calculate_sita_r(list_cone_sensored[i],[0,0]),list_cone_sensored[i]])
    
    cone_sort_temp=sorted(cone_sort_temp)
    
    for i in range (len(cone_sort_temp)):
        
        cone_sort_end.append(cone_sort_temp[i][1])
    
    state[0]=cone_sort_end
    
    if len(state[0])!=0:
        state[0]=np.vstack(state[0]).ravel()
    try:
        state[0]=state[0]/bound_lidar
    except:
        print("unsupported operand type(s) for /: 'list' and 'int'")

    state[1]=np.vstack([vel_msg.x,vel_msg.y,vel_msg.z]).ravel()/velocity_max
    state[2]=np.vstack([acc_msg.x,acc_msg.y,acc_msg.z]).ravel()
    state[3]=np.vstack([a_a_msg.w,a_a_msg.x,a_a_msg.y,a_a_msg.z]).ravel()/math.pi
    state[4]=np.vstack([a_v_msg.w,a_v_msg.x,a_v_msg.y,a_v_msg.z]).ravel()/math.pi      
    state[5]=np.vstack([eul_msg.x,eul_msg.y,eul_msg.z]).ravel()/math.pi
    
    state_input=np.hstack((car_controls.steering,state[1],state[2],state[3],state[4],state[5]))  
    
    for t in range(len(state[0])):
        
        observation[t]=state[0][t]
        
    for t in range(-len(state_input),0):
        
        observation[t]=state_input[t]
       
    reward=5*np.exp(reward_projection)

    if sin_projection_yellow<collision_distance or sin_projection_blue<collision_distance:
            
        collision=True
        
    if distance_coneback!=None:
        
        if distance_coneback<5.0:
            
            collide_finish=True
    
    reward_sum=reward_sum+reward

    if collision==True or collide_finish==True: 
        
        if collision==True :
               
            reward=-pow(np.exp(car_state.speed)*5,1)
     
        client.reset()
        reward_sum=reward_sum+reward
        running_reward=reward_sum
        reward_sum=0
        time_stamp = time.time()
        car_controls.steering=0
        car_controls.throttle=0
        car_controls.brake=0
        count=0
        collision=False
        collide_finish=False
    
    memory=[action, reward, observation]
    
    return memory

rospy.init_node('publish', anonymous=True)
pub_I = rospy.Publisher('UnrealImage', Image, queue_size=1000)
pub_E = rospy.Publisher('O_to_E', Vector3, queue_size=1000)
pub_velocity = rospy.Publisher('velocity', Vector3, queue_size=1000)
pub_acceleration = rospy.Publisher('acceleration', Vector3, queue_size=1000)
pub_angular_acceleration = rospy.Publisher('angular_acceleration', Quaternion, queue_size=1000)
pub_angular_velocity = rospy.Publisher('angular_velocity', Quaternion, queue_size=1000)
pub_Odometry_auto = rospy.Publisher('Odometry_auto', Odometry, queue_size=1000)
pub_action = rospy.Publisher('action', Float32MultiArray, queue_size=1000)
pub_blue_cone=rospy.Publisher('rightCones', PoseArray, queue_size=1000)
pub_yellow_cone=rospy.Publisher('leftCones', PoseArray, queue_size=1000)
#pub_Q = rospy.Publisher('Quaternion', Quaternion, queue_size=1000)
sub_throt=rospy.Subscriber("throttle",Float64,set_throttle)
sub_steer=rospy.Subscriber("steeringAngle",Float64, set_steering)

rate = rospy.Rate(60) # 10hz
car_state = client.getCarState()
#calibration white noise of velocity
init_v_x=car_state.kinematics_estimated.linear_velocity.x_val
init_v_y=car_state.kinematics_estimated.linear_velocity.y_val
init_v_z=car_state.kinematics_estimated.linear_velocity.z_val
#calibration white noise of velocity

while not rospy.is_shutdown():

    acc_msg=Vector3()
    vel_msg=Vector3()
    a_a_msg=Quaternion()
    a_v_msg=Quaternion()
    odo_msg=Odometry()
    qua_msg=Quaternion()
    eul_msg=Vector3()
    act_msg=Float32MultiArray()
    bcn_msg=PoseArray()
    ycn_msg=PoseArray()
    
    car_state = client.getCarState()
    
    acc_msg.x=car_state.kinematics_estimated.linear_acceleration.x_val
    acc_msg.y=car_state.kinematics_estimated.linear_acceleration.y_val
    acc_msg.z=car_state.kinematics_estimated.linear_acceleration.z_val
    
    vel_msg.x=car_state.kinematics_estimated.linear_velocity.x_val-init_v_x
    vel_msg.y=car_state.kinematics_estimated.linear_velocity.y_val-init_v_y
    vel_msg.z=car_state.kinematics_estimated.linear_velocity.z_val-init_v_z

    a_a_msg.w=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().w_val
    a_a_msg.x=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().x_val
    a_a_msg.y=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().y_val
    a_a_msg.z=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().z_val
    
    a_v_msg.w=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().w_val
    a_v_msg.x=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().x_val
    a_v_msg.y=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().y_val
    a_v_msg.z=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().z_val

    odo_msg.header.seq = 0
    odo_msg.header.stamp = rospy.get_rostime()
    odo_msg.header.frame_id = ""
    odo_msg.pose.pose.position.x=car_state.kinematics_estimated.position.x_val
    odo_msg.pose.pose.position.y=car_state.kinematics_estimated.position.y_val
    odo_msg.pose.pose.position.z=car_state.kinematics_estimated.position.z_val
    odo_msg.pose.pose.orientation.w=car_state.kinematics_estimated.orientation.w_val
    odo_msg.pose.pose.orientation.x=car_state.kinematics_estimated.orientation.x_val
    odo_msg.pose.pose.orientation.y=car_state.kinematics_estimated.orientation.y_val
    odo_msg.pose.pose.orientation.z=car_state.kinematics_estimated.orientation.z_val
    odo_msg.twist.twist.linear.x=vel_msg.x
    odo_msg.twist.twist.linear.y=vel_msg.y
    odo_msg.twist.twist.linear.z=vel_msg.z
    
    (eul_msg.x, eul_msg.y, eul_msg.z) = cal.euler_from_quaternion([odo_msg.pose.pose.orientation.x, odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])

    list_blue_cone=client.simGetObjectPoses("RightCone")
    list_yellow_cone=client.simGetObjectPoses("LeftCone")  
    coneback=client.simGetObjectPoses("finish")
    
    act_msg.data.append(car_controls.throttle)
    act_msg.data.append(car_controls.brake)
    act_msg.data.append(car_controls.steering)
    client.setCarControls(car_controls)
    
    bcn_msg.header.seq = 0
    bcn_msg.header.stamp = rospy.get_rostime()
    bcn_msg.header.frame_id = ""
    
    for c in list_blue_cone:
        
        new_pose=Pose()
        #            new_pose.position = Point()
        new_pose.position.x=c.position.x_val
        new_pose.position.y=c.position.y_val
        new_pose.position.z=c.position.z_val
        bcn_msg.poses.append(new_pose)
        
    ycn_msg.header.seq = 0
    ycn_msg.header.stamp = rospy.get_rostime()
    ycn_msg.header.frame_id = ""
    
    for c in list_yellow_cone:
#            print("c:",c)
        new_pose=Pose()            
        new_pose.position.x=c.position.x_val
        new_pose.position.y=c.position.y_val
        new_pose.position.z=c.position.z_val
        ycn_msg.poses.append(new_pose)
        

    
#    pub_I.publish(image_msg)
    pub_acceleration.publish(acc_msg)
    pub_velocity.publish(vel_msg)
    pub_angular_acceleration.publish(a_a_msg)
    pub_angular_velocity.publish(a_v_msg)
    pub_Odometry_auto.publish(odo_msg)
    pub_E.publish(eul_msg)
    pub_blue_cone.publish(bcn_msg)
    pub_yellow_cone.publish(ycn_msg)
    
    rate.sleep()
    