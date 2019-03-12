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
#from std_msgs.msg import Float64


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
    client.setCarControls(car_controls)
    print("befehl:",data.data)

def set_steering(data):
    
    car_controls.steering=data.data
    client.setCarControls(car_controls)
    print("befehl:",data.data)
    
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
#sub_throt=rospy.Subscriber("throttle",Float64,set_throttle)
#sub_steer=rospy.Subscriber("steeringAngle",Float64, set_steering)

rate = rospy.Rate(60) # 10hz
car_state = client.getCarState()


while True:
    
    list_blue_cone=client.simGetObjectPoses("RightCone")
    list_yellow_cone=client.simGetObjectPoses("LeftCone")  
    coneback=client.simGetObjectPoses("finish")
    
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
    