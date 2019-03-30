#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:05:50 2019

@author: spikezz
"""
import calculate as cal
import time

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image


def ROS_Publisher(rp):
    
    ros_publisher={'pub_Image':rp.Publisher('UnrealImage', Image, queue_size=1000),\
                   'pub_euler':rp.Publisher('O_to_E', Vector3, queue_size=1000),\
                   'pub_velocity':rp.Publisher('velocity', Vector3, queue_size=1000),\
                   'pub_acceleration':rp.Publisher('acceleration', Vector3, queue_size=1000),\
                   'pub_angular_acceleration':rp.Publisher('angular_acceleration', Quaternion, queue_size=1000),\
                   'pub_angular_velocity':rp.Publisher('angular_velocity', Quaternion, queue_size=1000),\
                   'pub_Odometry_auto':rp.Publisher('Odometry_auto', Odometry, queue_size=1000),\
                   'pub_action':rp.Publisher('action', Float32MultiArray, queue_size=1000),\
                   'pub_blue_cone':rp.Publisher('rightCones', PoseArray, queue_size=1000),\
                   'pub_yellow_cone':rp.Publisher('leftCones', PoseArray, queue_size=1000)}
                  
    rp.init_node('publish', anonymous=True)
    
    
    return ros_publisher

def ROS_Car_State_Message_Creater(rospy,car_state,initial_velocoty_noise):
    
    ros_state_message_=[]
    acc_msg=Vector3()
    vel_msg=Vector3()
    a_a_msg=Quaternion()
    a_v_msg=Quaternion()
    odo_msg=Odometry()
    eul_msg=Vector3()
    
    acc_msg.x=car_state.kinematics_estimated.linear_acceleration.x_val
    acc_msg.y=car_state.kinematics_estimated.linear_acceleration.y_val
    acc_msg.z=car_state.kinematics_estimated.linear_acceleration.z_val
    ros_state_message_.append(acc_msg)
    
    vel_msg.x=car_state.kinematics_estimated.linear_velocity.x_val-initial_velocoty_noise[0]
    vel_msg.y=car_state.kinematics_estimated.linear_velocity.y_val-initial_velocoty_noise[1]
    vel_msg.z=car_state.kinematics_estimated.linear_velocity.z_val-initial_velocoty_noise[2]
    ros_state_message_.append(vel_msg)
    
    a_a_msg.w=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().w_val
    a_a_msg.x=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().x_val
    a_a_msg.y=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().y_val
    a_a_msg.z=car_state.kinematics_estimated.angular_acceleration.to_Quaternionr().z_val
    ros_state_message_.append(a_a_msg)
    
    a_v_msg.w=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().w_val
    a_v_msg.x=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().x_val
    a_v_msg.y=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().y_val
    a_v_msg.z=car_state.kinematics_estimated.angular_velocity.to_Quaternionr().z_val
    ros_state_message_.append(a_v_msg)

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
    ros_state_message_.append(odo_msg)
    
    (eul_msg.x, eul_msg.y, eul_msg.z) = cal.euler_from_quaternion([odo_msg.pose.pose.orientation.x, odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])
    ros_state_message_.append(eul_msg)
    
    return ros_state_message_

    
def Cone_Coordinate_Extractor(cone,cone_message):
    
    new_pose=Pose()
    new_pose.position.x=cone.position.x_val
    new_pose.position.y=cone.position.y_val
    new_pose.position.z=cone.position.z_val
    cone_message.poses.append(new_pose)

def ROS_Cone_Message_Creater(rp,list_blue_cone,list_yellow_cone):
    
    cone_message=[]
    bcn_msg=PoseArray()
    ycn_msg=PoseArray()
    
    bcn_msg.header.seq = 0
    bcn_msg.header.stamp = rp.get_rostime()
    bcn_msg.header.frame_id = ""

    ycn_msg.header.seq = 0
    ycn_msg.header.stamp = rp.get_rostime()
    ycn_msg.header.frame_id = ""
    
    for cone in list_blue_cone:
        
        Cone_Coordinate_Extractor(cone,bcn_msg)
        
    for cone in list_yellow_cone:
        
        Cone_Coordinate_Extractor(cone,ycn_msg)
        
    cone_message.append(bcn_msg,ycn_msg)
    
    return cone_message
