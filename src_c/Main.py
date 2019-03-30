#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:23:00 2019

@author: spikezz
"""
#from airsim import ImageRequest
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
#from std_msgs.msg import Float64
from scipy.special import softmax
#import PythonTrackWrapper as ptw
import numpy as np
import calculate as cal
import tensorflow as tf
import rospy as rp
import Agent as ag
import main_function as mf
import ROS_Interface as ri

import airsim
import tools
import RL
import time


time.sleep(2)

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_state = client.getCarState()

initial_velocoty_noise=mf.Calibration_Speed_Sensor(car_state)

ros_publisher=ri.ROS_Publisher(rp)

rate = rp.Rate(60) # 10hz


##dimension of input data
input_dim = 60
##dimension of input data
##dimension of action
action_dim = 2
##dimension of action
action_bound=np.array([0.5,1])
#action_bound = np.array([-0.5,0.5],[-0.5,0.5],[-1,1])
# learning rate for actor
LR_A = 1.25e-8
# learning rate for actor
# imitation learning rate for actor
lr_i = 1.25e-4
# imitation learning rate for actor
# learning rate for critic
LR_C = 6.25e-8
# learning rate for critic
# reward discount
rd = 0.9
# reward discount
#after this learning number of main net update the target net of actor
replace_iter_a = 1024
#after this learning number of main net update the target net of actor
##after this learning number of main net update the target net of Critic
replace_iter_c = 1024
##after this learning number of main net update the target net of Critic

LOAD=False
summary=False
episode_counter=0

memory_capacity = 256
memory_capacity_bound = 65536

actionorigin=np.zeros(action_dim)
action_old=np.zeros(action_dim)
actionorigin_old=np.zeros(action_dim)

#inputs state of RL Agent
observation=np.zeros(input_dim)
#inputs state of RL Agent
#copy the state
observation_old=np.zeros(input_dim)

list_blue_cone=client.simGetObjectPoses("RightCone")
list_yellow_cone=client.simGetObjectPoses("LeftCone")  
coneback=client.simGetObjectPoses("finish")

sess = tf.Session()

agent_i=ag.Agent_Imitation(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
#agent_r=ag.Agent_Reinforcement(sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a)
memory_imitation = RL.Memory(memory_capacity,memory_capacity_bound, dims=2 * input_dim + 2*action_dim + 1 + 2)

all_var=True
reinforcement=False

saver = tools.Saver(sess,LOAD,agent_i.actor,None,all_var,reinforcement)

agent_i.actor.writer.add_graph(sess.graph,episode_counter)

#main loop
while not rp.is_shutdown():
    
    time_stamp = time.time()
    car_state = client.getCarState()
    
    ROS_car_state_message=ri.ROS_Car_State_Message_Creater(rp,car_state,initial_velocoty_noise)
    
    if summary==False:
     
        cone_message=ri.ROS_Cone_Message_Creater(rp,list_blue_cone,list_yellow_cone)
        
    
    
    
    
#    ros_publisher['pub_euler'].publish(ROS_car_state_message[5])
#    ros_publisher['pub_blue_cone'].publish(bcn_msg)
#    ros_publisher['pub_yellow_cone'].publish(ycn_msg)

    rate.sleep()
    