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

import airsim
import rospy
import os
#import cone
#import path_m
import tools
import RL
import time
#import math
    
time.sleep(2)

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

#calibration white noise of velocity
init_v_x=car_state.kinematics_estimated.linear_velocity.x_val
init_v_y=car_state.kinematics_estimated.linear_velocity.y_val
init_v_z=car_state.kinematics_estimated.linear_velocity.z_val
#calibration white noise of velocity

safty_distance_turning=2.4
safty_distance_impact=2.3
collision_distance=1.8
punish_turnin=False
reward_sum=0
punish_batch_size=3
idx_punish=0
episode_time=1000
elapsed_time=0
time_stamp=0
summary=False
LOAD=False
collision=False
running_reward_max=0
running_reward=0
ep_total=0
ep_lr=0
max_reward_reset=0
rr_set=[]
reward_mean_set=[]
#ratio of the max max running reward and the average running reward.1 is the goal
reward_mean_max_rate=[]
#ratio of the max max running reward and the average running reward.1 is the goal
rr_idx=0
#set of runing reward
count=0
#minimal exploration wide of action
VAR_MIN_0 = 0.01
VAR_MIN_updated_0=0.001
VAR_MIN_1 = 0.01
VAR_MIN_updated_1=0.001
VAR_MIN_2 = 0.02
VAR_MIN_updated_2=0.002
#minimal exploration wide of action
#initial exploration wide of action
var0 = 0.1
var1 = 0.1
var2 = 0.1
#var1 = 0.1
#var2 = 0.1
#initial exploration wide of action
#dimension of action
ACTION_DIM = 6
#dimension of action
#action boundary
ACTION_BOUND0 = np.array([0,1])
ACTION_BOUND1 = np.array([0,1])
ACTION_BOUND2 = np.array([-1,1])

#action boundary
#action boundary a[0]*ACTION_BOUND[0],a[1]*ACTION_BOUND[1]
ACTION_BOUND=np.array([0.5,1])
ACTION_BOUND=np.array([0.5,0.5,1,1,1,1])
#action boundary a[0]*ACTION_BOUND[0],a[1]*ACTION_BOUND[1]
# learning rate for actor
LR_A = 1e-6
# learning rate for actor
# learning rate for critic
LR_C = 1e-6
# learning rate for critic
# reward discount
rd = 0.9
# reward discount
##after this learning number of main net update the target net of actor
#REPLACE_ITER_A = 256
##after this learning number of main net update the target net of actor
##after this learning number of main net update the target net of Critic
#REPLACE_ITER_C = 256
##after this learning number of main net update the target net of Critic
#occupied memory
MEMORY_CAPACITY = 131072
MEMORY_CAPACITY = 1024
#occupied memory
#size of memory slice
BATCH_SIZE = 128
#size of memory slice
#after this learning number of main net update the target net of actor
REPLACE_ITER_A = 1024
#after this learning number of main net update the target net of actor
#after this learning number of main net update the target net of Critic
REPLACE_ITER_C = 1024
#after this learning number of main net update the target net of Critic
probability=[]

#constant of distance measure
bound_lidar=20
#constant of distance measure

state=[[],[],[],[],[],[]]
input_dim=140
#inputs state of RL Agent
observation=np.zeros(input_dim)
#inputs state of RL Agent
#copy the state
observation_old=np.zeros(input_dim)
#copy the state
for t in range (0,input_dim):
    observation[t]=0
    observation_old[t]=0


#spawn=False
###constant of path
#half_path_wide=4
#delta_path=5
###constant of path
#cone.auto_spawn(spawn,half_path_wide,delta_path,car_state)

all_var=True
sess = tf.Session()

actor = RL.Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A)
critic = RL.Critic(sess, input_dim, ACTION_DIM, LR_C, rd, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = RL.Memory(MEMORY_CAPACITY, dims=2 * input_dim + ACTION_DIM + 1)
saver = tools.Saver(sess,LOAD,actor,critic,all_var)

while not rospy.is_shutdown():
    
    if count<30:
#        
#        car_controls.throttle=1
#        car_controls.steering=-1
#        client.setCarControls(car_controls)
        count=count+1
        
    else:
#        car_controls.throttle=0
#        car_controls.steering=0.5
#        car_controls.brake = 0.1
#        client.setCarControls(car_controls)
#        print("Apply brakes")
#        time.sleep(0.1)   # let car drive a bit
#        car_controls.brake = 0 #remove brake        
#        client.setCarControls(car_controls)
#        client.reset()
#        client.enableApiControl(False)
        count=0
#        count=count+1
        
#    print("count:",count)
    
    responses = client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    
    try:
        img_rgba = img1d.reshape(response.height, response.width, 4)
        img_rgba = np.flipud(img_rgba)
        airsim.write_png(os.path.normpath('greener.png'), img_rgba) 
        img_rgba = np.flipud(img_rgba)	
        image_msg = Image()
        image_msg.height = img_rgba.shape[0];
        image_msg.width =  img_rgba.shape[1];
        image_msg.encoding = 'rgba8';
        image_msg.step = img_rgba.shape[0]*img_rgba.shape[1]*4
        image_msg.data = img_rgba.tobytes();
    except:
        print("Image acquisition failed")

	#print(image_msg)
    

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
#    print("a_a_msg:",a_a_msg)
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
    
    (eul_msg.x, eul_msg.y, eul_msg.z) = cal.euler_from_quaternion([odo_msg.pose.pose.orientation.x, odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])

    act_msg.data.append(car_controls.throttle)
    act_msg.data.append(car_controls.brake)
    act_msg.data.append(car_controls.steering)
#    print("act_msg.data",act_msg.data)
#    
    elapsed_time=time.time()-time_stamp
    if summary==False:
        
        list_blue_cone=client.simGetObjectPoses("RightCone")
        list_yellow_cone=client.simGetObjectPoses("LeftCone")   
        
        
        bcn_msg.header.seq = 0
        bcn_msg.header.stamp = rospy.get_rostime()
        bcn_msg.header.frame_id = ""
 #       bcn_msg.poses=list_blue_cone
 
        list_cone_sensored=[]
        
        dis_close_blue_cone_1=1000
        blue_cone_close_1=[]
        
        for c in list_blue_cone:
#            print("c:",c)
            new_pose=Pose()
#            new_pose.position = Point()
            new_pose.position.x=c.position.x_val
            new_pose.position.y=c.position.y_val
            new_pose.position.z=c.position.z_val
            bcn_msg.poses.append(new_pose)
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
    #      
            if distance_cone< bound_lidar:
                
                list_cone_sensored.append([c.position.x_val,c.position.y_val])
                
            if distance_cone<dis_close_blue_cone_1:
                blue_cone_close_1=[c.position.x_val,c.position.y_val]
                dis_close_blue_cone_1=distance_cone
                
        dis_close_blue_cone_2=1000
        blue_cone_close_2=[]
        
        for c in list_blue_cone:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
            
            if distance_cone<dis_close_blue_cone_2 and distance_cone>dis_close_blue_cone_1:
                blue_cone_close_2=[c.position.x_val,c.position.y_val]
                dis_close_blue_cone_2=distance_cone
                
        dis_between_blue_cone=cal.calculate_r(blue_cone_close_1,blue_cone_close_2)
        
        sin_projection_blue=cal.calculate_projection(True,dis_close_blue_cone_1,dis_close_blue_cone_2,dis_between_blue_cone)[1]
        
        ycn_msg.header.seq = 0
        ycn_msg.header.stamp = rospy.get_rostime()
        ycn_msg.header.frame_id = ""
#        ycn_msg.poses=list_yellow_cone
        
        dis_close_yellow_cone_1=1000
        yellow_cone_close_1=[]
        
        for c in list_yellow_cone:
#            print("c:",c)
            new_pose=Pose()
            new_pose.position.x=c.position.x_val
            new_pose.position.y=c.position.y_val
            new_pose.position.z=c.position.z_val
            ycn_msg.poses.append(new_pose)
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
            
            if distance_cone< bound_lidar:
                
                list_cone_sensored.append([c.position.x_val,c.position.y_val])
        
            if distance_cone<dis_close_yellow_cone_1:
                yellow_cone_close_1=[c.position.x_val,c.position.y_val]
                dis_close_yellow_cone_1=distance_cone
                
        dis_close_yellow_cone_2=1000
        yellow_cone_close_2=[]
        
        for c in list_yellow_cone:
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
            
            if distance_cone<dis_close_yellow_cone_2 and distance_cone>dis_close_yellow_cone_1:
                yellow_cone_close_2=[c.position.x_val,c.position.y_val]
                dis_close_yellow_cone_2=distance_cone
                
        dis_between_yellow_cone=cal.calculate_r(yellow_cone_close_1,yellow_cone_close_2)
        
        sin_projection_yellow=cal.calculate_projection(True,dis_close_yellow_cone_1,dis_close_yellow_cone_2,dis_between_yellow_cone)[1]
        
        action = actor.choose_action(observation)
#        print("action:",action)
        action[0]=action[0]+0.5
#        action=[0,0]
        action[0] = np.clip(np.random.normal(action[0], var1), *ACTION_BOUND0)
        action[1] = np.clip(np.random.normal(action[1], var2), *ACTION_BOUND1)
#        print("action:",action)
#        print("action:",action)
        
        
#        if car_state.speed>0.4 :
#            action[0]=0
#            print("speed:",car_state.speed)
#            action[0]=np.random.random_sample()
#            print("action[0]:",action[0])
        
#        action[1]=np.random.random_sample()*(ACTION_BOUND1[1]-ACTION_BOUND1[0])+ACTION_BOUND1[0]
    #    print("action[1]:",action[1])
        
#        if car_controls.steering<1 and car_controls.steering>-1 and car_controls.steering+action[1]<1 and car_controls.steering+action[1]>-1:
        if car_controls.steering<1 and car_controls.steering>-1:
            
            car_controls.steering=float(action[1])
             
        if car_state.speed<=0.5:
            
            action[0]=float(np.random.random_sample()*(ACTION_BOUND0[1]-0))
          
            #print("action:",action)

        
        if sin_projection_blue<safty_distance_turning:
            
            punish_turning=True
            
            if sin_projection_blue<safty_distance_impact:
                
                car_controls.steering=-1
                action[1]=car_controls.steering           
            
            elif sin_projection_blue>safty_distance_impact and car_controls.steering>0:
                
                car_controls.steering=0.1
                action[1]=car_controls.steering

                
        if sin_projection_yellow<safty_distance_turning:
            
            punish_turning=True
            
            if sin_projection_yellow<safty_distance_impact:
                
                car_controls.steering=1
                action[1]=car_controls.steering           
            
            elif sin_projection_blue>safty_distance_impact and car_controls.steering<0:
                
                car_controls.steering=-0.1
                action[1]=car_controls.steering
        
           
        print("action:",action)
        car_controls.throttle=float(action[0]) 
#        car_controls.steering=float(action[1]) 
        client.setCarControls(car_controls)
        actor.angle.append(action[1])
        actor.accelerate.append(action[0])
        
        qua_msg.w=odo_msg.pose.pose.orientation.w
        qua_msg.x=odo_msg.pose.pose.orientation.x
        qua_msg.y=odo_msg.pose.pose.orientation.y
        qua_msg.z=odo_msg.pose.pose.orientation.z
          
        act_msg.data.append(car_controls.throttle)
        act_msg.data.append(car_controls.brake)
        act_msg.data.append(car_controls.steering)
         

        cone_sort_temp=[]
        cone_sort_end=[]
        
#        for c in list_blue_cone:   
#    
#            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
#    #      
#            if distance_cone< bound_lidar:
#                
#                list_cone_sensored.append([c.position.x_val,c.position.y_val])
                
#        for c in list_yellow_cone:
#    
#            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
#            
#            if distance_cone< bound_lidar:
#                
#                list_cone_sensored.append([c.position.x_val,c.position.y_val])
                
        for i in range (len(list_cone_sensored)):
                  
            cone_sort_temp.append([cal.calculate_sita_r(list_cone_sensored[i],[0,0]),list_cone_sensored[i]])
        
        cone_sort_temp=sorted(cone_sort_temp)
        
    #    print("cone_sort_temp:",cone_sort_temp)
        
        for i in range (len(cone_sort_temp)):
            
            cone_sort_end.append(cone_sort_temp[i][1])
            
    #    print("cone_sort_end:",cone_sort_end)
        
        state[0]=cone_sort_end
        if len(state[0])!=0:
            state[0]=np.vstack(state[0]).ravel()
    #    print("state[0]:",state[0])
        state[1]=np.vstack([vel_msg.x,vel_msg.y,vel_msg.z]).ravel()
        state[2]=np.vstack([acc_msg.x,acc_msg.y,acc_msg.z]).ravel()
        state[3]=np.vstack([a_a_msg.w,a_a_msg.x,a_a_msg.y,a_a_msg.z]).ravel()
        state[4]=np.vstack([a_v_msg.w,a_v_msg.x,a_v_msg.y,a_v_msg.z]).ravel()
        state[5]=np.vstack([odo_msg.pose.pose.orientation.w,odo_msg.pose.pose.orientation.x,odo_msg.pose.pose.orientation.y,odo_msg.pose.pose.orientation.z]).ravel()
        state_input=np.hstack((car_controls.steering,state[1],state[2],state[3],state[4],state[5]))  
    #    print("state_input:",state_input)
        
        for t in range(len(state[0])):
            
            observation[t]=state[0][t]
            
        for t in range(-len(state_input),0):
            
            observation[t]=state_input[t]
            
    #    print("observation:",observation)
           
        reward=car_state.speed
        
#        collision=client.simGetCollisionInfo().has_collisiond()
        
        for c in list_blue_cone: 
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
          
            if distance_cone<collision_distance:
                
                collision=True
                
        for c in list_yellow_cone: 
            
            distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
          
            if distance_cone<collision_distance:
                
                collision=True
#                
#        if coneback:
#            
#            if dis_back<collision_distance*1.5:
#                
#                collision=True
#            
#        if M.pointer > MEMORY_CAPACITY and punish_turning==True:
#            #                reward=-math.sqrt(reward**2)
#            idx_punish = M.pointer % M.capacity
#            punished_reward = M.read(idx_punish,punish_batch_size)[:, -input_dim - 1]
#            for t in range(0,len(punished_reward)):
#                    
#                if punished_reward[t]>0:
#                        
#                    reward_sum=reward_sum-2*math.sqrt(punished_reward[t]**2)
#                    punished_reward[t]=punished_reward[t]-2*math.sqrt(punished_reward[t]**2)
#                #                print("punished_reward",punished_reward)
#                #                punished_reward =-2*car.maxspeed/punished_reward
#                #                print("punished_reward_new",punished_reward)
#                M.write(idx_punish,punish_batch_size,punished_reward)
#            punished_reward = M.read(idx_punish,punish_batch_size)[:, -input_dim - 1]
#            #                print("punished_reward_new",punished_reward)
#            punish_turning=False
#            #            print("idx_punish:",idx_punish)  
                
        reward_sum=reward_sum+reward
#        print("Collision:",client.simGetCollisionInfo().has_collisiond)

#        if elapsed_time>episode_time or collision==True: 
        if collision==True: 
            
            for c in list_blue_cone:   
    
                distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
#               print("koordinate:",odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y,c.position.x_val,c.position.y_val)
                print("distance_cone:",distance_cone)
                
            for c in list_yellow_cone:   
    
                distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
#               print("koordinate:",odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y,c.position.x_val,c.position.y_val)
                print("distance_cone:",distance_cone)
#            if collision==True :
#                
#                reward=-pow(car_state.speed,3)
#                if M.pointer > MEMORY_CAPACITY:
#                    
#                    idx_punish = M.pointer % M.capacity
#                    punished_reward = M.read(idx_punish,2*punish_batch_size)[:, -input_dim - 1]
#                    for t in range(0,len(punished_reward)):
#                        
#                        if punished_reward[t]>0:
#                            
#                            reward_sum=reward_sum-math.sqrt(punished_reward[t]**2)
#                            punished_reward[t]=punished_reward[t]-3*math.sqrt(punished_reward[t]**2)
#        
#        #                print("punished_reward_new",punished_reward)
#                    M.write(idx_punish,2*punish_batch_size,punished_reward)
#                    punished_reward = M.read(idx_punish,2*punish_batch_size)[:, -input_dim - 1]
#        #                print("punished_reward_new",punished_reward)
           
            client.reset()
            reward_sum=reward_sum+reward
            running_reward=reward_sum
            reward_sum=0
    #        distance_set.append(distance)
    #        distance=0
            time_stamp = time.time()
            car_controls.steering=0
            car_controls.throttle=0
            car_controls.brake=0
            count=0
            collision=False
            summary=True
    

    
#        print("reward_sum:",reward_sum)
        M.store_transition(observation_old, action, reward, observation)
        
        #print("MEMORY_CAPACITY:",M.pointer)
        if M.pointer > MEMORY_CAPACITY:
            if var1>VAR_MIN:
                    
                var1 = max([var1*0.99999, VAR_MIN])    # decay the action randomness
                var2 = max([var2*0.99999, VAR_MIN]) 
            else:
                var1 = max([var1*0.999999, VAR_MIN_updated])    # decay the action randomness
                var2 = max([var2*0.999999, VAR_MIN_updated]) 
                
        #                var1 = max([0.98*pow(1.00228,(-ep_total)), VAR_MIN])
        #                var2 = max([0.98*pow(1.00228,(-ep_total)), VAR_MIN])
            b_M = M.sample(BATCH_SIZE)
        #                print("BATCH_SIZE:",BATCH_SIZE)
            b_s = b_M[:, :input_dim]
            b_a = b_M[:, input_dim: input_dim + ACTION_DIM]
            b_r = b_M[:, -input_dim - 1: -input_dim]
            b_s_ = b_M[:, -input_dim:]
            
        #                print("b_M:",b_M)
        #                print("b_s:",b_s)
        #                print("b_a:",b_a)
        #                print("b_r:",b_r)
        #                print("b_s_:",b_s_)
            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)
            
        observation_old=observation
        
    else:
        print("var1:",var1,"var2:",var2)
        print("MEMORY_pointer:",M.pointer)
        print("MEMORY_CAPACITY:",M.capacity)
        saver.save(sess,running_reward)

        if running_reward_max<running_reward and ep_total>1:
            
            running_reward_max=running_reward
            ep_lr=0
            max_reward_reset=max_reward_reset+1
        
  

        print("running_reward:",running_reward)
        print("max_running_reward:",running_reward_max)
        
        rr_set.append(running_reward)   
        rr_idx=rr_idx+1
        reward_mean_set.append(sum(rr_set)/rr_idx)
        
        print("reward_mean:",reward_mean_set[rr_idx-1])
        if rr_idx>5:
            
            reward_mean_max_rate.append(running_reward_max/reward_mean_set[rr_idx-1])
        #if RL.learning_rate>0.001:

        print("max_reward_reset:",max_reward_reset)

        plt.subplot(321)
        plt.plot(rr_set)  
        plt.xlabel('episode steps')
        plt.ylabel('runing reward')

        #plt.subplot(432)
        #plt.plot(vt)    # plot the episode vt
        #plt.xlabel('episode steps')
        #plt.ylabel('normalized state-action value')
     
        plt.subplot(323)
        plt.plot(reward_mean_set)  
        plt.xlabel('episode steps')
        plt.ylabel('reward_mean')
        
        plt.subplot(324)
        plt.plot(actor.angle)  
        plt.xlabel('episode steps')
        plt.ylabel('angle')
        
        plt.subplot(325)
        plt.plot(actor.accelerate)  
        plt.xlabel('episode steps')
        plt.ylabel('accelerate')
                
        #plt.subplot(4,2,6)
        #plt.plot(lr_set)  
        #plt.xlabel('episode steps')
        #plt.ylabel('learning rate')
        
        plt.subplot(3,2,6)
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
        actor.angle=[]
        actor.accelerate=[]

        ep_total=ep_total+1
        print("totaol train:",ep_total)
        print("LOAD:",LOAD)
        ep_lr=ep_lr+1
        print("lr ep :",ep_lr)
        summary=False
        
        
    pub_I.publish(image_msg)
    pub_acceleration.publish(acc_msg)
    pub_velocity.publish(vel_msg)
    pub_angular_acceleration.publish(a_a_msg)
    pub_angular_velocity.publish(a_v_msg)
    pub_Odometry_auto.publish(odo_msg)
    pub_E.publish(eul_msg)
    pub_blue_cone.publish(bcn_msg)
    pub_yellow_cone.publish(ycn_msg)
    
    rate.sleep()
    