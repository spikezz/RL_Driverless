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
    
def set_brake(data):
    
    car_controls.brake=data.data
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

sin_projection_yellow=0
sin_projection_blue=0

safty_distance_turning=2.2
safty_distance_impact=2.0
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
collide_finish=False
running_reward_max=0
running_reward=0
ep_total=0
ep_lr=0
action_ori=[[],[],[],[],[],[]]
max_reward_reset=0
rr_set=[]
#set of runing reward
rr=[]
#set of runing reward
reward_mean_set=[]
reward_ep=[]
reward_one_ep_mean=[]
#average whole episode reward of the whole training  process
reward_mean=[]
#average whole episode reward of the whole training  process
#ratio of the max max running reward and the average running reward.1 is the goal
reward_mean_max_rate=[]
#ratio of the max max running reward and the average running reward.1 is the goal
rr_idx=0
#set of runing reward
count=0
#minimal exploration wide of action
VAR_MIN= [0.01,0.01,0.02]
VAR_MIN_updated=[0.001,0.001,0.002]
#minimal exploration wide of action
#initial exploration wide of action
var= [0.1,0.1,0.1]
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
#after this learning number of main net update the target net of actor
REPLACE_ITER_A = 1024
#after this learning number of main net update the target net of actor
#after this learning number of main net update the target net of Critic
REPLACE_ITER_C = 1024
#after this learning number of main net update the target net of Critic
#occupied memory
MEMORY_CAPACITY = 131072
MEMORY_CAPACITY = 1024
#occupied memory
#size of memory slice
BATCH_SIZE = 128
#size of memory slice
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

actor = RL.Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A,REPLACE_ITER_A)
critic = RL.Critic(sess, input_dim, ACTION_DIM, LR_C, rd, REPLACE_ITER_A,actor.a, actor.a_)
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
        coneback=client.simGetObjectPoses("finish")
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
        
        distance_coneback=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[coneback[0].position.x_val,coneback[0].position.y_val])
        print("sin_projection_yellow:",sin_projection_yellow,"\n","sin_projection_blue:",sin_projection_blue,"\n","distance_coneback:",distance_coneback)
        action = actor.choose_action(observation)
        probability=ss.softmax(np.array([action[3],action[4],action[5]]))
#            print("actionout",action)

        action[0] =action[0] +(ACTION_BOUND0[1]-ACTION_BOUND0[0])/2
        action[1] =action[1] +(ACTION_BOUND1[1]-ACTION_BOUND1[0])/2

        action_ori[0].append(action[0])
        action_ori[1].append(action[1])
        action_ori[2].append(action[2])
        action_ori[3].append(probability[0])
        action_ori[4].append(probability[1])
        action_ori[5].append(probability[2])
        
#            print("probability",probability)
#            action_ori3.append(action[3])
#            action_ori4.append(action[4])
#            action_ori5.append(action[5])
#            print("action1",action)
        
        action[0] = np.clip(np.random.normal(action[0], var[0]), *ACTION_BOUND0)
        action[1] = np.clip(np.random.normal(action[1], var[1]), *ACTION_BOUND1)
        action[2] = np.clip(np.random.normal(action[2], var[2]), *ACTION_BOUND2)
     
        if action[2]<1 and action[2]>-1 :
            
            car_controls.steering=float(action[2])
            
        else:
            
            action[2]=0
#            
        if sin_projection_blue<safty_distance_turning:
            
            punish_turning=True
            
            if sin_projection_blue<safty_distance_impact:
                
                car_controls.steering=-1
                action[2]=car_controls.steering       
#                    while debug:
#                        for event in pygame.event.get():
#                            if event.unicode == '\d':
#                                debug=False
            
            elif car_controls.steering>-1 and car_controls.steering<=0:
                
                car_controls.steering-=0.1
                action[2]=car_controls.steering
                
            elif car_controls.steering<-0.9 or car_controls.steering>0:
                
                car_controls.steering=0
                action[2]=car_controls.steering

                
        elif sin_projection_yellow<safty_distance_turning:
            
            punish_turning=True
            
            if sin_projection_yellow<safty_distance_impact:
                
                car_controls.steering=1
                action[2]=car_controls.steering   
                
            elif car_controls.steering<1 and car_controls.steering>=0:
                
                car_controls.steering+=0.1
                action[2]=car_controls.steering
                
            elif car_controls.steering>0.9 or car_controls.steering<0:
                
                car_controls.steering=0
                action[2]=car_controls.steering
                
                
        actor.angle.append(action[2])
        choice=np.random.choice(range(len(probability)),p=probability)
#            print("sess.run(probability):",sess.run(probability))
#            print("choice:",choice)
        if choice==0:
            car_controls.brake=0
            car_controls.throttle=float(action[0]) #beschleunigen
            actor.accelerate.append(action[0])
            actor.brake.append(0)
        elif choice==1:
            car_controls.throttle=0
            car_controls.brake=float(action[1]) #bremsen
            actor.brake.append(action[1])
            actor.accelerate.append(0)
            
        elif choice==2:
            actor.accelerate.append(0)
            actor.brake.append(0)
        #        car_controls.steering=float(action[1]) 
        client.setCarControls(car_controls)
        
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
#        print("Collision:",client.simGetCollisionInfo().has_collisiond)
        if sin_projection_yellow<collision_distance or sin_projection_blue<collision_distance:
                
            collision=True
                
        if distance_coneback<collision_distance*10:
            
            collide_finish=True

        reward_sum=reward_sum+reward
#        if elapsed_time>episode_time or collision==True: 
        if collision==True or collide_finish==True: 
                
            if collision==True :
#                
                reward=-pow(car_state.speed,3)
 
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
            summary=True
#            distance_set.append(distance)
#            distance=0
#        print("reward_sum:",reward_sum)
        reward_ep.append(reward)
        M.store_transition(observation_old, action, reward, observation)
        
        #print("MEMORY_CAPACITY:",M.pointer)
        if M.pointer > MEMORY_CAPACITY:
            if var[0]>VAR_MIN[0]:
                
                var[0] = max([var[0]*0.99999, VAR_MIN[0]])    # decay the action randomness
                
            else:
                
                var[0] = max([var[0]*0.999999, VAR_MIN_updated[0]])    # decay the action randomness
                
            if var[1]>VAR_MIN[1]:
                
                var[1] = max([var[1]*0.99999, VAR_MIN[1]]) 
                
            else:
                
                var[1] = max([var[1]*0.999999, VAR_MIN_updated[1]])
                 
            if var[2]>VAR_MIN[2]:
                
                var[2] = max([var[2]*0.99999, VAR_MIN[2]]) 
                
            else:
                
                var[2] = max([var[2]*0.999999, VAR_MIN_updated[2]])
                
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :input_dim]
            b_a = b_M[:, input_dim: input_dim + ACTION_DIM]
            b_r = b_M[:, -input_dim - 1: -input_dim]
            b_s_ = b_M[:, -input_dim:]
            
        #                print("b_M:",b_M)
        #                print("b_s:",b_s)
        #                print("b_a:",b_a)
        #                print("b_r:",b_r)
        #                print("b_s_:",b_s_)
            critic.learn(b_s, b_a, b_r, b_s_,ep_total)
            actor.learn(b_s)
            
        observation_old=observation
        
    else:
           
        reward_ep_mean=np.mean(reward_ep)
        reward_one_ep_mean.append(reward_ep_mean)
        ep_lr=ep_lr+1
        
        if running_reward_max<running_reward and ep_total>1:
            
            running_reward_max=running_reward
            ep_lr=0
            max_reward_reset=max_reward_reset+1
            
        rr.append(running_reward)   
        rr_idx=rr_idx+1
        reward_mean.append(sum(rr)/rr_idx)
        
        if rr_idx>5:
            
            reward_mean_max_rate.append(running_reward_max/reward_mean[rr_idx-1])
        Sum=tools.Summary()
        Sum.summary(LOAD,var,M.pointer,M.capacity,reward_ep,running_reward,
                running_reward_max,reward_mean,rr_idx,max_reward_reset,
                action_ori,actor,reward_one_ep_mean,rr,critic,reward_mean_max_rate,ep_lr)

#        saver.save(sess,running_reward)
        
        actor.angle=[]
        actor.accelerate=[]
        actor.brake=[]
        action_ori=[[],[],[],[],[],[]]
        reward_ep=[]

        ep_total=ep_total+1
        print("totaol train:",ep_total)
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
    