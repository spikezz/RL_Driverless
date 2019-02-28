import airsim
from airsim import ImageRequest
import rospy
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import time as tm
import calculate as cal
import cone
import path_m
import RL
import tensorflow as tf

tm.sleep(3)

client = airsim.CarClient()
client.confirmConnection()
#client.enableApiControl(True)
car_controls = airsim.CarControls()

rospy.init_node('publish', anonymous=True)
pub_I = rospy.Publisher('UnrealImage', Image, queue_size=1000)
pub_V = rospy.Publisher('velocity', Vector3, queue_size=1000)
pub_A = rospy.Publisher('acceleration', Vector3, queue_size=1000)
pub_aa = rospy.Publisher('angular_acceleration', Quaternion, queue_size=1000)
pub_av = rospy.Publisher('angular_velocity', Quaternion, queue_size=1000)
pub_E = rospy.Publisher('O_to_E', Vector3, queue_size=1000)
pub_O = rospy.Publisher('Odometry', Odometry, queue_size=1000)
pub_a = rospy.Publisher('action', Float32MultiArray, queue_size=1000)
#pub_Q = rospy.Publisher('Quaternion', Quaternion, queue_size=1000)

rate = rospy.Rate(60) # 10hz
car_state = client.getCarState()

#calibration white noise of velocity
init_v_x=car_state.kinematics_estimated.linear_velocity.x_val
init_v_y=car_state.kinematics_estimated.linear_velocity.y_val
init_v_z=car_state.kinematics_estimated.linear_velocity.z_val
#calibration white noise of velocity

#dimension of action
ACTION_DIM = 2
#dimension of action
#action boundary
ACTION_BOUND0 = np.array([-0.5,0.5])
ACTION_BOUND1 = np.array([-45,45])
#action boundary
#action boundary a[0]*ACTION_BOUND[0],a[1]*ACTION_BOUND[1]
ACTION_BOUND=np.array([1,3])
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
REPLACE_ITER_A = 1100
#after this learning number of main net update the target net of actor
#after this learning number of main net update the target net of Critic
REPLACE_ITER_C = 1000
#after this learning number of main net update the target net of Critic
#occupied memory
MEMORY_CAPACITY = 1000
#occupied memory

#constant of distance measure
bound_lidar=20
#constant of distance measure

#create the Group contains yellow cone
list_cone_yellow=0
#create the Group contains yellow cone
#create the Group contains blue cone
list_cone_blue=0
#create the Group contains blue cone
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


auto_spawn=False

#qe_convert=qe.QuatToEuler()

if auto_spawn==True:
    
    ##constant of path
    half_path_wide=2.5
    delta_path=5
    ##constant of path
    #create the Group contains track mittle point
    list_path_point=[]
    #create the Group contains track mittle point
    
    
    #start point
    startpoint=path_m.path_m(0,0,-5)
    last_point=[startpoint.x,startpoint.y]
    #start point
    
    cone_x, cone_y = startpoint.x-half_path_wide,startpoint.y
    print("cone_x:",cone_x)
    print("cone_y:",cone_y)
    
    cone_new=cone.cone(1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
    
    list_cone_yellow.append(cone_new)
    
    print("cone new:",cone_new)
    
    
    cone_x, cone_y = startpoint.x+half_path_wide,startpoint.y
    print("cone_x:",cone_x)
    print("cone_y:",cone_y)
    
    cone_new=cone.cone(-1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
    
    list_cone_yellow.append(cone_new)
    
    print("cone new:",cone_new)
    
    corner=[]
    corner.append(14)
    
    print("startpoint:",startpoint)
    
    for t in range (1,corner[0]):
        print("corner:",t)
        path_new= path_m.path_m(startpoint.x,startpoint.y+delta_path*t,-5)
        list_path_point.append(path_new)
        print("path new:",path_new)
    
    
    for pa in list_path_point:
        print("pa",pa)
        line=[[last_point[0],last_point[1]],[pa.x,pa.y]]
        print("line:",line)
         
        cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide)
        print("cone_x:",cone_x)
        print("cone_y:",cone_y)
        cone_new=cone.cone(1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
        
        list_cone_yellow.append(cone_new)
        
        print("cone new:",cone_new)
    
        cone_x, cone_y = cal.calculate_t(line,1,half_path_wide)
        print("cone_x:",cone_x)
        print("cone_y:",cone_y)
        cone_new=cone.cone(-1,cone_x,cone_y,car_state.kinematics_estimated.position.z_val)
        
        list_cone_blue.append(cone_new)
        
        print("cone new:",cone_new)
    
        last_point=[pa.x,pa.y]

        
sess = tf.Session()

actor = RL.Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A, REPLACE_ITER_A)
critic = RL.Critic(sess, input_dim, ACTION_DIM, LR_C, rd, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = RL.Memory(MEMORY_CAPACITY, dims=2 * input_dim + ACTION_DIM + 1)
saver = RL.Saver(sess)

while not rospy.is_shutdown():
	
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)

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
    
#    qua_msg.w=odo_msg.pose.pose.orientation.w
#    qua_msg.x=odo_msg.pose.pose.orientation.x
#    qua_msg.y=odo_msg.pose.pose.orientation.y
#    qua_msg.z=odo_msg.pose.pose.orientation.z
    
    (eul_msg.x, eul_msg.y, eul_msg.z) = cal.euler_from_quaternion([odo_msg.pose.pose.orientation.x, odo_msg.pose.pose.orientation.y, odo_msg.pose.pose.orientation.z, odo_msg.pose.pose.orientation.w])
    
    act_msg.data.append(car_controls.throttle)
    act_msg.data.append(car_controls.steering)
    
    list_blue_cone=client.simGetObjectPoses("LeftCone")
    list_cone_yellow=client.simGetObjectPoses("RightCone")   
    list_cone_sensored=[]
    cone_sort_temp=[]
    cone_sort_end=[]
    
    for c in list_blue_cone:   

        distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
#      
        if distance_cone< bound_lidar:
            
            list_cone_sensored.append([c.position.x_val,c.position.y_val])
            
    for c in list_cone_yellow:

        distance_cone=cal.calculate_r([odo_msg.pose.pose.position.x,odo_msg.pose.pose.position.y],[c.position.x_val,c.position.y_val])
        
        if distance_cone< bound_lidar:
            
            list_cone_sensored.append([c.position.x_val,c.position.y_val])
    for i in range (len(list_cone_sensored)):
              
        cone_sort_temp.append([cal.calculate_sita_r(list_cone_sensored[i],[0,0]),list_cone_sensored[i]])
    
    cone_sort_temp=sorted(cone_sort_temp)
    
#    print("cone_sort_temp:",cone_sort_temp)
    
    for i in range (len(cone_sort_temp)):
        
        cone_sort_end.append(cone_sort_temp[i][1])
        
#    print("cone_sort_end:",cone_sort_end)
    
    state[0]=cone_sort_end 
    state[0]=np.vstack(state[0]).ravel()
    print("state[0]:",state[0])
    state[1]=np.vstack([vel_msg.x,vel_msg.y,vel_msg.z]).ravel()
    state[2]=np.vstack([acc_msg.x,acc_msg.y,acc_msg.z]).ravel()
    state[3]=np.vstack([a_a_msg.w,a_a_msg.x,a_a_msg.y,a_a_msg.z]).ravel()
    state[4]=np.vstack([a_v_msg.w,a_v_msg.x,a_v_msg.y,a_v_msg.z]).ravel()
    state[5]=np.vstack([odo_msg.pose.pose.orientation.w,odo_msg.pose.pose.orientation.x,odo_msg.pose.pose.orientation.y,odo_msg.pose.pose.orientation.z]).ravel()
    state_input=np.hstack((state[1],state[2],state[3],state[4],state[5]))  
    print("state_input:",state_input)
    
    for t in range(len(state[0])):
        
        observation[t]=state[0][t]
        
    for t in range(-len(state_input),0):
        
        observation[t]=state_input[t]
        
    print("observation:",observation)
       
#

    pub_I.publish(image_msg)
    pub_A.publish(acc_msg)
    pub_V.publish(vel_msg)
    pub_aa.publish(a_a_msg)
    pub_av.publish(a_v_msg)
    pub_O.publish(odo_msg)
    pub_E.publish(eul_msg)
    pub_a.publish(act_msg)
    
    rate.sleep()
    