import airsim
from airsim import ImageRequest
import rospy
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import numpy as np
import time as tm
import calculate as cal
import cone
import path_m

#warte die Zeit zu kalibiren speed sensor
#client.simSpawnObject('/Game/test.test', airsim.Pose(position_val=airsim.Vector3r()))
tm.sleep(3)

client = airsim.CarClient()
client.confirmConnection()
#client.enableApiControl(True)
car_controls = airsim.CarControls()

#print(client.simSpawnObject('/Game/test.test_C', airsim.Pose(position_val=airsim.Vector3r(10,10,-10))))
#print(client.simSpawnObject('/Game/cone_yellow.cone_yellow_C', airsim.Pose(position_val=airsim.Vector3r(0,0,-1))))

rospy.init_node('publish', anonymous=True)
pub_I = rospy.Publisher('UnrealImage', Image, queue_size=1000)
pub_V = rospy.Publisher('velocity', Vector3, queue_size=1000)
pub_A = rospy.Publisher('acceleration', Vector3, queue_size=1000)
pub_P = rospy.Publisher('position', Vector3, queue_size=1000)
pub_O = rospy.Publisher('orientation', Float32MultiArray, queue_size=1000)
pub_a = rospy.Publisher('action', Float32MultiArray, queue_size=1000)

rate = rospy.Rate(60) # 10hz
car_state = client.getCarState()

#calibration white noise of velocity
init_v_x=car_state.kinematics_estimated.linear_velocity.x_val
init_v_y=car_state.kinematics_estimated.linear_velocity.y_val
init_v_z=car_state.kinematics_estimated.linear_velocity.z_val
#calibration white noise of velocity

#constant of distance measure
bound_lidar=10
#constant of distance measure

auto_spawn=True

if auto_spawn==True:
    
    ##constant of path
    half_path_wide=2.5
    delta_path=5
    ##constant of path
    #create the Group contains yellow cone
    list_cone_yellow=[]
    #create the Group contains yellow cone
    #create the Group contains blue cone
    list_cone_blue=[]
    #create the Group contains blue cone
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

#print(client.simGetObjectPoses("leftCone"))
#list_yellow_cone=client.simGetObjectPoses("LeftCone")
#list_blue_cone=client.simGetObjectPoses("RightCone")
#print(list_blue_cone[0].position.x_val)
while not rospy.is_shutdown():
	
#	print(client.simGetObjectPoses("leftCone"))

	car_controls.throttle = 0
	car_controls.steering = 0
	client.setCarControls(car_controls)
    
#    list_yellow_cone=client.simGetObjectPoses("LeftCone")
#    list_blue_cone=client.simGetObjectPoses("RightCone")
#    
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
	pos_msg=Vector3()
	ort_msg=Float32MultiArray()
	act_msg=Float32MultiArray()

	car_state = client.getCarState()

	acc_msg.x=car_state.kinematics_estimated.linear_acceleration.x_val
	acc_msg.y=car_state.kinematics_estimated.linear_acceleration.y_val
	acc_msg.z=car_state.kinematics_estimated.linear_acceleration.z_val

	vel_msg.x=car_state.kinematics_estimated.linear_velocity.x_val-init_v_x
	vel_msg.y=car_state.kinematics_estimated.linear_velocity.y_val-init_v_y
	vel_msg.z=car_state.kinematics_estimated.linear_velocity.z_val-init_v_z


	pos_msg.x=car_state.kinematics_estimated.position.x_val
	pos_msg.y=car_state.kinematics_estimated.position.y_val
	pos_msg.z=car_state.kinematics_estimated.position.z_val

	ort_msg.data.append(car_state.kinematics_estimated.orientation.w_val)
	ort_msg.data.append(car_state.kinematics_estimated.orientation.x_val)
	ort_msg.data.append(car_state.kinematics_estimated.orientation.y_val)
	ort_msg.data.append(car_state.kinematics_estimated.orientation.z_val)

	act_msg.data.append(car_controls.throttle)
	act_msg.data.append(car_controls.steering)

	#print(car_state)
	#print(vel_msg)
	#print(pos_msg)
	#print(ort_msg)
	#print(car_state.kinematics_estimated.linear_velocity)
	#print(car_state.kinematics_estimated.orientation)
    #print(speed,gear)
	#print(client.simGetGroundTruthEnvironment())
	#print(client.simGetGroundTruthKinematics())

#    for cone_blue in list_blue_cone:
#        coordinate_cone=[]
#        coordinate_cone.append(cone_blue.position.x_val)
#        coordinate_cone.append(cone_blue.position.y_val)
#        coordinate_cone.append(cone_blue.position.z_val)
#        
#        coordinate_car=[]
#        coordinate_car.append(pos_msg.x)
#        coordinate_car.append(pos_msg.y)
#        coordinate_car.append(pos_msg.z)
#        if cal.calculate_r(coordinate_cone,coordinate_car)<:
#            
#    print(list_blue_cone[0].position.x_val)
#
#









	pub_I.publish(image_msg)
	pub_A.publish(acc_msg)
	pub_V.publish(vel_msg)
	pub_P.publish(pos_msg)
	pub_O.publish(ort_msg)
	pub_a.publish(act_msg)
	
	rate.sleep()

   



