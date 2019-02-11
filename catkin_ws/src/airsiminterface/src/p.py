import airsim
from airsim import ImageRequest
import rospy
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
import numpy as np
import time as tm

#warte die Zeit zu kalibiren speed sensor

tm.sleep(3)

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

rospy.init_node('publish', anonymous=True)
pub_I = rospy.Publisher('UnrealImage', Image, queue_size=1000)
pub_V = rospy.Publisher('velocity', Vector3, queue_size=1000)
pub_A = rospy.Publisher('acceleration', Vector3, queue_size=1000)
pub_P = rospy.Publisher('position', Vector3, queue_size=1000)
pub_O = rospy.Publisher('orientation', Float32MultiArray, queue_size=1000)
pub_a = rospy.Publisher('action', Float32MultiArray, queue_size=1000)


rate = rospy.Rate(60) # 10hz
car_state = client.getCarState()

init_v_x=car_state.kinematics_estimated.linear_velocity.x_val
init_v_y=car_state.kinematics_estimated.linear_velocity.y_val
init_v_z=car_state.kinematics_estimated.linear_velocity.z_val

while not rospy.is_shutdown():
	
	car_controls.throttle = 1
	car_controls.steering = 1
	client.setCarControls(car_controls)

	responses = client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)])
	response = responses[0]

	img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
	img_rgba = img1d.reshape(response.height, response.width, 4)
	img_rgba = np.flipud(img_rgba)
	airsim.write_png(os.path.normpath('greener.png'), img_rgba) 
	img_rgba = np.flipud(img_rgba)	
	image_msg = Image()
	#image_msg.header= 
	image_msg.height = img_rgba.shape[0];
	image_msg.width =  img_rgba.shape[1];
	image_msg.encoding = 'rgba8';
	image_msg.step = img_rgba.shape[0]*img_rgba.shape[1]*4
	image_msg.data = img_rgba.tobytes();

	#print(image_msg)

	#print(speed,gear)
	acc_msg=Vector3()
	vel_msg=Vector3()
	pos_msg=Vector3()
	ort_msg=Float32MultiArray()
	act_msg=Float32MultiArray()

	#ort_msg.layout.dim.append('w_val')
	#ort_msg.layout.dim.append('x_val')
	#ort_msg.layout.dim.append('y_val')
	#ort_msg.layout.dim.append('z_val')

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
	#print(sc)
	#print(car_state.kinematics_estimated.linear_velocity)
	#print(car_state.kinematics_estimated.orientation)
    #print(speed,gear)
	#print(client.simGetGroundTruthEnvironment())
	#print(client.simGetGroundTruthKinematics())

	pub_I.publish(image_msg)
	pub_A.publish(acc_msg)
	pub_V.publish(vel_msg)
	pub_P.publish(pos_msg)
	pub_O.publish(ort_msg)
	pub_a.publish(act_msg)
	
	rate.sleep()





