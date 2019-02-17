import airsim
from airsim import ImageRequest
import rospy
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
import numpy as np
import time as tm
import path
import calculate as cal

#warte die Zeit zu kalibiren speed sensor
#client.simSpawnObject('/Game/test.test', airsim.Pose(position_val=airsim.Vector3r()))
tm.sleep(3)

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

#print(client.simSpawnObject('/Game/test.test_C', airsim.Pose(position_val=airsim.Vector3r(10,10,-10))))
print(client.simSpawnObject('/Game/cone_yellow.cone_yellow_C', airsim.Pose(position_val=airsim.Vector3r(0,0,-1))))

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

##constant of path
half_path_wide=80
##constant of path

#create the Group contains track mittle point
list_path_point=[]
#create the Group contains track mittle point

#start point
startpoint=Vector3()
startpoint.x=car_state.kinematics_estimated.position.x_val
startpoint.y=car_state.kinematics_estimated.position.y_val
startpoint.z=car_state.kinematics_estimated.position.z_val
last_point=startpoint
list_path_point.append(startpoint)
#start point

path_man=[]
corner=[]
corner.append(3)
#corner.append(6)
#corner.append(41)
#corner.append(43)
#corner.append(44)
#corner.append(45)
#corner.append(46)
#corner.append(47)
#corner.append(48)
#corner.append(49)
#corner.append(54)

for t in range (1,corner[0]):
    path_man.append([startpoint.x-4.9*t,startpoint.y-0.3*t])

#for t in range (1,5):
#    path_man.append([path_man[corner[0]-2][0]-49.74*t,path_man[corner[0]-2][1]-5*t])
#    
#for t in range (1,6):
#    path_man.append([path_man[corner[1]-2][0]-49*t,path_man[corner[1]-2][1]-10*t])
#
#for t in range (1,3):
#    path_man.append([path_man[corner[2]-2][0]-45.82*t,path_man[corner[2]-2][1]-20*t])
#
#path_man.append([path_man[corner[3]-2][0]-40,path_man[corner[3]-2][1]-30])
#path_man.append([path_man[corner[4]-2][0]-31,path_man[corner[4]-2][1]-39.23])
#path_man.append([path_man[corner[5]-2][0]-19.6,path_man[corner[5]-2][1]-46])
#path_man.append([path_man[corner[6]-2][0]+15,path_man[corner[6]-2][1]-47.7])
#path_man.append([path_man[corner[7]-2][0]+30,path_man[corner[7]-2][1]-40])
#path_man.append([path_man[corner[8]-2][0]+35,path_man[corner[8]-2][1]-35.71])
#for t in range (1,6):
#    path_man.append([path_man[corner[9]-2][0]+43.5*t,path_man[corner[9]-2][1]-24.65*t])
#    
#for t in range (1,23):
#    path_man.append([path_man[corner[10]-2][0]+47.37*t,path_man[corner[10]-2][1]-16*t])
#    
for pa in path_man:
    path_x=pa[0]
    path_y=pa[1]
     
    path_new=path.path(path_x,path_y)
    list_path_point.append(path_new)  
   
    line=[last_point,[path_new.x,path_new.y]]
     
    cone_x, cone_y = cal.calculate_t(line,1,half_path_wide,car.x,car.y)
    cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
    
#    list_cone_yellow.append(cone_new)
#    cone_s.add(cone_new)
#     
#    draw_yellow_cone.append([0,0])
#    dis_yellow.append(0)
#    vektor_yellow.append([0,0])
#    p=p+1
#    
#    cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide,car.x,car.y)
#    cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
#    list_cone_blue.append(cone_new)
#    cone_s.add(cone_new)
#   
#    draw_blue_cone.append([0,0])
#    dis_blue.append(0)
#    vektor_blue.append([0,0])
#    q=q+1
#
#    
#    last_point=[path_x+car.x,path_y+car.y]
#    draw_path.append([0,0])
#    j=j+1


while not rospy.is_shutdown():
	
	print(client.simGetObjectPoses("leftCone"))

	car_controls.throttle = 0
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
		#image_msg.header= 
		image_msg.height = img_rgba.shape[0];
		image_msg.width =  img_rgba.shape[1];
		image_msg.encoding = 'rgba8';
		image_msg.step = img_rgba.shape[0]*img_rgba.shape[1]*4
		image_msg.data = img_rgba.tobytes();
	except:
		print("Image acquisition failed")

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

    



