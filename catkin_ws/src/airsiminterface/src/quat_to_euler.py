#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:00:33 2019

@author: spikezz
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Start up ROS pieces.
#PKG = 'my_pkg'
#import roslib; roslib.load_manifest(PKG)
import rospy
import tf

# ROS messages.
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
import calculate as cal

#from my_pkg.msg import Eulers

class QuatToEuler():
    
    def __init__(self):
        
        self.got_new_msg = False
        self.euler_msg=Vector3()


        # Create subscribers and publishers.
#        sub_imu   = rospy.Subscriber("imu", Imu, self.imu_callback)
        self.sub_odom  = rospy.Subscriber("odom", Odometry, self.odom_callback)
#        sub_ori  = rospy.Subscriber("orientation", Odometry, self.ori_callback)
        self.pub_euler = rospy.Publisher("euler", Vector3, queue_size=1000)

#        #Main while loop.
#        while not rospy.is_shutdown():
#            # Publish new data if we got a new message.
#            if self.got_new_msg:
#                pub_euler.publish(self.euler_msg)
#                self.got_new_msg = False

    # Odometry callback function.
    def odom_callback(self, msg):
        # Convert quaternions to Euler angles.
        (r, p, y) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.fill_euler_msg(msg, r, p, y)

    # IMU callback function.
#    def imu_callback(self, msg):
#        # Convert quaternions to Euler angles.
#        (r, p, y) = cal.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
#        self.fill_euler_msg(msg, r, p, y)
#        
#    def ori_callback(self, msg):
#        # Convert quaternions to Euler angles.
#        (r, p, y) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
#        self.fill_euler_msg(msg, r, p, y)
    # Fill in Euler angle message.
    def fill_euler_msg(self, msg, r, p, y):
        self.got_new_msg = True
#        self.euler_msg.header.stamp = msg.header.stamp
        self.euler_msg.x = r#roll
        self.euler_msg.y = p#pitch
        self.euler_msg.z = y#yaw

## Main function.    
#if __name__ == '__main__':
#    # Initialize the node and name it.
#    rospy.init_node('quat_to_euler')
#    # Go to class functions that do all the heavy lifting. Do error checking.
#    try:
#        quat_to_euler = QuatToEuler()
#    except rospy.ROSInterruptException: pass