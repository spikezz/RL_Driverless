# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:14:08 2018

@author: Asgard
"""
import tensorflow as tf
import sys, pygame, math
import player,maps,tracks,camera,traffic_cone , path
import canvas as cv
import calculation as cal
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os
import shutil
from loader import load_image
from operator import itemgetter, attrgetter
from pygame.locals import *

np.random.seed(1)
tf.set_random_seed(1)


#LR_A = 1e-6  # learning rate for actor
#LR_C = 1e-6  # learning rate for critic
#rd = 0.9  # reward discount
#REPLACE_ITER_A = 1100
#REPLACE_ITER_C = 1000
#MEMORY_CAPACITY = 500
#BATCH_SIZE = 2000
#VAR_MIN = 0.1

H1=180
H2=10
input_dim = 80
#half_Max_angle=45
#ACTION_DIM = 2
#ACTION_BOUND0 = np.array([-0.1,0.5])
#ACTION_BOUND1 = np.array([-45,45])
#ACTION_BOUND=np.array([1,3])

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='s_')




class Actor(object):
    def __init__(self, sess, n_a, action_bound, LR, t_replace_iter):
        self.sess = sess
        self.a_dim = n_a
        self.action_bound = action_bound
        self.lr = LR
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        
        self.angle=[]
        self.accelerate=[]
        
        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            #init_w = tf.contrib.layers.xavier_initializer()
            init_w=tf.random_uniform_initializer(-0.23,0.23)
            init_b = tf.constant_initializer(0.000)
            
            layer1 = tf.layers.dense(s, H1, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l1',trainable=trainable)
            layer2 = tf.layers.dense(layer1, H1, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l2',trainable=trainable)
            #layer3 = tf.layers.dense(layer2, H2, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l3',trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            #init_w = tf.contrib.layers.xavier_initializer()
            init_w=tf.random_uniform_initializer(-0.23,0.23)
            init_b = tf.constant_initializer(0.00)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layer1 = tf.layers.dense(net, H1, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            layer2 = tf.layers.dense(layer1, H2, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(layer2,1,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


#initial stuff
pygame.init()
screen = pygame.display.set_mode((1360,768),0)
pygame.display.set_caption('Karat Simulation')
font = pygame.font.Font(None, 40)
#initial stuff

##black background for render when car is out of map 
background = pygame.Surface(screen.get_size())
background = background.convert_alpha()
background.fill((1, 1, 1))
##black background for render when car is out of map 

##the transparent canvas for drawing the necessary geometric relationship.The Zero point is at (cam.x,cam.y)
canvas = pygame.Surface(screen.get_size(),SRCALPHA ,32)
canvas = canvas.convert_alpha()
canvas.set_alpha(0)
##the transparent canvas for drawing the necessary geometric relationship.The Zero point is at (cam.x,cam.y)

#testcode for shell
#CENTER_X = 800
#CENTER_Y = 450

##find the center of screen
CENTER_X =  float(pygame.display.Info().current_w /2)
CENTER_Y =  float(pygame.display.Info().current_h /2)
CENTER=(CENTER_X,CENTER_Y)
##find the center of screen

##constant of path
half_path_wide=80
##constant of path

##create some objects
clock = pygame.time.Clock()
car = player.Player()
cam = camera.Camera()
##create some objects

##create the spriteGroup contains objects
list_cone_yellow=[]
list_cone_blue=[]
list_path_point=[]
map_s= pygame.sprite.Group()
player_s= pygame.sprite.Group()
tracks_s= pygame.sprite.Group()
cone_s  = pygame.sprite.Group()
path_s  = pygame.sprite.Group()
cone_h  = pygame.sprite.Group()
##create the spriteGroup contains objects

##tracks  initalize. tracks are points left  while driving
tracks.initialize()
cam.set_pos(car.x, car.y)
##tracks  initalize. tracks are points left  while driving

##add objects
#car
player_s.add(car)
#car
#cone
coneb=traffic_cone.cone(CENTER[0],CENTER[1]-half_path_wide,-1,car.x,car.y)
coney=traffic_cone.cone(CENTER[0],CENTER[1]+half_path_wide,1,car.x,car.y)
coneback=traffic_cone.cone(CENTER[0]+half_path_wide/2,CENTER[1],-1,car.x,car.y)
cone_s.add(coneb)
cone_s.add(coney)
cone_h.add(coneback)
list_cone_blue.append(coneb)
list_cone_yellow.append(coney)
#cone
#middle point
startpoint=path.path(CENTER[0],CENTER[1],car.x,car.y)
path_s.add(startpoint)
list_path_point.append(startpoint)
#middle point
##add objects

##set orientation
car.set_start_direction(90)
#the turning angle of the wheel 
angle=0
#the turning angle of the wheel  
#saved angle
angle_old=0
#saved angle
##set orientation

###initial Model of the car
##specification of the car
half_Max_angle=45
half_middle_axis_length=20
half_horizontal_axis_length=17
radius_of_wheel=10
el_length=500
##specification of the car
##action relevant
turing_speed=1.0
##action relevant
model=cv.initialize_model(CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)
###initial Model of the car

##count/COUNT_FREQUENZ is the real time
#every loop +1 for timer
count=0
#every loop +1 for timer
#FPS Frame(loop times) per second
COUNT_FREQUENZ=500
#FPS Frame(loop times) per second
#switch for timer
start_timer=False
#switch for timer
##count/COUNT_FREQUENZ is the real time

##map picture size
FULL_TILE = 1000
##map picture size

##offset of the start position of the car
xc0=car.x
yc0=car.y
##offset of the start position of the car

##the old (initial) position for speed measurements
x_old=car.x
y_old=car.y
##the old (initial) position for speed measurements

##read the maps and draw
for tile_num in range (0, len(maps.map_tile)):

    #add submap idx to array
    maps.map_files.append(load_image(maps.map_tile[tile_num],False))
    #add submap idx to array
    
for x in range (0,7):
    
    for y in range (0, 20):
        
        #add submap to mapgroup
        map_s.add(maps.Map(maps.map_1[x][y], x * FULL_TILE, y * FULL_TILE))
        #add submap to mapgroup
##read the maps and draw
       
##collision detection
collide=False
##collision detection   

##constant for cone
#cone_x=CENTER[0]
#cone_y=CENTER[1]
#position_sensor=[0,0]
#index
i=0
#index
#counter of yellow cone
p=0
#counter of yellow cone
#counter of blue cone
q=0#blue
#counter of blue cone
#Projection of point on the canvas of lidar data from blue cone
draw_blue_cone=[]
#Projection of point on the canvas of lidar data from blue cone
#Projection of point on the canvas of lidar data from yellow cone
draw_yellow_cone=[]
#Projection of point on the canvas of lidar data from yellow cone
#absolute value of distance from yellow cone
dis_yellow=[]
#absolute value of distance from yellow cone
#absolute value of distance from blue cone
dis_blue=[]
#absolute value of distance from blue cone
#absolute value of distance from back cone
dis_back=0
#absolute value of distance from back cone
#input data form of RL algorithm of blue cone
vektor_blue=[]
#input data form of RL algorithm of blue cone
#input data form of RL algorithm of yellow cone
vektor_yellow=[]
#input data form of RL algorithm of yellow cone
#sum of square of blue cone
dis_blue_sqr_sum=0
#sum of square of blue cone
#sum of square of yellow cone
dis_yellow_sqr_sum=0
#sum of square of yellow cone
#positive deviation
diff_sum_yb=0
#positive deviation
##constant for cone

##initial draw element
draw_blue_cone.append([0,0])
draw_yellow_cone.append([0,0])
dis_blue.append(0)
dis_yellow.append(0)
vektor_blue.append([0,0])
vektor_yellow.append([0,0])
##initial draw element

##constant for path
#index
j=0
#index
#temporary path coordinate for mouse and auto path build
path_x=0
path_y=0
#temporary path coordinate for mouse and auto path build
ctrl_pressed=False
#lidar effective distance
dis_path=0
#lidar effective distance
#temporary last path milestone
last_point=[startpoint.x,startpoint.y]
#temporary last path milestone
#path milestone
draw_path=[]
#path milestone
draw_path.append([0,0])#init
#find 2 path point
path_close_1=[0,0]
path_close_2=[0,0]
dis_close_path_1=0
dis_close_path_2=0
dis_between_path=0
dis_close_path_temp=0
base_path_mileage=0
tag=0
tag_2=1
path_tag=[]
path_tag_2=[]
ta=0
ta_2=0
projection=[0,0]

swich_cal_projection=True
#find 2 path point

##constant for path

##constant for lidar
#number of sensored blue cone
k=0
#number of sensored blue cone
#number of sensored yellow cone
l=0
#number of sensored yellow cone
#range of lidar
bound_lidar=CENTER[0]*2/5
#range of lidar
##constant for lidar

##temporary state to form all the inputs
state=[[],0,0,0]
##temporary state to form all the inputs

##konstant of RL
#numer of completed process
ep_total=0
#numer of completed process
#switch of the summary
summary=False
#switch of the summary
#episode after max reward updated
ep_lr=0
#episode after max reward updated
#weight for the speed in reward
speed_faktor=1
#weight for the speed in reward
#speed_faktor_enhance=1
#angle_faktor_enhance=1
#weight for the distance in reward
distance_faktor=0
#weight for the distance in reward
#minimum distance before impact
safty_distance_impact=60
#minimum distance before impact
#minimum distance before turning
safty_distance_turning=65
#minimum distance before turning
#distance which means impact
collision_distance=40
#distance which means impact
#total moved distance
distance=0
#total moved distance
#total moved distance on middle line
distance_projection=0
distance_projection_old=0
speed_projection=0
v_distance_projection=0
#total moved distance on middle line
#distance every episode
distance_set=[]
#distance every episode
#reward of the current state and action
reward=1
#reward of the current state and action
#show reward for the last episode
reward_show=0
#show reward for the last episode
#temporary reward of the whole episode 
reward_sum=0
#temporary reward of the whole episode 
#average whole episode reward of the whole training  process
reward_mean=[]
#average whole episode reward of the whole training  process
#set of runing reward
rr=[]
#set of runing reward
#index of runing reward set
rr_idx=0
#index of runing reward set
#reward of the whole episode 
running_reward =0
#reward of the whole episode 
#max running reward until now
running_reward_max=0
#max running reward until now
#ratio of the max max running reward and the average running reward.1 is the goal
reward_mean_max_rate=[]
#ratio of the max max running reward and the average running reward.1 is the goal
#vt=0
#learning start
start_action=False
#learning start
#Rendering start
Render=True
#Rendering star
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
#occupyed memory
MEMORY_CAPACITY = 500
#occupyed memory
#size of memory slice
BATCH_SIZE = 2000
#size of memory slice
#minimal exploration wide of action
VAR_MIN = 0.1
#minimal exploration wide of action
#initial exploration wide of action
var1 = 0.99
var2 = 0.99
#initial exploration wide of action
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
#max reward reset
max_reward_reset=0
#max reward reset
#set of manual changed learning rate 
lr_set=[]
#set of manual changed learning rate 
#dimension of inputs for RL Agent
features_n=input_dim
#dimension of inputs for RL Agent
#inputs state of RL Agent
observation=np.zeros(input_dim)
#inputs state of RL Agent
#copy the state
observation_old=np.zeros(input_dim)
#copy the state
for t in range (0,input_dim):
    observation[t]=0
    observation_old[t]=0
##konstant of RL
input_max=np.zeros(input_dim)
input_min=np.zeros(input_dim)
#
path_man=[]
corner=[]
corner.append(32)
corner.append(36)
corner.append(41)
corner.append(43)
corner.append(44)
corner.append(45)
corner.append(46)
corner.append(47)
corner.append(48)
corner.append(49)
corner.append(54)
for t in range (1,corner[0]):
    path_man.append([CENTER[0]-49*t,CENTER[1]-3*t])

for t in range (1,5):
    path_man.append([path_man[corner[0]-2][0]-49.74*t,path_man[corner[0]-2][1]-5*t])
    
for t in range (1,6):
    path_man.append([path_man[corner[1]-2][0]-49*t,path_man[corner[1]-2][1]-10*t])

for t in range (1,3):
    path_man.append([path_man[corner[2]-2][0]-45.82*t,path_man[corner[2]-2][1]-20*t])

path_man.append([path_man[corner[3]-2][0]-40,path_man[corner[3]-2][1]-30])
path_man.append([path_man[corner[4]-2][0]-31,path_man[corner[4]-2][1]-39.23])
path_man.append([path_man[corner[5]-2][0]-19.6,path_man[corner[5]-2][1]-46])
path_man.append([path_man[corner[6]-2][0]+15,path_man[corner[6]-2][1]-47.7])
path_man.append([path_man[corner[7]-2][0]+30,path_man[corner[7]-2][1]-40])
path_man.append([path_man[corner[8]-2][0]+35,path_man[corner[8]-2][1]-35.71])
for t in range (1,6):
    path_man.append([path_man[corner[9]-2][0]+43.5*t,path_man[corner[9]-2][1]-24.65*t])
    
for t in range (1,23):
    path_man.append([path_man[corner[10]-2][0]+47.37*t,path_man[corner[10]-2][1]-16*t])
    

#for t in range (1,5):
    #path_man.append([path_man[8][0]-50*t,path_man[8][1]+10*math.sqrt(t)])

#path_man.append([path_man[30][0]-48.29,path_man[30][1]-12.94])
#path_man.append([path_man[31][0]-25*math.sqrt(3),path_man[31][1]-25])
    
# =============================================================================
# coneb_back=traffic_cone.cone(CENTER[0]+100,CENTER[1]-20,-1,car.x,car.y)
# coney_back=traffic_cone.cone(CENTER[0]+100,CENTER[1]+20,1,car.x,car.y)
# 
# cone_s.add(coneb_back)
# cone_s.add(coney_back)
# 
# list_cone_blue.append(coneb_back)
# list_cone_yellow.append(coney_back)
# 
# draw_yellow_cone.append([0,0])
# dis_yellow.append(0)
# vektor_yellow.append([0,0])
# p=p+1
# 
# draw_blue_cone.append([0,0])
# dis_blue.append(0)
# vektor_blue.append([0,0])
# q=q+1
# =============================================================================


    
for pa in path_man:
    path_x=pa[0]
    path_y=pa[1]
     
    path_new=path.path(path_x,path_y,car.x,car.y)
    list_path_point.append(path_new)  
    path_s.add(path_new)  
     
     
     
    line=[last_point,[path_new.x,path_new.y]]
     
    cone_x, cone_y = cal.calculate_t(line,1,half_path_wide,car.x,car.y)
    cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
    list_cone_yellow.append(cone_new)
    cone_s.add(cone_new)
     
    draw_yellow_cone.append([0,0])
    dis_yellow.append(0)
    vektor_yellow.append([0,0])
    p=p+1
    
    cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide,car.x,car.y)
    cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
    list_cone_blue.append(cone_new)
    cone_s.add(cone_new)
   
    draw_blue_cone.append([0,0])
    dis_blue.append(0)
    vektor_blue.append([0,0])
    q=q+1

    
    last_point=[path_x+car.x,path_y+car.y]
    draw_path.append([0,0])
    j=j+1
# =============================================================================
##


sess = tf.Session()

actor = Actor(sess, ACTION_DIM, ACTION_BOUND, LR_A, REPLACE_ITER_A)
critic = Critic(sess, input_dim, ACTION_DIM, LR_C, rd, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * input_dim + ACTION_DIM + 1)
saver = tf.train.Saver()

#LOAD = False
LOAD = True
MODE = ['online', 'cycle']
n_model = 0

di = './'+MODE[n_model]
###main loop process
if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(di))
else:
    sess.run(tf.global_variables_initializer())
        
while True:

    if summary==False:
        #show1=len(list_cone_yellow)
        ##key event continually
        
        
        keys = pygame.key.get_pressed()
        
    
        if keys[K_1]:
    
            angle=half_Max_angle
            
        if keys[K_2]:
      
            angle=30
            
        if keys[K_3]:
           
            angle=15    
            
        if keys[K_4]:
    
            angle=2.3
        
        if keys[K_6]:
            
            angle=0  
            
        if keys[K_5]:
            
            car.speed=10
            
        if keys[K_7]:
    
            angle=-2.3
            
        if keys[K_8]:
      
            angle=-15
            
        if keys[K_9]:
         
            angle=-30   
            
        if keys[K_0]:
          
            angle=-half_Max_angle
            
        if keys[K_LEFT]:
            
            turing_speed=1
            
            if angle<0:
                
                angle=-1
                
            if angle<half_Max_angle+1:
                
                angle=angle+turing_speed
    
            #if angle==0:
               
        if keys[K_RIGHT]:
            
            turing_speed=-1
            
            if angle>0:
                
                angle=1
            
            if angle>-half_Max_angle-1:
                
                angle=angle+turing_speed
       
            #if angle==0:
               
        if keys[K_UP]:
            
            car.accelerate(car.acceleration)
            
    
            
        if keys[K_DOWN]:
            
            car.deaccelerate()
            
        if keys[K_BACKSPACE]:
            
            car.speed=0
        
        if keys[K_LCTRL] : 
            
            ctrl_pressed=True
            
            if pygame.mouse.get_pressed()==(True,False,False):
                
                path_x, path_y = pygame.mouse.get_pos()
                dis_path=cal.calculate_r((path_x+car.x,path_y+car.y),last_point)
                
                if dis_path>50:
                    #pass
                    
                    path_new=path.path(path_x,path_y,car.x,car.y)
                    list_path_point.append(path_new)  
                    path_s.add(path_new)  
                    
                    line=[last_point,[path_new.x,path_new.y]]
                    
                    cone_x, cone_y = cal.calculate_t(line,1,half_path_wide,car.x,car.y)
                    cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
                    list_cone_yellow.append(cone_new)
                    cone_s.add(cone_new)
                    
                    draw_yellow_cone.append([0,0])
                    dis_yellow.append(0)
                    vektor_yellow.append([0,0])
                    p=p+1
                    
                    cone_x, cone_y = cal.calculate_t(line,-1,half_path_wide,car.x,car.y)
                    cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
                    list_cone_blue.append(cone_new)
                    cone_s.add(cone_new)
                   
                    draw_blue_cone.append([0,0])
                    dis_blue.append(0)
                    vektor_blue.append([0,0])
                    q=q+1
                    
                    last_point=[path_x+car.x,path_y+car.y]
                    draw_path.append([0,0])
                    j=j+1
        else:
            
            ctrl_pressed=False
                
        ##system event
        for event in pygame.event.get():
            
            # quit for windows
            if event.type == QUIT:
                        
                pygame.quit()
                
                sys.exit()
    
                            
            if event.type == KEYDOWN :
                
                # quit for esc key
                if event.key == K_ESCAPE:  
                                
                    pygame.quit()
                    
                    sys.exit()
                    
                #timer
                if event.unicode == ' ':  
                    
                    if start_timer==False: 
                        
                        start_timer=True
                        
                    else: 
                        
                        start_timer=False
                        
                if event.unicode == '\r':
                    
                    start_action=True
                    
                    
                if event.unicode == '\x08':
                    
                    car.reset()
                    car.set_start_direction(90)
                
                if event.unicode == 'r':
                    
                    if  Render==False: 
                        
                        Render=True
                        
                    else: 
                        
                        Render=False
                    
                if event.key == K_e:
                    
                    #ep_lr=0
                    max_reward_reset=max_reward_reset+1
                    
                if event.unicode == 'd':
                    
                    lr=lr/10
                    RL.learning_rate=lr
                    print("max lr:",lr)

                if event.unicode == 'm':

                    lr=lr*10
                    RL.learning_rate=lr
                    print("max lr:",lr)
                             
                if event.key == K_t:
                    pass

                    #if os.path.isdir(path): shutil.rmtree(path)
                    #os.mkdir(path)
                    #ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
                    #save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
                    #print("\nSave Model %s\n" % save_path)
                                        

                    
            if event.type == KEYUP :    
                
                if event.key == K_LEFT or event.key == K_RIGHT:
                    
                    angle=0

                    
            if event.type == MOUSEBUTTONDOWN and ctrl_pressed==False:
                
    
                    
                if pygame.mouse.get_pressed()==(True,False,False):
                    
                    cone_x, cone_y = pygame.mouse.get_pos()
                    cone_new=traffic_cone.cone(cone_x,cone_y,1,car.x,car.y)
                    list_cone_yellow.append(cone_new)
                    cone_s.add(cone_new)
                    
                    draw_yellow_cone.append([0,0])
                    dis_yellow.append(0)
                    vektor_yellow.append([0,0])
                    p=p+1
                    
                elif pygame.mouse.get_pressed()==(False,False,True) :
                    
                    cone_x, cone_y = pygame.mouse.get_pos()
                    cone_new=traffic_cone.cone(cone_x,cone_y,-1,car.x,car.y)
                    list_cone_blue.append(cone_new)
                    cone_s.add(cone_new)
                   
                    draw_blue_cone.append([0,0])
                    dis_blue.append(0)
                    vektor_blue.append([0,0])
                    q=q+1
                    
        ##system event
    # =============================================================================
    #     if keys[K_DELETE]:
    #         
    #         cone_s.empty()
    #         list_cone_blue=[]
    #         list_cone_yellow=[]
    # =============================================================================
        ##key event 

                    
            
# =============================================================================
#             if action[1]> angle+var:
# 
#                 angle=angle+car.steering
#                 
#             if action[1]< angle-var:
# 
#                 angle=angle-car.steering
# =============================================================================
            #print("angle:",angle)
            #print("action[1]:",action[1])

        ##reward
        
        distance=distance+car.speed
        #reward=speed_faktor*car.speed
            
        car.soften()
        #camera position reset
        cam.set_pos(car.x, car.y)
        
        
        #speed calculation for center of the car
#        speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))/(1/COUNT_FREQUENZ)#pixel/s
        speed=math.sqrt(pow(car.x-x_old,2)+pow(car.y-y_old,2))
#        print("speed",speed)
        #update the old position
        x_old=car.x
        y_old=car.y
        
        ##
        
            
        #postion_sensor=[(model[7][0][0]-CENTER[0]+car.x),(model[7][0][1]-CENTER[1]+car.y)]
        ##
           
        ##text setting
        
        text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (0, 0, 102))
        textpos_fps = text_fps.get_rect(centery=25, left=20)
        
        #print("FPS:",clock.get_fps())
        
        text_timer = font.render('Timer: ' + str(round(float(count/COUNT_FREQUENZ),2)) +'s', 1, (0, 0, 102))
        textpos_timer = text_timer.get_rect(centery=65, left=20)
         
        text_loop = font.render('Loop: ' + str(int(clock.get_time())) +'ms', 1, (0, 0, 102))
        textpos_loop = text_loop.get_rect(centery=105, left=20)
        
        text_pos= font.render('POS: ' + '( '+str(round(float(car.x-xc0),2))+' , '+str(round(float(car.y-yc0),2))+' )', 1, (0, 0, 102))   
        textpos_pos = text_pos.get_rect(centery=145, left=20)
        
        text_posr= font.render('POS_RAW: ' +  '( '+str(round(float(car.x),2))+' , '+str(round(float(car.y),2))+' )', 1, (0, 0, 102))   
        textpos_posr = text_posr.get_rect(centery=185, left=20)
        
        text_dir= font.render('Direction: ' + str(round(float(car.dir),2)), 1, (0, 0, 102))   
        textpos_dir = text_dir.get_rect(centery=225, left=20)
        
        text_speed= font.render('speed: ' + str(round(float(speed),2))+'pixel/s'+'|'+str(round(float(car.speed),2))+'pixel/loop', 1, (0, 0, 102))   
        textpos_speed = text_speed.get_rect(centery=265, left=20)
        
        text_colour= font.render('colour: ' + str(round(screen.get_at(((int(CENTER[0]-50), int(CENTER[1]-50)))).g,2)), 1, (0, 0, 102))   
        textpos_colour = text_colour.get_rect(centery=305, left=20)
        
#        text_posr= font.render('POS_RAW: ' +  '( '+str(round(float(car.x),2))+' , '+str(round(float(car.y),2))+' )', 1, (0, 0, 102))   
#        textpos_posr = text_posr.get_rect(centery=345, left=20)
        #text_dis_yellow= font.render('distance to yellow cone: ' + str(round(float(dis_yellow[0]),2)), 1, (0, 0, 102))   
        #textpos_dis_yellow = text_dis_yellow.get_rect(centery=345, left=20)
        
        #text_dis_blue= font.render('distance to blue cone: ' + str(round(float(dis_blue[0]),2)), 1, (0, 0, 102))   
        #textpos_dis_blue =text_dis_blue.get_rect(centery=385, left=20)
        
        
        
        
        ##text setting
        
        
        #model of car
        #35*41 rect
        #middle axis 40
        #axis 35
        #wheel 20
        
        #angle signal give to the object car
        car.wheelangle=angle
        model=cv.turning(model,angle,CENTER,half_middle_axis_length,half_horizontal_axis_length,radius_of_wheel,el_length)
    
        
        if angle>0 :
        
            car.rrl=model[19]
            car.steerleft(angle)
            #vektor_speed=[car.speed*,car.speed*]
           
        elif angle<0  :
    
            car.rrr=model[21]
            car.steerright(angle)
        
    
        model=cv.rotate(model,CENTER,car.dir)
        
        for i in range (0, p+1):
            
            dis_yellow[i]=cal.calculate_r((car.x,car.y),(list_cone_yellow[i].x-model[7][0][0],list_cone_yellow[i].y-model[7][0][1]))
            draw_yellow_cone[i]=[list_cone_yellow[i].x-cam.x,list_cone_yellow[i].y-cam.y]
        
        for i in range (0, q+1):
            
            dis_blue[i]=cal.calculate_r((list_cone_blue[i].x-model[7][0][0],list_cone_blue[i].y-model[7][0][1]),(car.x,car.y))
            draw_blue_cone[i]=[list_cone_blue[i].x-cam.x,list_cone_blue[i].y-cam.y]         
    
        for i in range (0, j+1):
            
            draw_path[i]=[list_path_point[i].x-cam.x,list_path_point[i].y-cam.y]
       
        dis_close_path_1=1000
        
        for i in range (0, j+1):
            
            dis_close_path_temp=cal.calculate_r((list_path_point[i].x-model[7][0][0],list_path_point[i].y-model[7][0][1]),(car.x,car.y))
            
            if dis_close_path_temp<dis_close_path_1:
                path_close_1=draw_path[i]
                dis_close_path_1=dis_close_path_temp
                tag=i
        
        dis_close_path_2=1000
       
        for i in range (0, j+1):
            
            dis_close_path_temp=cal.calculate_r((list_path_point[i].x-model[7][0][0],list_path_point[i].y-model[7][0][1]),(car.x,car.y))

            if dis_close_path_temp<dis_close_path_2 and dis_close_path_temp>dis_close_path_1:
                path_close_2=draw_path[i]
                dis_close_path_2=dis_close_path_temp
                tag_2=i
        #init             
        if path_tag==[]:
            
            path_close_1=draw_path[0]
            path_tag.append(tag)

        #init         
                
        if path_tag_2==[]:
            
            path_close_2=draw_path[1]
            path_tag_2.append(tag_2)
            
        dis_between_path=cal.calculate_r(path_close_1,path_close_2)
        
        if tag!=path_tag[ta]:
            
            path_tag.append(tag)
            ta=ta+1
            swich_cal_projection=False
#            print("ta",ta) 
        if tag_2!=path_tag_2[ta_2]:
            
            path_tag_2.append(tag_2)
            if tag_2>path_tag_2[ta_2]:
                swich_cal_projection=True 
                base_path_mileage=dis_between_path*ta
            ta_2=ta_2+1
#            print("ta_2",ta_2) 
            
        
        
#        print("dis_between_path",dis_between_path)
        projection=cal.calculate_projection(swich_cal_projection,dis_close_path_1,dis_close_path_2,dis_between_path)
        distance_projection=base_path_mileage+projection[0]
        v_distance_projection=projection[1]
        
        if distance_projection_old==0:
            
            distance_projection_old=distance_projection
            
        speed_projection=distance_projection-distance_projection_old
        distance_projection_old=distance_projection
#        print("distance_projection",distance_projection)
#        print("speed_projection",speed_projection)
#        print("projection",projection)  
#        print("path_close_1",path_close_1)  
#        print("path_close_2",path_close_2)
#        print("draw_path[1]",draw_path[1])  
#        print("draw_path[2]",draw_path[2])
#        print("draw_path[3]",draw_path[3])

#        sin_projection=cal.calculate_sin()
#        print("path_tag",path_tag)
#        print("tag length",len(path_tag))
#        print("path_tag2",path_tag_2)
#        print("tag2 length",len(path_tag_2))

        
#        print("dis_close_path_2",dis_close_path_2)  
#        print("dis_close_path_1",dis_close_path_1)
#        print("dis_between_path",dis_between_path)
        
        dis_back=cal.calculate_r((car.x,car.y),(coneback.x-model[7][0][0],coneback.y-model[7][0][1]))
        ##start drawing
#        car.speed=0.1
        if start_action==True:
            
            start_timer=True
            #print("observation:",observation)
            #action=RL. choose_action(observation)
            #print("action:",action)
            
            action = actor.choose_action(observation)
            #exploration
#            epsilon=np.random.random_sample()
#            
#            if epsilon< var1:
#                
#                action[0]=np.random.random_sample()*(ACTION_BOUND0[1]-ACTION_BOUND0[0])+ACTION_BOUND0[0]
#                #print("action[0]:",action[0])
#                action[1]=np.random.random_sample()*(ACTION_BOUND1[1]-ACTION_BOUND1[0])+ACTION_BOUND1[0]
#                
#            else:
#                action[0]=(action[0]+1)/2*(ACTION_BOUND0[1]-ACTION_BOUND0[0])+ACTION_BOUND0[0]
                #print("epsilon:",epsilon)
                #print("var1:",var1)
                #print("action[0]:",action[0])
                #print("action[1]:",action[1])
                #print("action[1]:",action[1])
            action[0] = np.clip(np.random.normal(action[0], var1), *ACTION_BOUND0)
            action[1] = np.clip(np.random.normal(action[1], var2), *ACTION_BOUND1)
            #exploration
            #print("action:",action)
            #print("car.speed:",car.speed)
            
            if car.speed<=1:
                action[0]=np.random.random_sample()*(ACTION_BOUND0[1]-0)
          
            #print("action:",action)
            
#            if angle<half_Max_angle and angle>-half_Max_angle and angle+action[1]<half_Max_angle and angle+action[1]>-half_Max_angle:
#                
#                angle=angle+action[1]
#                
#            elif angle<half_Max_angle and angle+action[1]>half_Max_angle:
#                
#                angle=half_Max_angle
#                action[1]=half_Max_angle-angle
#                
#            elif angle>-half_Max_angle and angle+action[1]<-half_Max_angle:
#                
#                angle=-half_Max_angle
#                action[1]=-half_Max_angle-angle
#                
            angle_old=angle
               
            if angle<half_Max_angle and angle>-half_Max_angle and angle+action[1]<half_Max_angle and angle+action[1]>-half_Max_angle:
                
                angle=angle+action[1]
            
            else:
                
                action[1]=0
                
            for i in range (0, q+1):
            
                if dis_blue[i]<safty_distance_turning:
                    
                    if dis_blue[i]<safty_distance_impact:
                        
                        angle=half_Max_angle
                        action[1]=half_Max_angle-angle_old
                        break
                    
                    elif angle>0:
                        
                        angle=0
                        action[1]=0-angle_old
                        break
                    
                    if angle<half_Max_angle and angle>-half_Max_angle:
                        
                        action[1]=5
                        angle=angle+action[1]
                        
                    break
                
            for i in range (0, p+1):
            
                if dis_yellow[i]<safty_distance_turning:
                    
                    if dis_yellow[i]<safty_distance_impact:
                        
                        angle=-half_Max_angle
                        action[1]=-half_Max_angle-angle_old
                        break
                    
                    elif angle<0:
                        
                        angle=0
                        action[1]=0-angle_old
                        break
                     
                    if angle<half_Max_angle and angle>-half_Max_angle:
                        
                        action[1]=-5
                        angle=angle+action[1]
                        
                    break
                
            car.accelerate(action[0])
            #print("car.speed",car.speed)
            #print("action[0]",action[0])
            actor.angle.append(action[1])
            actor.accelerate.append(action[0])
        
        ##draw background
        if Render==True:
            
            screen.blit(background, (0,0))
        ##
        
        
        ##update map
        map_s.update(cam.x, cam.y)
        if Render==True:
            map_s.draw(screen)
        ##update map
        
        ##draw cones
        cone_s.update(cam.x, cam.y)
        cone_h.update(cam.x, cam.y)
        if Render==True:
            cone_s.draw(screen)
            cone_h.draw(screen)
        ##draw cones
        
        ##draw path point
        path_s.update(cam.x,cam.y)
        if Render==True:
            
            path_s.draw(screen)
        
        
        ##draw path point
        
        ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
        car.rpl=(model[18][0][0]-CENTER[0]+car.x,model[18][0][1]-CENTER[1]+car.y)
        car.rpr=(model[20][0][0]-CENTER[0]+car.x,model[20][0][1]-CENTER[1]+car.y)
        ##determine the center of the car moving circle in the coordinate of the sprite layer,connection between surface canvas and sprite layer
      
        
    
        
        
        ##
        for i in range (0, q+1):
            
            vektor_blue[i]=cv.input_vektor_position(model,draw_blue_cone[i],CENTER,car.dir)
        
        for i in range (0, p+1):
            
            vektor_yellow[i]=cv.input_vektor_position(model,draw_yellow_cone[i],CENTER,car.dir)
    
        vektor_speed=[car.speed*math.cos(math.radians(270-angle)),car.speed*math.sin(math.radians(270-angle))]
#        print("vektor_speed",vektor_speed)
        ##
        
        
        ##
        text_yellow= font.render('vektor_yellow: ' + '( '+str(round(float(vektor_yellow[0][0]),2))+' , '+str(round(float(vektor_yellow[0][1]),2))+' )', 1, (0, 0, 102))   
        textpos_yellow = text_yellow.get_rect(centery=425, left=20)
        
        text_blue= font.render('vektor_blue: '+ '( ' + str(round(float(vektor_blue[0][0]),2))+' , '+str(round(float(vektor_blue[0][1]),2))+' )', 1, (0, 0, 102))   
        textpos_blue = text_blue.get_rect(centery=465, left=20)
        
        text_speed_v= font.render('vektor_speed: ' +  '( ' + str(round(float(vektor_speed[0]),2))+' , '+str(round(float(vektor_speed[1]),2))+' )', 1, (0, 0, 102))   
        textpos_speed_v = text_speed_v.get_rect(centery=505, left=20)
        ##
        
        
        #anything want to show
        text_show1= font.render('distance: ' + str(round(float(distance),2)), 1, (0, 0, 102))   
        textpos_show1 = text_dir.get_rect(centery=545, left=20)
        
        text_show2= font.render('reward: ' + str(round(float(reward_show),2)), 1, (0, 0, 102))   
        textpos_show2 = text_dir.get_rect(centery=585, left=20)
        #anything want to show
        
        
        
        ##
        player_s.update(cam.x, cam.y)
        if Render==True:
            
            player_s.draw(screen)
        ##
        
        
        ##
        tracks_s.add(tracks.Track(cam.x + CENTER[0] , cam.y + CENTER[1], car.dir))
        tracks_s.update(cam.x, cam.y)
        if Render==True:
            
            tracks_s.draw(screen)
        ##
        
        
        ##
        canvas.fill((255, 255, 255,0))
        ##
        
        ##draw lines to find center of the car
        pygame.draw.line(canvas, (0,100,0), (100,CENTER[1]), (CENTER[0],CENTER[1]),2)
        pygame.draw.line(canvas, (0,100,0), (CENTER[0],100), (CENTER[0],CENTER[1]),2)
        ##draw lines to find center of the car
        
        
        ##draw model of the car
        pygame.draw.line(canvas, (255,255,255), model[0][0], model[1][0],4)
        pygame.draw.line(canvas, (255,255,255), model[2][0], model[3][0],4)#front wheel
        pygame.draw.line(canvas, (255,255,255), model[4][0], model[5][0],4)#front axis
        pygame.draw.line(canvas, (255,255,255), model[6][0], model[7][0],4)#middle axis
        pygame.draw.line(canvas, (255,255,255), model[10][0], model[11][0],4)
        pygame.draw.line(canvas, (255,255,255), model[12][0], model[13][0],4)#back wheel
        pygame.draw.line(canvas, (255,255,255), model[8][0], model[9][0],4)#back axis
        ##draw model of the car
           
    
        ##draw back axis extension
        pygame.draw.line(canvas, (255,255,102), model[16][0],model[17][0],2)
        ##draw back axis extension
        
        ##draw distance to cones
        #print("model[7][0]:",model[7][0])
        for i in range (0, q+1):
            
            if dis_blue[i]<bound_lidar:
                
                pygame.draw.line(canvas, (0,255,255), model[7][0],draw_blue_cone[i],2)
            
        for i in range (0, p+1):
            
            if dis_yellow[i]<bound_lidar:
                
                pygame.draw.line(canvas, (255,255,0), model[7][0],draw_yellow_cone[i],2)
        ##draw distance to cones
        
        
        ##draw path
        for i in range (1, j+1):
            
            pygame.draw.line(canvas, (0,255,0), draw_path[i-1],draw_path[i],2)
            
        ##draw path

        pygame.draw.line(canvas, (220,20,60), model[7][0],path_close_1,3)
        pygame.draw.line(canvas, (220,20,60), model[7][0],path_close_2,3)  
        ##draw front axis extension
        if angle >= 2.3 or angle==0:
            
            pygame.draw.line(canvas, (255,255,102), model[5][0],model[14][0],2)#frontwheel turing axis
       
        if angle <= -2.3 or angle==0:
            
            pygame.draw.line(canvas, (255,255,102), model[4][0],model[15][0],2)#frontwheel turing axis
        
        ##draw front axis extension
        
        
        ##draw  the circle which car moving along in canvas
        if  angle>0:
            pass
            #pygame.draw.arc(canvas, (255,255,102), (model[18][0][0]-model[19],model[18][0][1]-model[19],2*model[19],2*model[19]), 0, 360, 0)
        
        if  angle<0:
            pass
            #pygame.draw.arc(canvas, (255,255,102), (model[20][0][0]-model[21],model[20][0][1]-model[21],2*model[21],2*model[21]), 0, 360, 0)
        
        ##draw  the car moving circle in canvas
        
        
        ##show canvas
        if Render==True:
            
            screen.blit(canvas, (0,0))
        ##show canvas
        
        
        ##show text
        if Render==True:
            screen.blit(text_fps, textpos_fps)
            screen.blit(text_timer, textpos_timer)
            screen.blit(text_loop, textpos_loop)
            screen.blit(text_pos, textpos_pos)
            screen.blit(text_posr, textpos_posr)
            screen.blit(text_dir, textpos_dir)
            screen.blit(text_speed, textpos_speed)
            screen.blit(text_colour, textpos_colour)
            #screen.blit(text_dis_yellow, textpos_dis_yellow)
            #screen.blit(text_dis_blue, textpos_dis_blue)
            screen.blit(text_yellow, textpos_yellow)
            screen.blit(text_blue, textpos_blue)
            screen.blit(text_speed_v, textpos_speed_v)
            screen.blit(text_show1, textpos_show1)
            screen.blit(text_show2, textpos_show2)
        ##show text
        
        
        ##start drawing
        
        #vektor_blue Within radar range
        vektor_blue_temp=[]
        #vektor_blue Within radar range
        #vektor_yellow Within radar range
        vektor_yellow_temp=[]
        #vektor_yellow Within radar range
        
        for i in range (0, q+1):
            
            if dis_blue[i]<bound_lidar:
                
                vektor_blue_temp.append(vektor_blue[i])
                dis_blue_sqr_sum=dis_blue_sqr_sum+pow(dis_blue[i],2)
                
        for i in range (0, p+1):
            
            if dis_yellow[i]<bound_lidar:
                
                vektor_yellow_temp.append(vektor_yellow[i])
                dis_yellow_sqr_sum=dis_yellow_sqr_sum+pow(dis_yellow[i],2)
                
                

        k=len(vektor_blue_temp)
        l=len(vektor_yellow_temp)
        if k>0 and l>0:
            diff_sum_yb=math.sqrt(pow((math.sqrt(dis_blue_sqr_sum/k)-math.sqrt(dis_yellow_sqr_sum/l)),2))
            #print("diff:",diff_sum_yb)
            state_sort=np.vstack((np.vstack(vektor_blue_temp),np.vstack(vektor_yellow_temp)))
            state_sort_temp=[]
            state_sort_end=[]
            
            for i in range (0, k+l):
              
               state_sort_temp.append([cal.calculate_sita_r(state_sort[i],[0,0]),state_sort[i]])
        
            state_sort=sorted(state_sort_temp)
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #print (state_sort)
            
            for i in range (0, k+l):
                
                state_sort_end.append(state_sort[i][1])
                
            #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            state[0]=state_sort_end    
            state[3]=np.vstack(state[0]).ravel()
            state[1]=np.vstack(vektor_speed).ravel()
            state[2]=angle
            state[1][0]=state[1][0]/car.maxspeed
            state[1][1]=state[1][1]/car.maxspeed
            state[2]=angle/half_Max_angle
            
            t=0
            for s in state[3]:
                state[3][t]=s/(CENTER[0]*2/5)
                t=t+1
                
#            print ("state[3]",state[3])
            #state_input=np.hstack((state[1]*speed_faktor_enhance,state[2]*angle_faktor_enhance,state[3]))                  
            state_input=np.hstack((state[1],state[2],state[3]))      
#            print (state_input)
            #print ('size:',state_input.size)
            for t in range(len(state_input)):
                observation[t]=state_input[t]
    
        #print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')      
        
        ##timer 
        if start_timer==True:
    
            count=count+1
        ##timer 
        #reward=distance_faktor*distance
        if start_action==True:
#            if math.sqrt(diff_sum_yb)>1:
#                
#                reward=car.speed*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/math.sqrt(diff_sum_yb))
#                
#            else:
#            
#                reward=car.speed*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/1)
#                
            if v_distance_projection>0.25:
                
                reward=speed_projection*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/math.sqrt(v_distance_projection))
                
            else:
            
                reward=speed_projection*speed_faktor+distance_faktor*distance+((2*car.maxspeed/car.acceleration)/0.5)
            #reward=car.speed*speed_faktor+distance_faktor*distance

#            if math.sqrt(diff_sum_yb)>1:
#                
#                reward=
#                
#            else:
#            
#                reward=
            #reward=car.speed*speed_faktor+distance_faktor*distance
            
            reward_sum=reward_sum+reward

            for i in range (0, q+1):
            
                if dis_blue[i]<collision_distance:
                    
                    collide=True
                    
            for i in range (0, p+1):
            
                if dis_yellow[i]<collision_distance:
                    
                    collide=True
                    
            if dis_back<collision_distance*1.5:
                
                collide=True
                
            if collide==True or count/COUNT_FREQUENZ>1.5: 
            #if pygame.sprite.spritecollide(car, cone_s, False) or count/COUNT_FREQUENZ>2:
                if collide==True :
                    
                    reward=-pow(car.speed,3)
                car.impact()
                car.reset()
                car.set_start_direction(90)

                #print("neg_reward:",reward)
                reward_sum=reward_sum+reward
                reward_show=reward_sum
                #print("reward:",reward)
                #print("episode:",episode)
                
                print("FPS:",clock.get_fps())
                reward_sum=0
                dis_blue_sqr_sum=0
                dis_yellow_sqr_sum=0
                distance_set.append(distance)
                distance=0
                angle=0
                count=0
                collide=False
                summary=True
    
            #RL.store_transition(observation, action, reward)
            M.store_transition(observation_old, action, reward, observation)
            #print("MEMORY_CAPACITY:",M.pointer)
            if M.pointer > MEMORY_CAPACITY:
                var1 = max([var1*0.9999, VAR_MIN])    # decay the action randomness
                var2 = max([var2*0.99999, VAR_MIN]) 
#                var1 = max([0.98*pow(1.00228,(-ep_total)), VAR_MIN])
#                var2 = max([0.98*pow(1.00228,(-ep_total)), VAR_MIN])
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :input_dim]
                b_a = b_M[:, input_dim: input_dim + ACTION_DIM]
                b_r = b_M[:, -input_dim - 1: -input_dim]
                b_s_ = b_M[:, -input_dim:]
                
                #print("b_M:",b_M)
                #print("b_s:",b_s)
                #print("b_a:",b_a)
                #print("b_r:",b_r)
                #print("b_s_:",b_s_)
#
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
                
            observation_old=observation
        #print(episode)

        ##clock tick
        clock.tick_busy_loop(COUNT_FREQUENZ)
        ##clock tick
        
        ##update screen
        if Render==True:
            pygame.display.update()
        ##update screen
        
    else:
        print("var1:",var1,"var2:",var2)
        print("MEMORY_CAPACITY:",M.pointer)
        if os.path.isdir(di): shutil.rmtree(di)
        os.mkdir(di)
        ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
        save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)
        #print("RL.r_set:",RL.r_set)
        #rs_sum = sum(RL.r_set)
        
        #if 'running_reward' not in globals():
            
            #running_reward = rs_sum

        #else:
            #running_reward = running_reward * 0.01 + rs_sum * 0.99
            
        running_reward = reward_show

        if running_reward_max<running_reward and ep_total>1:
            
            running_reward_max=running_reward
            ep_lr=0
            max_reward_reset=max_reward_reset+1
           
  

        print("running_reward:",running_reward)
        print("max_running_reward:",running_reward_max)
        
        rr.append(running_reward)   
        rr_idx=rr_idx+1
        reward_mean.append(sum(rr)/rr_idx)
        
        print("reward_mean:",reward_mean[rr_idx-1])
        if rr_idx>5:
            
            reward_mean_max_rate.append(running_reward_max/reward_mean[rr_idx-1])
        #if RL.learning_rate>0.001:

        print("max_reward_reset:",max_reward_reset)
        #RL.learning_rate=0.3/(ep_lr+3000)
        #lr_set.append(RL.learning_rate)
        #print("learning rate:",RL.learning_rate)
        #print("max lr:",lr)
        #print("deterministic:",RL.deterministic)
        #print("deterministic_count:",deterministic_count)
        #print("rr:",rr)
        #vt=RL.learn(car.maxspeed,car.acceleration)
        #print("RL.learn:",vt)

        plt.subplot(321)
        plt.plot(rr)  
        plt.xlabel('episode steps')
        plt.ylabel('runing reward')

        #plt.subplot(432)
        #plt.plot(vt)    # plot the episode vt
        #plt.xlabel('episode steps')
        #plt.ylabel('normalized state-action value')
        
     
        plt.subplot(323)
        plt.plot(reward_mean)  
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

        
        #print(actor.policy_grads[0])
        ep_total=ep_total+1
        print("totaol train:",ep_total)
        print("LOAD:",LOAD)
        ep_lr=ep_lr+1
        print("lr ep :",ep_lr)
        summary=False

###main loop process