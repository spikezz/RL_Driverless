# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:02:08 2018

@author: Asgard
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

#hidden layer
H=300
#hidden layer



resume = False 

class PolicyGradient:
    def __init__(self,n_a,n_f,LR,RD,OG=False):
        self.n_actions=n_a
        self.n_features=n_f
        self.learning_rate=LR
        self.reward_decay=RD
        
        self.prob0=[]
        self.prob1=[]
        self.prob2=[]
        self.prob3=[]
        self.prob4=[]
        self.prob5=[]
        self.prob6=[]
        
        self.deterministic=False
        
        self.ob_set,self.a_set,self.r_set=[],[],[]
        self.loss=tf.Variable(0,dtype=tf.float32)
        self._build_net()
        
        self.sess=tf.Session()
        
        if OG==True:
            
            tf.summary.FileWrite("net/",self.sess.graph)
            
        self.sess.run(tf.global_variables_initializer())
    
    def _build_net(self):
        
        with tf.name_scope('inputs'):
            self.tf_obs=tf.placeholder(tf.float32,[None,self.n_features],name="observations")
            self.tf_acts=tf.placeholder(tf.int32,[None, ],name="actions")
            self.tf_vt=tf.placeholder(tf.float32,[None, ],name="action_values")
        
        layer_1=tf.layers.dense(
                
                inputs=self.tf_obs,
                units=H,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
                #kernel_initializer=tf.random_uniform_initializer(-0.23,0.23),
                bias_initializer=tf.constant_initializer(0),
                name='h_layer1',             
                
                )
        layer_2=tf.layers.dense(
                
                inputs=layer_1,
                units=H,
                activation=tf.nn.tanh,
                #kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
                kernel_initializer=tf.random_uniform_initializer(-0.23,0.23),
                bias_initializer=tf.constant_initializer(0),
                name='h_layer2',             
                
                )
        
        all_act=tf.layers.dense(
                inputs=layer_2,
                units=self.n_actions,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(-0.23,0.23),
                #kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),         
                #kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0),
                name='output'
        )

        self.all_act_prob =tf.nn.softmax(all_act, name='act_prob') 
        
        loss=tf.log(self.all_act_prob)
        
        with tf.name_scope('loss'):
            
            neg_log_prob=tf. reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts,self.n_actions),axis=1)
            loss=tf.reduce_mean(neg_log_prob*self.tf_vt)

        
        with tf.name_scope('optimizer'):
            
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            
            #self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            #self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
            #self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            #self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            #self.train = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(loss)
            
    def choose_action(self, observation):
        
        prob=self.sess.run(self.all_act_prob,feed_dict={self.tf_obs:observation[np.newaxis, :]})
        
        if self.deterministic==False:
            
            action = np.random.choice(range(prob.shape[1]),p=prob.ravel())
            
        else:
            
            prob_array= np.array(prob)
            prob_array=prob_array.flatten() 
            action=prob_array.argmax(axis=0)
            
        self.prob0.append(prob[0][0])
        self.prob1.append(prob[0][1])
        self.prob2.append(prob[0][2])
        self.prob3.append(prob[0][3])
        self.prob4.append(prob[0][4])
        self.prob5.append(prob[0][5])
        self.prob6.append(prob[0][5])
        #print("prob:",prob)
        #print("prob0:",self.prob0)
        return action
    def store_transition(self, s, a, r):
        
        self.ob_set.append(s)
        self.a_set.append(a)
        self.r_set.append(r)
        #print("action:",self.a_set)
        
    def learn(self,max_speed,acceleration):

        discounted_r_set_norm= self._discount_norm_rewards(max_speed,acceleration)
        #discounted_r_set_norm=self.r_set
        self.sess.run(self.train,feed_dict={self.tf_obs:np.vstack(self.ob_set),self.tf_acts:np.array(self.a_set),self.tf_vt:discounted_r_set_norm,})

        self.ob_set,self.a_set,self.r_set=[],[],[]  
        
        return discounted_r_set_norm
    
    def _discount_norm_rewards(self,max_speed,acceleration):
        
        discounted_rs=np.zeros_like(self.r_set)
        running_add = 0
        
        for t in reversed(range(0,len(self.r_set))):
            #print("t:",t)
            if t>(max_speed/acceleration):
                running_add = running_add*self.reward_decay+ self.r_set[t]
                discounted_rs[t]=running_add
                
            else:
                
                discounted_rs[t]=running_add
                
        SD=np.std(discounted_rs)
        MN=np.mean(discounted_rs)
        discounted_rs=(discounted_rs-MN)/SD
        
            
        
        
        return discounted_rs
        
        
        
        
        
        
        
        