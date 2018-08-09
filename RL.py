# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:02:08 2018

@author: Asgard
"""

import numpy as np
import tensorflow as tf

#np.random.seed(1)
#tf.set_random_seed(1)

#hidden layer
H=450
#hidden layer

#learning rate
#lr = 1e-4
#learning rate

#reward dacay
#rd= 0.99
#reward dacay

#input units

input_units=100
#input units


resume = False 

class PolicyGradient:
    def __init__(self,n_a,n_f,LR,RD,OG=False):
        self.n_actions=n_a
        self.n_features=n_f
        self.learning_rate=LR
        self.reward_decay=RD
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
                #kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
                kernel_initializer=tf.random_uniform_initializer(-0.23,0.23),
                bias_initializer=tf.constant_initializer(0),
                name='inputs',             
                
                )
        all_act=tf.layers.dense(
                inputs=layer_1,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),         
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
            
            #self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
            self.train =tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
    
    def choose_action(self, observation):
        
        prob=self.sess.run(self.all_act_prob,feed_dict={self.tf_obs:observation[np.newaxis, :]})
        action = np.random.choice(range(prob.shape[1]),p=prob.ravel())
        print("prob:",prob)
        return action
    def store_transition(self, s, a, r):
        
        self.ob_set.append(s)
        self.a_set.append(a)
        self.r_set.append(r)
        #print("action:",self.a_set)
        #print("reward:",self.r_set)
    def learn(self):

        discounted_r_set_norm=  self._discount_norm_rewards()
        
        self.sess.run(self.train,feed_dict={self.tf_obs:np.vstack(self.ob_set),self.tf_acts:np.array(self.a_set),self.tf_vt:discounted_r_set_norm,})

        self.ob_set,self.a_set,self.r_set=[],[],[]  
        
        return discounted_r_set_norm
    
    def _discount_norm_rewards(self):
        
        discounted_rs=np.zeros_like(self.r_set)
        running_add = 0
        
        for t in reversed(range(0,len(self.r_set))):
            
            running_add = running_add*self.reward_decay+ self.r_set[t]
            discounted_rs[t]=running_add
        
        discounted_rs=discounted_rs-np.mean(discounted_rs)
        discounted_rs=discounted_rs/np.std(discounted_rs)
            
        
        
        return discounted_rs
        
        
        
        
        
        
        
        