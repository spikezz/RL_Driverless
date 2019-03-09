#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:06:58 2019

@author: spikezz
"""
import tensorflow as tf
import numpy as np
import os
import shutil

np.random.seed(1)
tf.set_random_seed(1)

H1=300
H2=300
input_dim = 140
TAU=0.01

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='s_')




class Actor(object):
    def __init__(self, sess, n_a, action_bound, LR):
        self.sess = sess
        self.a_dim = n_a
        self.action_bound = action_bound
        self.lr = LR
        self.t_replace_counter = 0
        self.Momentum=0.9
        
        self.angle=[]
        self.accelerate=[]
        
        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.t_params, self.e_params)]
        
    def _build_net(self, s, scope, trainable):
        
        with tf.variable_scope(scope):
            
            #init_w = tf.contrib.layers.xavier_initializer()
            init_w=tf.random_uniform_initializer(-0.23,0.23)
#            init_b = tf.constant_initializer(0.000)
            init_b=tf.random_uniform_initializer(-1,1)
            layer1 = tf.layers.dense(s, H1, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l1',trainable=trainable)
            layer2 = tf.layers.dense(layer1, H1, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l2',trainable=trainable)
            #layer3 = tf.layers.dense(layer2, H2, activation=tf.nn.relu6,kernel_initializer=init_w, bias_initializer=init_b, name='l3',trainable=trainable)
            with tf.variable_scope('a'):
                
                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op, feed_dict={S: s})
        
    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        a=self.sess.run(self.a, feed_dict={S: s})[0] 
#        print("action:",a)
        return a  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
#            opt = tf.train.MomentumOptimizer(-self.lr,self.Momentum)# (- learning rate) for ascent policy
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, a, a_):
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_counter = 0
        self.Momentum=0.9

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
#            self.train_op = tf.train.MomentumOptimizer(self.lr,self.Momentum).minimize(self.loss)
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
        
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.t_params, self.e_params)]
        
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            #init_w = tf.contrib.layers.xavier_initializer()
            init_w=tf.random_uniform_initializer(-0.23,0.23)
#            init_b = tf.constant_initializer(0.00)
            init_b=tf.random_uniform_initializer(-1,1)
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                
            layer1 = tf.layers.dense(net, H1, activation=tf.nn.relu,
                    kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                    trainable=trainable)
            layer2 = tf.layers.dense(layer1, H2, activation=tf.nn.relu,
                    kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                    trainable=trainable)
            
            layer3 = tf.layers.dense(layer2, H1, activation=tf.nn.relu,
                    kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                    trainable=trainable)
            
            with tf.variable_scope('q'):
                q = tf.layers.dense(layer3,1,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        
        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        
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
    
    def read(self,idx,punish_batch_size):
        
        idxs = np.zeros(punish_batch_size) 
        i=0
        for t in range(idx-punish_batch_size,idx):
            
            idxs[i]=t
            i=i+1
        idxs=idxs.astype(int)
        return self.data[idxs, :]
    
    def write(self,idx,punish_batch_size,punished_reward):
        
        idxs = np.zeros(punish_batch_size) 
        i=0
        for t in range(idx-punish_batch_size,idx):
            idxs[i]=t
            i=i+1
        idxs=idxs.astype(int)
        
        self.data[idxs, -input_dim - 1]=punished_reward
    
class Saver(object):
    
    def __init__(self,sess,load,actor,critic,all_var):
        
        List_net=[]
        List_at=[v for v in actor.t_params]
        List_ae=[v for v in actor.e_params]
        List_ct=[v for v in critic.t_params]
        List_ce=[v for v in critic.e_params]
        List_net.extend(List_at)
        List_net.extend(List_ae)
        List_net.extend(List_ct)
        List_net.extend(List_ce)
        
        if all_var==True:
            
            self.saver=tf.train.Saver(max_to_keep=10000)
            
        else:
            
            self.saver=tf.train.Saver(var_list=List_net,max_to_keep=10000)
            
        self.LOAD = load
        #LOAD = True
        self.MODE = ['0']
        self.n_model = 0
        self.di = './Model/Model_'+self.MODE[self.n_model]
        di_load = './Model/Model_0'
        if self.LOAD:
            
            sess.run(tf.global_variables_initializer())
    
            self.saver.restore(sess, tf.train.latest_checkpoint(di_load))
        else:
            
            if os.path.isdir(di_load): shutil.rmtree(di_load)
            sess.run(tf.global_variables_initializer())
            os.mkdir(di_load)
    
    def save(self,sess,running_reward):
        
        self.n_model+=1
        self.MODE.append(str(self.n_model))
        
        if os.path.isdir(self.di): shutil.rmtree(self.di)
        os.mkdir(self.di)
        ckpt_path = os.path.join( './Model/Model_'+self.MODE[self.n_model], 'DDPG.ckpt')
        save_path = self.saver.save(sess, ckpt_path, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)

        file = os.path.join( './Model/Model_'+self.MODE[self.n_model], 'episode_reward.txt')
        fw=open(file, mode='w')
        reward_str= str(running_reward)
        fw.seek(0,0)
        fw.write( reward_str)