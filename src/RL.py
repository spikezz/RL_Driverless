#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:06:58 2019

@author: spikezz
"""
import tensorflow as tf
import numpy as np


np.random.seed(2)
tf.set_random_seed(2)

H1=140
H2=140
input_dim = 60
TAU=0.01

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='s_')




class Actor(object):
    def __init__(self, sess, n_a, action_bound, LR,t_replace_iter):
        self.sess = sess
        self.a_dim = n_a
        self.action_bound = action_bound
        self.lr = LR
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter =0
        self.Momentum=0.9
        
        self.angle=[]
        self.accelerate=[]
        self.brake=[]
        self.writer = tf.summary.FileWriter("/home/spikezz/Driverless/aktuelle zustand/RL_Driverless/src/logs")
        
        
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
            init_w=tf.random_uniform_initializer(-0.5,0.5)
            init_b=tf.random_uniform_initializer(-1,1)
            
            layer1 = tf.layers.dense(s, H1, activation=tf.nn.softplus,kernel_initializer=init_w, bias_initializer=init_b, name='l1',trainable=trainable)
            layer2 = tf.layers.dense(layer1, H1, activation=tf.nn.softplus,kernel_initializer=init_w, bias_initializer=init_b, name='l2',trainable=trainable)
            
            with tf.variable_scope('a'):
                
                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
#                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.softsign, kernel_initializer=init_w,
#                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op, feed_dict={S: s})
        
        if self.t_replace_counter == self.t_replace_iter:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
            self.t_replace_counter = 0
        self.t_replace_counter += 1
        
    def choose_action(self, s):
        
        s = s[np.newaxis, :]    # single state
        a=self.sess.run(self.a, feed_dict={S: s})[0] 
#        print("action:",a)
        return a  # single action

    def add_grad_to_graph(self, a_grads):
        
        with tf.variable_scope('policy_grads'):
            
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            
            opt = tf.train.MomentumOptimizer(-self.lr,self.Momentum)# (- learning rate) for ascent policy
#            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter,a, a_):
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.loss_step=0
#        self.one_ep_step=0
        self.Momentum=0.9
        self.rank_q_max=0
        self.rank_TD_max=0
        self.rank_q_min=0
        self.rank_TD_min=0
        self.model_localization=[]
        self.writer = tf.summary.FileWriter("/home/spikezz/Driverless/aktuelle zustand/RL_Driverless/src/logs")
        
        
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q,self.q_rank = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)[0]   # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
            self.loss_scalar=tf.summary.scalar('loss_', self.loss)
            
        with tf.variable_scope('C_train'):
#            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.train.MomentumOptimizer(self.lr,self.Momentum).minimize(self.loss)
            
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
        
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.t_params, self.e_params)]
        
    def _build_net(self, s, a, scope, trainable):
        
        with tf.variable_scope(scope):
            #init_w = tf.contrib.layers.xavier_initializer()
            init_w=tf.random_uniform_initializer(-0.5,0.5)
            init_b=tf.random_uniform_initializer(-1,1)
            
            with tf.variable_scope('l1'):
                n_l1 = H1
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
#                q = tf.layers.dense(layer3,1,activation=tf.nn.tanh,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
                q_mean,q_var=tf.nn.moments(q,0)
                self.q_scalar=tf.summary.scalar('q_scalar_', tf.reshape(q_mean,[]))
                q_tg=tf.nn.tanh(q*0.001)
                
            with tf.variable_scope('q_scale'):
                q_scale=tf.matmul(q_tg, [[100.]]) 
        
        return q_scale,q

    def learn(self, s, a, r, s_,ep_total):       

        self.sess.run(self.soft_replace)
        _, self.loss_summary, self.q_summary=self.sess.run([self.train_op,self.loss_scalar,self.q_scalar], feed_dict={S: s, self.a: a, R: r, S_: s_})
#        print(self.sess.run(self.q_histogramm, feed_dict={S: s, self.a: a, R: r, S_: s_})[1])
        self.writer.add_summary(self.loss_summary,self.loss_step)
        self.writer.add_summary(self.q_summary,self.loss_step)
        if self.t_replace_counter == self.t_replace_iter:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
            self.t_replace_counter = 0
        self.t_replace_counter += 1
        self.loss_step+=1
#        self.one_ep_step+=1
        self.model_localization.append(ep_total)
        
    def rank_priority(self,s, a, r, s_):
        if self.loss_step==1000:
            self.rank_TD_min=10000000
        rank_q,rank_TD=self.sess.run([self.q_rank,self.loss], feed_dict={S: s, self.a: a, R: r, S_: s_})
        rank_q=np.reshape(rank_q,())
        if rank_q>self.rank_q_max:
            self.rank_q_max=rank_q
        if rank_TD>self.rank_TD_max:
            self.rank_TD_max=rank_TD
        if rank_q<self.rank_q_min:
            self.rank_q_min=rank_q
        if rank_TD<self.rank_TD_min:
            self.rank_TD_min=rank_TD
        return rank_q,rank_TD
        if self.loss_step==1000:
            print("rank_TD_max:",self.rank_TD_max,"rank_TD_min:",self.rank_TD_min)
    def get_rank_probability(self,rank_q,rank_TD):
        
        rank_q_correction=rank_q+(0-(self.rank_q_min+(self.rank_q_max-self.rank_q_min)/2))
        beta=3/(self.rank_q_max/2)#99%,1%
        probability_q=np.exp(beta*rank_q_correction)/(1+np.exp(beta*rank_q_correction))
        
        rank_TD_correction=rank_TD+(0-(self.rank_TD_min+(self.rank_TD_max-self.rank_TD_min)/2))
        beta=3/(self.rank_TD_max/2)#99%,1%
        probability_TD=np.exp(beta*rank_TD_correction)/(1+np.exp(beta*rank_TD_correction))
        
        return probability_q,probability_TD
        
class Memory(object):
    
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_,rank_q,rank_TD):
        transition = np.hstack((s, a, [r], s_,rank_q,rank_TD))
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
    