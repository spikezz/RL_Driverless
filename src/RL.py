#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:06:58 2019

@author: spikezz
"""
import tensorflow as tf
import numpy as np

from keras import regularizers


np.random.seed(1)
tf.set_random_seed(1)

H1=140
H2=140
H3=140
input_dim = 60
TAU=0.01

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, input_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, input_dim], name='s_')




class Actor(object):
    def __init__(self, sess, n_a,n_s, action_bound, LR,t_replace_iter):
        self.sess = sess
        self.s_dim = n_s
        self.a_dim = n_a
        self.action_bound = action_bound
        self.lr = LR
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter =0
        self.Momentum=0.9
        
        self.angle=[]
        self.accelerate=[]
        self.brake=[]
        self.summary_set=[]
        
        with tf.variable_scope('std'):
            
            init_std_1=tf.constant_initializer(0.002)
            init_std_2=tf.constant_initializer(0.0015)
            self.l1_n_std=tf.get_variable('w1_s_n_std',[],initializer=init_std_1, trainable=False)
            self.l2_n_std=tf.get_variable('w2_s_n_std',[],initializer=init_std_2, trainable=False)
            
        self.std_decay=tf.constant(0.99999)
        
        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)[1]

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)[1]
            

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.t_params, self.e_params)]
        
    def _build_net(self, s, scope, trainable):
        
        with tf.variable_scope(scope):
            
            init_w=tf.random_uniform_initializer(-0.1,0.1)
            init_b=tf.random_uniform_initializer(-0.15,0.15)
#            init_w = tf.contrib.layers.xavier_initializer()
#            init_b = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope('l1'):
                n_l1 = H1
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                layer1 = tf.nn.softplus(tf.matmul(s, w1_s) + b1)
                
            with tf.variable_scope('l1_n'):
                
                w1_s_noise=tf.random_normal(w1_s.shape,stddev=self.l1_n_std,name='w1_s_n')
#                w1_s_noise=tf.random_uniform(w1_s.shape,stddev=self.l1_n_std,name='w1_s_n')
                w1_s_n=w1_s+w1_s_noise
                b1_noise=tf.random_normal(b1.shape,stddev=self.l1_n_std,name='b1_n')
                b1_n=b1+b1_noise
                layer1_n=tf.nn.softplus(tf.matmul(s, w1_s_n) + b1_n)
                
#                self.l1_w_n_hist=tf.summary.histogram('l1_a_w_n_',w1_s_n)
#                self.summary_set.append(self.l1_w_n_hist)
#                self.l1_b_n_hist=tf.summary.histogram('l1_a_b_n_',b1_n)
#                self.summary_set.append(self.l1_b_n_hist)
#            layer1 = tf.layers.dense(s, H1, activation=tf.nn.softplus,kernel_initializer=init_w, bias_initializer=init_b, name='l1',trainable=trainable)
                
                self.l1_w_hist=tf.summary.histogram('l1_a_w_',w1_s)
                self.summary_set.append(self.l1_w_hist)
                self.l1_b_hist=tf.summary.histogram('l1_a_b_',b1)
                self.summary_set.append(self.l1_b_hist)
                
            with tf.variable_scope('l1_normalization'):
                
                layer1_n_n=tf.layers.batch_normalization(layer1_n,name='layer1_n_n_')
                self.l1_hist=tf.summary.histogram('l1_a_n_n_',layer1_n_n)
                self.summary_set.append(self.l1_hist)
#                self.p_layer1_n=tf.Print(layer1_n,[layer1_n],message="no normal",summarize=30)
#                self.p_layer1_n_n=tf.Print(layer1_n_n,[layer1_n_n],message="with normal",summarize=30)
                

                
            with tf.variable_scope('l2'):
                
                n_l2 = H2
                w2_s = tf.get_variable('w1_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b1', [1, n_l2], initializer=init_b, trainable=trainable)
                layer2 = tf.nn.softplus(tf.matmul(layer1, w2_s) + b2)
                
            with tf.variable_scope('l2_n'):
                w2_s_noise=tf.random_normal(w2_s.shape,stddev=self.l2_n_std,name='w2_s_n')
                w2_s_n=w2_s+w2_s_noise
                b2_noise=tf.random_normal(b2.shape,stddev=self.l2_n_std,name='b2_n')
                b2_n=b2+b2_noise
                layer2_n=tf.nn.softplus(tf.matmul(layer1_n_n, w2_s_n) + b2_n)
                
#                self.l2_w_n_hist=tf.summary.histogram('l2_a_w_n_',w2_s_n)
#                self.summary_set.append(self.l2_w_n_hist)
#                self.l2_b_n_hist=tf.summary.histogram('l2_a_b_n_',b2_n)
#                self.summary_set.append(self.l2_b_n_hist)
                
#                self.l2_w_hist=tf.summary.histogram('l2_a_w_',w2_s)
#                self.summary_set.append(self.l2_w_hist)
#                self.l2_b_hist=tf.summary.histogram('l2_a_b_',b2)
#                self.summary_set.append(self.l2_b_hist)
                
            with tf.variable_scope('l2_normalization'):
                
                layer2_n_n=tf.layers.batch_normalization(layer2_n,name='layer2_n_n_')
                self.l2_hist=tf.summary.histogram('l2_a_n_n_',layer2_n_n)
                self.summary_set.append(self.l2_hist)
#            layer2 = tf.layers.dense(layer1, H2, activation=tf.nn.softplus,kernel_initializer=init_w, bias_initializer=init_b, name='l2',trainable=trainable)
            with tf.variable_scope('a'):
                
                n_lo = self.a_dim
                w3_s = tf.get_variable('w1_s', [n_l2, n_lo], initializer=init_w, trainable=trainable)
                actions= tf.nn.tanh(tf.matmul(layer2, w3_s))
                
                self.actions_hist=tf.summary.histogram('actions_',actions)
                self.summary_set.append(self.actions_hist)
#                self.actions_w_hist=tf.summary.histogram('actions_w_',w3_s)
#                self.summary_set.append(self.actions_w_hist)
                
            with tf.variable_scope('a_n'):
                
                actions_n= tf.nn.tanh(tf.matmul(layer2_n_n, w3_s))

                
                self.actions_n_hist=tf.summary.histogram('actions_n',actions_n)
                self.summary_set.append(self.actions_n_hist)
                
#            with tf.variable_scope('a'):
                
#                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,name='a', trainable=trainable)
#                actions = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.softsign, kernel_initializer=init_w,
#                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
                scaled_a_n = tf.multiply(actions_n, self.action_bound, name='scaled_a_n')
                
        return scaled_a,scaled_a_n
    
    def merge_summary_end(self,s,time_step,critic):
        
        try:
            self.merge_summary = tf.summary.merge(self.summary_set)
            self.summary_actor=self.sess.run(self.merge_summary,feed_dict={S: s,S_: s})
            critic.writer.add_summary(self.summary_actor,time_step)
        except:
            print("nothing is here")
        
    def learn(self, s):   # batch update
        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op, feed_dict={S: s})
#        if self.t_replace_counter == self.t_replace_iter:
#            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#            self.t_replace_counter = 0
#        self.t_replace_counter += 1
        
    def choose_action(self, s,critic,ep_total):
        
        s = s[np.newaxis, :]    # single state
#        _,a=self.sess.run([self.p_layer1_n_n,self.a],feed_dict={S: s,S_: s})
        a=self.sess.run(self.a,feed_dict={S: s})
        self.l1_n_std=tf.multiply(self.l1_n_std, self.std_decay)
        self.l2_n_std=tf.multiply(self.l2_n_std, self.std_decay)
        return a[0]  # single action

    def add_grad_to_graph(self, a_grads):
        
        with tf.variable_scope('policy_grads'):
            
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=-a_grads)
            self.actions_policy_grads_hist=tf.summary.histogram('actions_policy_grads_',self.policy_grads[0])
            self.summary_set.append(self.actions_policy_grads_hist)
            
        with tf.variable_scope('A_train'):
            
            opt = tf.train.MomentumOptimizer(self.lr,self.Momentum)# (- learning rate) for ascent policy
#            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter,a, a_,C_TD):
        
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
        self.rank_reward_max=1
        self.rank_TD_max=1
        self.rank_reward_min=1
        self.rank_TD_min=1
        self.c_TD=C_TD
        self.TD_set = np.zeros(self.c_TD)
        self.TD_set_no_zero=np.array([])
        self.TD_step=0
        self.model_localization=[]
        self.summary_set=[]
#        self.summary_step=0
        self.writer = tf.summary.FileWriter("/home/spikezz/Driverless/aktuelle zustand/RL_Driverless/src/logs")
        
        
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q= self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)[0]   # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
            self.loss_scalar=tf.summary.scalar('loss_', self.loss)
            self.summary_set.append(self.loss_scalar)
            
        with tf.variable_scope('C_train'):
#            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
#            self.train_op = tf.train.MomentumOptimizer(self.lr,self.Momentum).minimize(self.loss)
            
            self.opt = tf.train.MomentumOptimizer(-self.lr,self.Momentum)
##            self.q_grads = self.opt.compute_gradients(self.loss)
##            self.summary_set.extend([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.q_grads])
            
            self.q_grads = tf.gradients(ys=self.loss, xs=self.e_params, grad_ys=None)
            self.q_grads_hist=tf.summary.histogram('q_grads_',self.q_grads[0])
            self.summary_set.append(self.q_grads_hist)
#            self.summary_set.extend([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.q_grads])
            
            self.train_op=self.opt.apply_gradients(zip(self.q_grads,self.e_params))
            
        with tf.variable_scope('a_grad'):
            
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
            self.a_grads_hist=tf.summary.histogram('a_grads_',self.a_grads)
            self.summary_set.append(self.a_grads_hist)
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.t_params, self.e_params)]
        
    def _build_net(self, s, a, scope, trainable):
        
        with tf.variable_scope(scope):
            
            init_w=tf.random_uniform_initializer(-0.1,0.1)
            init_b=tf.random_uniform_initializer(-0.15,0.15)
#            init_w = tf.contrib.layers.xavier_initializer()
#            init_b = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('l1'):
                
                n_l1 = H1
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                layer1 = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                
                self.l1_ws_hist=tf.summary.histogram('l1_c_ws_',w1_s)
                self.summary_set.append(self.l1_ws_hist)
                self.l1_wa_hist=tf.summary.histogram('l1_c_wa_',w1_a)
                self.summary_set.append(self.l1_wa_hist)
                self.l1_b_hist=tf.summary.histogram('l1_c_b_',b1)
                self.summary_set.append(self.l1_b_hist)
                
            with tf.variable_scope('l1_normalization'):
                
                layer1_n=tf.layers.batch_normalization(layer1,name='layer1_n_')
                self.l1_hist=tf.summary.histogram('l1_c_n_',layer1_n)
                self.summary_set.append(self.l1_hist)
                
            with tf.variable_scope('l2'):
                n_l2 = H2
                w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
                layer2 = tf.nn.relu(tf.matmul(layer1_n, w2_s) + b2)


#                self.l2_w_hist=tf.summary.histogram('l2_c_w_',w2_s)
#                self.summary_set.append(self.l2_w_hist)
#                self.l2_b_hist=tf.summary.histogram('l2_c_b_',b2)
#                self.summary_set.append(self.l2_b_hist)
                

            with tf.variable_scope('l2_normalization'):
                
                layer2_n=tf.layers.batch_normalization(layer2,name='layer2_n_')
                self.l2_hist=tf.summary.histogram('l2_c_n_',layer2_n)
                self.summary_set.append(self.l2_hist)
                
            with tf.variable_scope('l3'):
                n_l3 = H3
                w3_s = tf.get_variable('w3_s', [n_l2, n_l3], initializer=init_w, trainable=trainable)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
                layer3 = tf.nn.relu(tf.matmul(layer2_n, w3_s) + b3)


#                self.l3_w_hist=tf.summary.histogram('l3_c_w_',w3_s)
#                self.summary_set.append(self.l3_w_hist)
#                self.l3_b_hist=tf.summary.histogram('l3_c_b_',b3)
#                self.summary_set.append(self.l3_b_hist)              

            with tf.variable_scope('l3_normalization'):
                
                layer3_n=tf.layers.batch_normalization(layer3,name='layer3_n_')
                self.l3_hist=tf.summary.histogram('l3_n_c_',layer3_n)
                self.summary_set.append(self.l3_hist)
                
#            layer1 = tf.layers.dense(net, H1, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l2',
#                    trainable=trainable)
#            layer2 = tf.layers.dense(layer1, H1, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                    trainable=trainable)
            
#            layer2 = tf.layers.dense(layer1, H2, activation=tf.nn.relu,
#                    kernel_initializer=init_w, bias_initializer=init_b, name='l3',
#                    trainable=trainable)
            
            with tf.variable_scope('q'):

                wq_s = tf.get_variable('wq_s', [n_l3, 1], initializer=init_w, trainable=trainable)
#                bq = tf.get_variable('bq', [1, 1], initializer=init_b, trainable=trainable)
                q = tf.matmul(layer3_n, wq_s)
                
                self.q_hist=tf.summary.histogram('q_hist_',q)
                self.summary_set.append(self.q_hist)
#                self.q_w_hist=tf.summary.histogram('q_c_w_',wq_s)
#                self.summary_set.append(self.q_w_hist)
                
#                self.q_b_hist=tf.summary.histogram('q_c_b_',bq)
#                self.summary_set.append(self.q_b_hist)  
                
#                q = tf.layers.dense(layer3,1,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
#                q = tf.layers.dense(layer3,1,activation=tf.nn.tanh,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
                q_mean,q_var=tf.nn.moments(q,0)
                
                self.q_scalar=tf.summary.scalar('q_scalar_', tf.reshape(q_mean,[]))
                self.summary_set.append(self.q_scalar)  
                
                
                q_tg=tf.nn.tanh(q*0.1)
                
            with tf.variable_scope('q_scale'):
                q_scale=tf.matmul(q_tg, [[1.]]) 
        
        return q

    def learn(self, s, a, r, s_,ep_total):       

        self.sess.run(self.soft_replace)
        self.sess.run(self.train_op,feed_dict={S: s, self.a: a, R: r, S_: s_})

#        if self.t_replace_counter == self.t_replace_iter:
#            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
#            self.t_replace_counter = 0
#        self.t_replace_counter += 1
        self.loss_step+=1
        self.model_localization.append(ep_total)
        
    def merge_summary_end(self,s, a, r, s_):
        
        try:
            self.merge_summary = tf.summary.merge(self.summary_set)
            self.summary_critic=self.sess.run(self.merge_summary,feed_dict={S: s, self.a: a, R: r, S_: s_})
            self.writer.add_summary(self.summary_critic,self.loss_step)  
        except:
            print("nothing is here")
    
    def get_rank(self,s, a, r, s_):
        
        rank_q,rank_TD=self.sess.run([self.q,self.loss], feed_dict={S: s, self.a: a, R: r, S_: s_})

        return rank_TD
    
    def get_rank_probability(self,reward,max_v):
        
        rank_reward_correction=reward+(0-np.exp(max_v)*5/2)
        beta=3/(np.exp(max_v)*5/2)#95%,5%
        probability_reward=np.exp(beta*rank_reward_correction)/(1+np.exp(beta*rank_reward_correction))
        
#        rank_TD_correction=rank_TD+(0-(self.rank_TD_min+(self.rank_TD_max-self.rank_TD_min)/2))
#        beta=3/((self.rank_TD_max-self.rank_TD_min)/2)#99%,1%
#        probability_TD=np.exp(beta*rank_TD_correction)/(1+np.exp(beta*rank_TD_correction))
#        
#        return probability_q,probability_TD
        return probability_reward
        
class Memory(object):
    
    def __init__(self, capacity,capacity_bound, dims):
        self.capacity = capacity
        self.capacity_bound = capacity_bound
        self.dim=dims
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.epsilon=1e-16
        self.grow_trigger =False
        
    def store_transition(self, s, a, r, s_,rank_q,rank_TD):
        
#        s=(s-np.mean(s))/np.sqrt(np.std(s)**2+self.epsilon)
#        s_=(s_-np.mean(s_))/np.sqrt(np.std(s_)**2+self.epsilon)
        
        transition = np.hstack((s, a, [r], s_,rank_q,rank_TD))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1
        if index==self.capacity-1:
            self.grow_trigger=True
        if self.capacity<self.capacity_bound and self.grow_trigger==True:
#            print("self.capacitybefore:",self.capacity)
            self.capacity+= 1
#            print("self.capacityafter:",self.capacity)
#            print("self.databefore:",self.data)
            self.data=np.append(self.data,np.zeros((1,self.dim)),axis=0)
#            print("self.dataafter:",self.data)
#        elif self.capacity>=self.capacity_bound:

#            print("self.capacity:",self.capacity)
    def sample(self, n):
        
#        idx = critic.TD_step % critic.c_TD
#        critic.TD_set[idx]=rank_TD
#        critic.rank_TD_max=max(critic.TD_set)
#        critic.TD_set_no_zero=filter(lambda x: x !=0,critic.TD_set)
#        critic.TD_set_no_zero= [i for i in critic.TD_set_no_zero]
#        critic.rank_TD_min=min(critic.TD_set_no_zero)
#        critic.TD_step+=1

#        indices=[]
#        while len(indices)<n:
#            
#            idx=np.random.choice(self.capacity)
##            rank_q_temp=np.reshape(self.data[idx][-2:-1],())
#            p_temp_TD=np.reshape(self.data[idx, -2:-1],())
##            print("p_temp_TD:",p_temp_TD)
#            choice_TD=np.random.choice(range(2),p=[p_temp_TD,1-p_temp_TD])
#            if choice_TD==0:
#                
#                indices.append(idx)
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
    