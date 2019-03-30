#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:49:30 2019

@author: spikezz
"""

import RL


class Agent_Imitation(object):
    
    def __init__(self,sess,action_dim,input_dim,action_bound,lr_i,replace_iter_a):

        self.actor = RL.Actor(sess, action_dim, input_dim, action_bound, 0, lr_i, replace_iter_a)
        
        
class Agent_Reinforcement(object):
    
    def __init__(self,sess,action_dim,input_dim,action_bound,lr_a,lr_c,rd,replace_iter_a,replace_iter_c,C_TD):
        
        self.actor = RL.Actor(sess, action_dim, input_dim, action_bound, lr_a, 0, replace_iter_a)
        self.critic = RL.Critic(sess, input_dim, action_dim, lr_c, rd, replace_iter_c , self.actor.a, self.actor.a_, C_TD)
        self.actor.add_grad_to_graph(self.critic.a_grads)