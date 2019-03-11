#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:20:46 2019

@author: spikezz
"""
import tensorflow as tf
import numpy as np
import os
import shutil

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
    
    def save(self,sess,running_reward,reward_ep_mean):
        
        self.n_model+=1
        self.MODE.append(str(self.n_model))
        
        if os.path.isdir(self.di): shutil.rmtree(self.di)
        os.mkdir(self.di)
        ckpt_path = os.path.join( './Model/Model_'+self.MODE[self.n_model], 'DDPG.ckpt')
        save_path = self.saver.save(sess, ckpt_path, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)

        file = os.path.join( './Model/Model_'+self.MODE[self.n_model], 'episode_reward.txt')
        fw=open(file, mode='w')
        reward_str= 'running_reward:'+str(running_reward)+'\n'+'reward_ep_mean:'+str(reward_ep_mean)
        fw.seek(0,0)
        fw.write( reward_str)
        
        current_path = os.getcwd()
        model_dir = os.path.join(current_path, 'logs')
        writer=tf.summary.FileWriter(model_dir, sess.graph)
        writer.close()
        
class Summary(object):
    
    def summary(self,var,pointer,capacity,lr):
        
        print("var0:",var[0],"var1:",var[1],"var2:",var[2])
        print("MEMORY_pointer:",pointer)
        print("MEMORY_CAPACITY:",capacity)
        print("learning rate actor:",lr[0],"learning rate critic:",lr[1])
        if running_reward_max<running_reward and ep_total>1:
            
            running_reward_max=running_reward
            ep_lr=0
            max_reward_reset=max_reward_reset+1