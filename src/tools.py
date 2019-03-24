#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:20:46 2019

@author: spikezz
"""
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

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

#        current_path = os.getcwd()
#        model_dir = os.path.join(current_path, 'logs')


        
class Summary(object):
    
    def summary(self,LOAD,var,pointer,capacity,reward_ep,running_reward,
                running_reward_max,reward_mean,rr_idx,max_reward_reset,
                action_ori,actor,reward_one_ep_mean,rr,critic,reward_mean_max_rate,ep_lr,time_set):
        
        print("LOAD:",LOAD)
        print("critic.rank_TD_max:",critic.rank_TD_max,"critic.rank_TD_min:",critic.rank_TD_min)
#        print("FPS:",clock.get_fps())
        print("var0:",var[0],"var1:",var[1],"var2:",var[2])
        print("MEMORY_pointer:",pointer)
        print("MEMORY_CAPACITY:",capacity)
        print("running_reward:",running_reward)
        print("max_running_reward:",running_reward_max)
        print("reward_mean:",reward_mean[rr_idx-1])
        print("max_reward_reset:",max_reward_reset)
        print("lr ep :",ep_lr)
         #accelerate 
        plt.figure()
        plt.subplot(211)
        plt.plot(action_ori[0])  
        plt.xlabel('steps')
        plt.ylabel('a0 output')
        plt.subplot(212)
        plt.plot(actor.accelerate)  
        plt.xlabel('steps')
        plt.ylabel('accelerate')
        #accelerate      
        
        #brake
        plt.figure()
        plt.subplot(211)
        plt.plot(action_ori[1])  
        plt.xlabel('steps')
        plt.ylabel('a1 output')
        plt.subplot(212)
        plt.plot(actor.brake)  
        plt.xlabel('steps')
        plt.ylabel('brake')
        #brake
        
        #steering angle
        plt.figure()
        plt.subplot(211)
        plt.plot(action_ori[2])  
        plt.xlabel('steps')
        plt.ylabel('a2 output')
        plt.subplot(212)
        plt.plot(actor.angle)  
        plt.xlabel('steps')
        plt.ylabel('angle')
        #steering angle
        
        #probability
        plt.figure()
        plt.subplot(311)
        plt.plot(action_ori[3])  
        plt.xlabel('steps')
        plt.ylabel('p accelerate')
        plt.subplot(312)
        plt.plot(action_ori[4])  
        plt.xlabel('steps')
        plt.ylabel('p brake')
        plt.subplot(313)
        plt.plot(action_ori[5])  
        plt.xlabel('steps')
        plt.ylabel('p idle')
        #probability
        
        
        #reward
        plt.figure()
        plt.subplot(211)
        plt.plot(reward_ep)  
        plt.xlabel('steps')
        plt.ylabel('reward one ep')
        plt.subplot(212)
        plt.plot(reward_one_ep_mean)  
        plt.xlabel('episode steps')
        plt.ylabel('reward mean for one ep')
        plt.figure()
        plt.subplot(311)
        plt.plot(rr)  
        plt.xlabel('episode steps')
        plt.ylabel('runing reward whole episode')
        plt.subplot(312)
        plt.plot(reward_mean)  
        plt.xlabel('episode steps')
        plt.ylabel('reward_mean')
        plt.subplot(313)
        plt.plot(reward_mean_max_rate)  
        plt.xlabel('episode steps')
        plt.ylabel('reward Max/mean')
        
        plt.figure()
        plt.subplot(211)
        plt.plot(critic.model_localization)
        plt.xlabel('learning steps')
        plt.ylabel('model_localization')
        plt.subplot(212)
        plt.plot(time_set)
        plt.xlabel('cycle steps')
        plt.ylabel('time_set')

        
        plt.show()
        
