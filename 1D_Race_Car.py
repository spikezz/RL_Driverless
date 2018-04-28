"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

"""
@author: Asgard

Change a little bit from the original explorer to a old 1D race car game,
for the original code pls go to  https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd
import time
#import math


np.random.seed() 

N_STATES = 11 # the length of the 1 dimensional world
ACTIONS = ['left_2','left_1', 'hold','right_1','right_2']  # available actions
EPSILON = 0.99   # greedy police je stabil die umgebung je höhe die Epsilon
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100  # maximum episodes
FRESH_TIME = 0.0001    # fresh time for one move 


def build_q_table(n_states, actions):
     table = pd.DataFrame(np.zeros((n_states, len(actions))),columns=actions)# q_table initial values,actions's name
     return table


def choose_action(state, q_table):
     state_actions = q_table.iloc[state, :]
     if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act greedy
          action_name = np.random.choice(ACTIONS)
     else:    # act non-greedy or state-action have no value
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
     return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'left_2':    # move right
        
        if S == N_STATES - 2 or S == N_STATES - 3:   # terminate
            
            S_ = 'terminal'
            R = -1   
            
        else:
            
            S_ = S + 2            
            R = 0.01*(10/(1+abs((S-1)-(N_STATES-S-2)))-1)
            
    elif A == 'left_1':    # move right
        
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = -1
            
        else:
            
            S_ = S + 1           
            R = 0.01*(10/(1+abs((S-1)-(N_STATES-S-2)))-1)
            
    elif A == 'hold':    # hold
        
        S_ = S        
        R = 0.01*(10/(1+abs((S-1)-(N_STATES-S-2)))-1)     
            
    elif A == 'right_1':   # move left
        
        if S == 1:   # terminate
            
            S_ = 'terminal'
            R = -1
            
        else:
            
            S_ = S - 1            
            R = 0.01*(10/(1+abs((S-1)-(N_STATES-S-2)))-1)
            
    elif A == 'right_2':   # move left
        
        if S == 1 or S==2:   # terminate
            
            S_ = 'terminal'
            R = -1
            
        else:
            
            S_ = S - 2            
            R = 0.01*(10/(1+abs((S-1)-(N_STATES-S-2)))-1)       
        
    return S_, R

#map building
#def right_shift(step_counter,curving_factor):
    
    #if  step_counter<50:#straight
        
        #shiftright=0
        
    #elif step_counter>=50 & step_counter<=100:#turn left 
        
        #shiftright=math.floor((step_counter-50)/curving_factor)
        
    #elif step_counter>100:
        
        #shiftright=math.floor((100-50)/curving_factor)
        
    #return shiftright   
    
    
def update_env(S, episode, step_counter):
    # This is how environment be updated
    
    #shift=right_shift(step_counter,4)
    #shift_=right_shift(step_counter+1,4)
    
    #env_list = [' ']*shift+['|']+[' ']*(N_STATES-2) + ['|']   # '---------T' our environment
    env_list =['|']+[' ']*(N_STATES-2) + ['|']   # '---------T' our environment
    if S == 'terminal':
        
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\n{}\n'.format(interaction), end='')
        #time.sleep(2)
        #print('\n                                ', end='')
        
    else:
        
        
        #env_list[S+shift] = 'V'
        env_list[S] = 'V'
        interaction = ''.join(env_list)
        print('\r{}\n'.format(interaction), end='')
        
        
        #env_list_ = [' ']*shift_+['|']+[' ']*(N_STATES-2) + ['|'] 
        env_list_ =['|']+[' ']*(N_STATES-2) + ['|'] 
        interaction_=''.join(env_list_)
        print('{}'.format(interaction_), end='')
            
        time.sleep(FRESH_TIME)
        
    #return shift_


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    step_table=build_q_table(MAX_EPISODES,['step'])
    maxstep=0
    step=[]
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 3
        is_terminated = False      
        
        #shift_s=update_env(S, episode, step_counter)##initialize
        update_env(S, episode, step_counter)##initialize
        while ((not is_terminated)and(step_counter<1000)):
             A = choose_action(S, q_table)
             S_, R = get_env_feedback(S, A)  # take action & get next state and reward
             q_predict = q_table.loc[S, A]
             if S_ != 'terminal':
                 q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
             else:
                 q_target = R   # next state is terminal
                 is_terminated = True    # terminate this episode
                 
             q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
             S = S_  # move to next state
             
             
             #shift_s=update_env(S, episode, step_counter+1)
             update_env(S, episode, step_counter+1)
             step_counter += 1  
        
        step_table.loc[episode,'step']=step_counter
        step.append(step_counter)

        if(step_counter>=maxstep):            
            maxstep = step_counter      
        
    print("\nmaxstep:%s\n" % (maxstep))   
    print("all steps:%s\n" % (step))
    print('\r\nstep:\n')
    print(step_table)
    return q_table                 
    
    
if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
        
        
        
        