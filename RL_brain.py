#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 02:37:09 2018

@author: zhe
"""

#import tensorflow as tf
import numpy as np
import pickle as pickle
import gym


#np.random.seed(1)
#tf.set_random_seed(1)

#hyper

h=200
batch_size=10
learning_rate=1e-4
#discount
gamma=0.99
decay_rate=0.99
resume=False

D=80*80

if resume:
    model=pickle.load(open('save.p','rb'))
else:
    model={}
    model['W1'] = np.random.randn(h,D) / np.sqrt(D)
    model['W2'] = np.random.randn(h) / np.sqrt(h)


grad_buffer = { k : np.zeros_like(v) for k,v in model.items()} 
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items()} 

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def prepro (I):
    I=I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    
    return I.astype(np.float).ravel() #flattens 

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    #initilize discount reward matrix as empty
    discounted_r = np.zeros_like(r)
    #to store reward sums
    running_add = 0
    #for each reward
    for t in reversed(range(0, r.size)):
        #if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific!)
        if r[t] != 0: running_add = 0 
        #increment the sum 
        #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
        running_add = running_add * gamma + r[t]
        #earlier rewards given more value over time 
        #assign the calculated sum to our discounted reward matrix
        discounted_r[t] = running_add
    return discounted_r

#forward propagation via numpy woot!
def policy_forward(x):
    #matrix multiply input by the first set of weights to get hidden state
    #will be able to detect various game scenarios (e.g. the ball is in the top, and our paddle is in the middle)
    h = np.dot(model['W1'], x)
    #apply an activation function to it
    #f(x)=max(0,x) take max value, if less than 0, use 0
    h[h<0] = 0 # ReLU nonlinearity
    #repeat process once more
    #will decide if in each case we should be going UP or DOWN.
    logp = np.dot(model['W2'], h)
    #squash it with an activation (this time sigmoid to output probabilities)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph,epdlogp):
    
    dW2 = np.dot(eph.T, epdlogp).ravel()
    
    dh = np.outer(epdlogp, model['W2'])
    
    dh[eph <= 0] = 0 
    
    dW1 = np.dot(dh.T, epx)
    
    return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
#Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
#The process gets started by calling reset, which returns an initial observation
observation = env.reset()
prev_x = None # used in computing the difference frame
#observation, hidden state, gradient, reward
xs,hs,dlogps,drs = [],[],[],[]
#current reward
running_reward = None
#sum rewards
reward_sum = 0
#where are we?
episode_number = 0


while True:

    # preprocess the observation, set input to network to be difference image
    #Since we want our policy network to detect motion
    #difference image = subtraction of current and last frame
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    #so x is our image difference, feed it in!
    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    #this is the stochastic part 
    #since not apart of the model, model is easily differentiable
    #if it was apart of the model, we'd have to use a reparametrization trick (a la variational autoencoders. so badass)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    
    
    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    if done: # an episode finished
        episode_number += 1
    
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        #each episode is a few dozen games
        epx = np.vstack(xs) #obsveration
        eph = np.vstack(hs) #hidden
        epdlogp = np.vstack(dlogps) #gradient
        epr = np.vstack(drs) #reward
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    
        #the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important
        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
    
        #advatnage - quantity which describes how good the action is compared to the average of all the action.
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
    
        # perform rmsprop parameter update every batch_size episodes
        #http://68.media.tumblr.com/2d50e380d8e943afdfd66554d70a84a1/tumblr_inline_o4gfjnL2xK1toi3ym_500.png
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
    
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
    
    
    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')