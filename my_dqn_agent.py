#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:22:41 2022

@author: yumouwei

References: 
    DQN: https://arxiv.org/abs/1312.5602
    Double DQN: https://www.nature.com/articles/nature14236
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import random
from collections import deque
import time
import os

import imageio

class ReplayMemoryBuffer():
    '''
    Replay memory buffer class for the DQN agent
    transition = [state, action, reward, new_state, is_terminal]
    '''
    def __init__(self, buffer_size=1e6):
        self.replay_memory = deque(maxlen=buffer_size) # deque automatically removes oldest transition from memory
        self.buffer_size = buffer_size
        
    def get_current_size(self):
        # Return the current size of the replay memory
        return len(self.replay_memory)

    def store_transition(self, transition):
        '''
        Store new transition to the buffer
        Remove oldest transition if total memory exceeds buffer_size
        '''
        if len(transition) == 5: # check the shape of new transition
            self.replay_memory.append(transition)
        else:
            return False
    
    def sample_batch(self, batch_size=32):
        # Sample a minibatch from memory buffer
        batch = random.sample(self.replay_memory, batch_size)
        return batch


def build_mlp_network(input_shape, output_shape, dense_layers=[64, 64], learning_rate=1e-3, max_grad_norm=10):
    # Build MLP neural network
    # inputs = current state; outputs = prob. of all possible actions
    
    x = inputs = Input(shape = input_shape, name='inputs')
    for i, f in enumerate(dense_layers):
        x = layers.Dense(f, name='dense_%i'%i)(x)
        x = layers.Activation('relu', name='dense_act_%i'%i)(x)
    outputs = layers.Dense(output_shape, name='outputs')(x)
    
    model = Model(inputs, outputs)
    adam = Adam(learning_rate=learning_rate, clipnorm=max_grad_norm)
    model.compile(optimizer=adam, loss='mse')
    
    return model

    
class MyDQNAgent():
    def __init__(self, env, learning_rate=1e-4, buffer_size=1e6, batch_size=32, 
                 gamma=0.99, exploration_fraction=0.1, exploration_initial_eps=1.0, 
                 exploration_final_eps=0.05, max_grad_norm=10):
        '''
        gamma (float): discount factor
        epsilon: explore-exploit factor, probability of taking a random action during training 
            epsilon starts from [exploration_initial_eps] and reduces linearly 
            down to [exploration_final_eps] in [total_timesteps] * [exploration_fraction]
            number of steps.
        max_grad_norm (float): maximum value for gradient clipping
        '''
        
        self.env = env
        self.replay_memory = ReplayMemoryBuffer(buffer_size=int(buffer_size))
        self.q_network = build_mlp_network(self.env.observation_space.shape, self.env.action_space.n,
                                           dense_layers=[64, 64], learning_rate=learning_rate, max_grad_norm=max_grad_norm)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.gamma = gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        
        self.training_steps=0
    
    
    def choose_action(self, state, epsilon):
        '''
        Choose an action given the current state and epsilon-value
        If epsilon = 0, then choose deterministic action (deterministic=True)
        In gym env, action is an integer, not a one-hot encoding
        '''
        seed = random.random()
        if seed < epsilon: # take random action
            action = random.randint(0, self.env.action_space.n - 1)
        else: # follow greedy strategy
            actions = self.q_network.predict(np.array([state]), verbose=0)
            action = np.argmax(actions)
        return action
    
    
    def train_step(self, epsilon, terminal_state=False):
        '''
        Sample a minibatch from replay memory, then train the q network for 1 epoch
        '''      
        # start training only if replay_memory is filled up
        if self.replay_memory.get_current_size() == self.buffer_size:
            
            # sample a minibatch from replay memory
            train_batch = self.replay_memory.sample_batch(self.batch_size) # list
            
            ''' 
            #extremely time intensive ~1.2 s
            # transition = [state, action, reward, new_state, is_terminal]
            x_batch = np.zeros((self.batch_size, self.env.observation_space.shape[0]))
            y_batch = np.zeros((self.batch_size, self.env.action_space.n))
            
            for i in range(self.batch_size):
                x_batch[i] = train_batch[i][0]
                
                y = self.q_network.predict(np.array([train_batch[i][0]]), verbose=0)
                y = y[0]
                if train_batch[i][4] == True:  # terminal state
                    y[train_batch[i][1]] = train_batch[i][2]  # yj = reward
                else:   #non-terminal state
                    q_new_state = self.q_network.predict(np.array([train_batch[i][3]]), verbose=0)
                    y[train_batch[i][1]] = train_batch[i][2] + self.gamma * max(q_new_state[0]) # equation (3)
                y_batch[i] = y
            
            #print(time.time() - TIME) # ~1.3 s per iteration
            
            # perform a gradient descent step
            self.q_network.fit(x_batch, y_batch, verbose=0, epochs=1, batch_size=32)
            '''

            # ~0.04 s upto model.fit
            # transition = [state, action, reward, new_state, is_terminal]
            state_batch = np.zeros((self.batch_size, self.env.observation_space.shape[0]))
            action_batch = np.zeros(self.batch_size)
            reward_batch = np.zeros(self.batch_size)
            new_state_batch = np.zeros((self.batch_size, self.env.observation_space.shape[0]))
            done_batch = np.zeros(self.batch_size)
            
            for i in range(self.batch_size):
                state_batch[i] = train_batch[i][0]
                action_batch[i] = train_batch[i][1]
                reward_batch[i] = train_batch[i][2]
                new_state_batch[i] = train_batch[i][3]
                done_batch[i] = train_batch[i][4]
            
            q_state_batch = self.q_network.predict(state_batch, verbose=0)
            q_new_state_batch = self.q_network.predict(new_state_batch, verbose=0)
            
            for i in range(self.batch_size):
                if done_batch[i] == True:  # terminal state
                    q_state_batch[i][int(action_batch[i])] = reward_batch[i]
                else:  # non-terminal state
                    q_state_batch[i][int(action_batch[i])] = reward_batch[i] + self.gamma * max(q_new_state_batch[i])
            
            # perform a gradient descent step
            self.q_network.fit(state_batch, q_state_batch, verbose=2, epochs=1, batch_size=32)
            
            
    def learn(self, total_timesteps, save_freq=False, save_path='models/'):
        '''
        Train the DQN agent for [total_timesteps] number of steps
        SAVE_FREQ: save current model every [SAVE_FREQ] steps
        '''
        exploration_final_step = total_timesteps * self.exploration_fraction
        exploration_slope_eps = self.exploration_final_eps - self.exploration_initial_eps
        
        # reset environment before training
        state = self.env.reset()
        current_episode = 0
        episode_score = 0
        
        for step in range(int(total_timesteps)):
            # get the current value of epsilon
            epsilon = max((step - exploration_final_step)/exploration_final_step * exploration_slope_eps, 0) + self.exploration_final_eps
            #print('step='+str(step) + ', epsilon=' + str(epsilon))
            
            # select and execute an action
            action = self.choose_action(state, epsilon)
            new_state, reward, done, info = self.env.step(action)
            episode_score += reward
            
            # store new transition to replay memory
            # transition = [state, action, reward, new_state, is_terminal]
            transition = [state, action, reward, new_state, done]
            self.replay_memory.store_transition(transition)
            
            # do one train_step()
            self.train_step(epsilon, done)
            self.training_steps += 1
            
            if save_freq and self.training_steps % save_freq == 0:
                self.save_network(save_path + 'model_step_' + str(self.training_steps)+'.h5')
            
            # update state
            if done == True: 
                state = self.env.reset()
                print('Episode '+str(current_episode)+' (step '+str(step)+'): '+str(episode_score))
                current_episode += 1
                episode_score = 0
            else:
                state = new_state
                
    def evaluate(self, n_episode=1, render=False):
        '''
        Evaluate the current agent for [n_episode] episodes
        '''

        for i in range(n_episode):
            state = self.env.reset()
            done = False
            score = 0
            
            while not done:
                if render == True:
                    self.env.render()
                    time.sleep(0)
                action = self.choose_action(state, epsilon=0)
                state, reward, done, info = self.env.step(action)
                score += reward
                
            print('Episode:{} Score:{}'.format(i, score))
    
    def save_network(self, file='model.h5'):
        self.q_network.save(file)
    
    def load_network(self, file='model.h5'):
        self.q_network = tf.keras.models.load_model(file)
        
    def make_animation(self, filename='gym_animation.gif', RETURN_FRAMES=False):
        '''
        Make an animation of the rendered screen
        '''
        # run policy
        frames = []
        state = self.env.reset()
        done = False
        
        while not done:
            #frames.append(self.env.render(mode="rgb_array"))
            im = self.env.render(mode="rgb_array")
            frames.append(im.copy())
            action = self.choose_action(state, epsilon=0)
            state, reward, done, info = self.env.step(action)
            
        if RETURN_FRAMES == False:
            # make animation
            imageio.mimsave(filename, frames, fps=50)
        else: # make animation manually in case Mario gets stuck in the level and drags the animation for too long
            return frames
            
        
        
            
class MyDoubleDQNAgent():
    def __init__(self, env, dense_layers=[64, 64], learning_rate=2.5e-4, buffer_size=1e6, 
                 learning_start=50e3, batch_size=32, gamma=0.99, train_freq=4, 
                 target_update_interval=10000, exploration_fraction=0.1, 
                 exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10):
        '''
        gamma (float): discount factor
        epsilon: explore-exploit factor, probability of taking a random action during training 
            epsilon starts from [exploration_initial_eps] and reduces linearly 
            down to [exploration_final_eps] in [total_timesteps] * [exploration_fraction]
            number of steps.
        max_grad_norm (float): maximum value for gradient clipping
        '''
        
        self.env = env
        self.replay_memory = ReplayMemoryBuffer(buffer_size=int(buffer_size))
        self.action_q_network = build_mlp_network(self.env.observation_space.shape, self.env.action_space.n,
                                                  dense_layers=dense_layers, learning_rate=learning_rate, max_grad_norm=max_grad_norm)
        self.target_q_network = build_mlp_network(self.env.observation_space.shape, self.env.action_space.n,
                                                  dense_layers=dense_layers, learning_rate=learning_rate, max_grad_norm=max_grad_norm)
        self.target_q_network.set_weights(self.action_q_network.get_weights())
        
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_start = learning_start
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        
        self.training_steps=0
    
    
    def choose_action(self, state, epsilon):
        '''
        Choose an action given the current state and epsilon-value
        Use action q network
        If epsilon = 0, then choose deterministic action (deterministic=True)
        In gym env, action is an integer, not a one-hot encoding
        '''
        seed = random.random()
        if seed < epsilon: # take random action
            action = random.randint(0, self.env.action_space.n - 1)
        else: # follow greedy strategy
            actions = self.action_q_network.predict(np.array([state]), verbose=0)
            action = np.argmax(actions)
        return action
    
    
    def train_step(self, epsilon, terminal_state=False):
        '''
        Sample a minibatch from replay memory, then train the q network for 1 epoch
        '''      
        # starts learning before filling up the replay memory buffer
        if self.replay_memory.get_current_size() > self.learning_start:   
            
            # sample a minibatch from replay memory
            train_batch = self.replay_memory.sample_batch(self.batch_size) # list

            # transition = [state, action, reward, new_state, is_terminal]
            state_batch = np.zeros((self.batch_size, self.env.observation_space.shape[0]))
            action_batch = np.zeros(self.batch_size)
            reward_batch = np.zeros(self.batch_size)
            new_state_batch = np.zeros((self.batch_size, self.env.observation_space.shape[0]))
            done_batch = np.zeros(self.batch_size)
            
            for i in range(self.batch_size):
                state_batch[i] = train_batch[i][0]
                action_batch[i] = train_batch[i][1]
                reward_batch[i] = train_batch[i][2]
                new_state_batch[i] = train_batch[i][3]
                done_batch[i] = train_batch[i][4]
            
            q_state_batch = self.action_q_network.predict(state_batch, verbose=0)  # update and train action network
            q_new_state_batch = self.target_q_network.predict(new_state_batch, verbose=0)  # get target q from target network
            
            for i in range(self.batch_size):
                if done_batch[i] == True:  # terminal state
                    q_state_batch[i][int(action_batch[i])] = reward_batch[i]
                else:  # non-terminal state
                    q_state_batch[i][int(action_batch[i])] = reward_batch[i] + self.gamma * max(q_new_state_batch[i])
            
            # perform a gradient descent step
            self.action_q_network.fit(state_batch, q_state_batch, verbose=2, epochs=1, batch_size=self.batch_size)  # update and train action network
            
            
    def learn(self, total_timesteps, save_freq=False, save_path='models/'):
        '''
        Train the DQN agent for [total_timesteps] number of steps
        SAVE_FREQ: save current model every [SAVE_FREQ] steps
        '''
        exploration_final_step = total_timesteps * self.exploration_fraction
        exploration_slope_eps = self.exploration_final_eps - self.exploration_initial_eps
        
        # reset environment before training
        state = self.env.reset()
        current_episode = 0
        episode_score = 0
        
        for step in range(int(total_timesteps)):                
            # get the current value of epsilon
            epsilon = max((step - exploration_final_step)/exploration_final_step * exploration_slope_eps, 0) + self.exploration_final_eps
            
            # select and execute an action
            action = self.choose_action(state, epsilon)
            new_state, reward, done, info = self.env.step(action)
            episode_score += reward
            
            # store new transition to replay memory
            # transition = [state, action, reward, new_state, is_terminal]
            transition = [state, action, reward, new_state, done]
            self.replay_memory.store_transition(transition)
            
            # do one train_step()
            if self.training_steps % self.train_freq == 0:  # train network every [self.train_freq] steps
                self.train_step(epsilon, done)
            self.training_steps += 1
            
            # update target q network
            if self.training_steps % self.target_update_interval == 0:
                self.target_q_network.set_weights(self.action_q_network.get_weights())
            
            if save_freq and self.training_steps % save_freq == 0:
                self.save_network(save_path + 'model_step_' + str(self.training_steps)+'.h5')
            
            # update state
            if done == True: 
                state = self.env.reset()
                print('Episode '+str(current_episode)+' (step '+str(step)+'): '+str(episode_score))
                current_episode += 1
                episode_score = 0
            else:
                state = new_state
                
    def evaluate(self, n_episode=1, render=False):
        '''
        Evaluate the current agent for [n_episode] episodes
        '''

        for i in range(n_episode):
            state = self.env.reset()
            done = False
            score = 0
            
            while not done:
                if render == True:
                    self.env.render()
                    time.sleep(0)
                action = self.choose_action(state, epsilon=0)
                state, reward, done, info = self.env.step(action)
                score += reward
                
            print('Episode:{} Score:{}'.format(i, score))
    
    def save_network(self, file='model.h5'):
        self.action_q_network.save(file)
    
    def load_network(self, file='model.h5'):
        self.action_q_network = tf.keras.models.load_model(file)
        self.target_q_network = tf.keras.models.load_model(file)
        
    def make_animation(self, filename='gym_animation.gif', RETURN_FRAMES=False):
        '''
        Make an animation of the rendered screen
        '''
        # run policy
        frames = []
        state = self.env.reset()
        done = False
        
        while not done:
            #frames.append(self.env.render(mode="rgb_array"))
            im = self.env.render(mode="rgb_array")
            frames.append(im.copy())
            action = self.choose_action(state, epsilon=0)
            state, reward, done, info = self.env.step(action)
            
        if RETURN_FRAMES == False:
            # make animation
            imageio.mimsave(filename, frames, fps=50)
        else: # make animation manually in case Mario gets stuck in the level and drags the animation for too long
            return frames
                             
            
            
