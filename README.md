# My DQN & Double DQN
My own implementation of the Deep Q Network and Double Deep Q Network algorithms using Tensorflow. The implementations were tested in CartPole environment and achieved the maximum (500) episode score.

<img src="https://github.com/yumouwei/my-dqn/blob/main/models/cartpole-my-double-dqn-40k.gif" width="400" >

## Algorithms



## Experiment

The Double DQN agent was tested in CartPole-v1 environment using the following hyperparameters:


network: fully connected, 2x64, ‘relu’  
minibatch size: 32  
replay memory size: 1,000  
replay start size: 500  
target network update frequency: 100  
discount factor (gamma): 0.99  
update frequency: 4  
learning_rate: 1e-3  
optimizer: Adam  
clipnorm: 10  
initial exploration: 1  
final exploration: 0.05  
exploration fraction: 0.1  
total timesteps: 40,000  


## Reference

Mnih, 2013, Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602  
Mnih, 2015, Human-level control through deep reinforcement learning: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf  
Hasselt, 2015, Deep Reinforcement Learning with Double Q-learning: https://arxiv.org/abs/1509.06461
