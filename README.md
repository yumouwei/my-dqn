# My DQN & Double DQN
My own implementation of the Deep Q Network and Double Deep Q Network algorithms using Tensorflow. The implementations were tested in CartPole environment and achieved the maximum (500) episode score.

<img src="https://github.com/yumouwei/my-dqn/blob/main/models/cartpole-my-double-dqn-40k.gif" width="400" >  
<em>Double DQN Agent, 40,000 training steps, episode score=500</em>



## Algorithms

Descriptions of the DQN & Double DQN algorithms can be found in references below.

Essentially a **DQN** agent has the following components:

<ol>
  <li>An action-value function Q(state, action),</li>
  <li>A replay memory buffer (implemented using a <em>deque</em> object) for storing previous transitions = [state, action, reward, new_state, is_terminal].</li>
</ol>

The Q function explores the environment and generate new transitons. After each step a random minibatch is sampled and used to update the Q network through minibatch gradient descent:

<ul>
  <li>x = state</li>  
  <li>y = reward if is_terminal == True, else y = reward + gamma*max(Q(new_state, all possible actions))</li>
  <li>loss = MSE(x, y)
</ul>

Because the Q function is implemented to predict the value of each possible action given the current state (Q(state)->y_array(shape=(actions,))), therefore only the y value corresponding to the selected action (y_array[action]) is modified. Conceptually, the training process is similar to training a supervised learning model, except the training data is generated by the model itself through interacting with the environment, and the training labels are dynamic at each epoch & depend on model weights. Because of this, a small perturbation in network weights can be amplified and cause the training loss to explode.

---

In the **Double DQN** algorithm, the role of the Q function is split into two functions:

<ol>
  <li>An action Q function for exploring the environment, </li>
  <li>A target Q function for calculating the gradient descent target yi = reward + gamma*max(Q_target(new_state, actions)).</li>
</ol>

The weights of the target Q function are copied from the action Q function every <em>target network update frequency</em> steps and are held frozen until the next update. The action Q function is trained every <em>update frequency</em> steps. This approach prevents network weights & losses from exploding and stabilize the training process.


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
