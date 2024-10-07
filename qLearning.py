import tkinter as tk
import numpy as np
import random

class QLearning:
   """
   Q-Learning algorithm for agent navigation on a grid with obstacles and a goal.

   Attributes:
       alpha (float): Learning rate for Q-learning updates.
       gamma (float): Discount factor for future rewards.
       epsilon (float): Exploration rate for the epsilon-greedy strategy.
       epochs (int): Number of training epochs.

       exploration_steps_delay (int): Delay between exploration steps in milliseconds.
       epochs_delay (int): Delay between epochs in milliseconds.
       test_steps_delay (int): Delay between steps during testing in milliseconds.
       
       matrix (list): Grid representing the environment (0 for empty, 1 for obstacles, -1 for goal).
       r_matrix (numpy.ndarray): Reward matrix for state-action pairs.
       q_matrix (numpy.ndarray): Q-value matrix for state-action pairs.
       actions (dict): Dictionary mapping actions to their effects on the agent's position.
   """

   def __init__ (self, _matrix):
      """
      Initialize the QLearning instance with the environment matrix.

      Args:
          _matrix (list): The grid environment (2D matrix) where 0 is empty space, 1 is an obstacle, and -1 is the goal.
      """
      self.alpha = 0.8 
      self.gamma = 0.9  
      self.epsilon = 0.5  
      self.epochs = 1000

      self.exploration_steps_delay = 10
      self.epochs_delay = 50
      self.test_steps_delay = 100

      self.matrix = _matrix
      self.r_matrix = None  
      self.q_matrix = None

      self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


      self.initialize_r_matrix()
      self.initialize_q_matrix()
   
   def initialize_r_matrix(self):
      """
      Initialize the reward matrix (R-matrix) based on the environment grid.

      The reward matrix is filled with values representing rewards for state-action pairs.
      Obstacles have a reward of -1, valid moves have a reward of 0, and the goal has a reward of 100.
      """
      n = len(self.matrix) 
      num_states = n * n   
      self.r_matrix = np.zeros((num_states, len(self.actions)))
        
      invalid_move_value = -1 
      exit_reward_value = 100 

      for i in range(n):
         for j in range(n):
               current_state = i * n + j

               if self.matrix[i][j] == 1:
                    continue
               
               if self.matrix[i][j] == -1:
                  self.r_matrix[current_state][0] = invalid_move_value
                  self.r_matrix[current_state][1] = invalid_move_value
                  self.r_matrix[current_state][2] = invalid_move_value
                  self.r_matrix[current_state][3] = invalid_move_value
                   

               for action_idx, (action, (di, dj)) in enumerate(self.actions.items()):
                    new_i, new_j = i + di, j + dj
                    
                    if 0 <= new_i < n and 0 <= new_j < n:
                        if self.matrix[new_i][new_j] == 1:
                           self.r_matrix[current_state, action_idx] = invalid_move_value
                        elif self.matrix[new_i][new_j] == -1:
                           self.r_matrix[current_state, action_idx] = exit_reward_value 
                    else:
                        self.r_matrix[current_state, action_idx] = invalid_move_value 

   def initialize_q_matrix(self):
      """
      Initialize the Q-matrix with zeros.

      The Q-matrix represents the learned values for state-action pairs. It is initialized with zeros, except for the goal state which gets its reward value.
      """
      n = len(self.r_matrix) 
      self.q_matrix = np.zeros((n, len(self.actions)))

      for i in range(n):
         for j in range(len(self.actions)):
            if (self.r_matrix[i][j] == 100):
               self.q_matrix[i][j] = self.r_matrix[i][j]

   def choose_action(self, state):
      """
      Choose an action based on the current Q-values for the given state.

      Args:
          state (int): The current state of the agent.

      Returns:
          int: The index of the chosen action.
      """
      if np.all(self.q_matrix[state] == self.q_matrix[state][0]):
         action_id = random.randint(0, len(self.actions) - 1)
      else:
         action_id = np.argmax(self.q_matrix[state])
      return action_id
   

   def update_q_value(self, state, action_idx, reward, new_state):
      """
      Update the Q-value for a state-action pair using the Q-learning update rule.

      Args:
          state (int): The current state of the agent.
          action_idx (int): The index of the chosen action.
          reward (float): The reward received after taking the action.
          new_state (int): The resulting state after taking the action.
      """
      current_q = self.q_matrix[state, action_idx]
      max_future_q = np.max(self.q_matrix[new_state])
      new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
      self.q_matrix[state, action_idx] = new_q

   def is_valid_move(self, new_x, new_y):
      """
      Check if the move is valid (i.e., within bounds and not into an obstacle).

      Args:
          new_x (int): The new x-coordinate after the move.
          new_y (int): The new y-coordinate after the move.

      Returns:
          bool: True if the move is valid, False otherwise.
      """
      n = len(self.matrix)
      return 0 <= new_x < n and 0 <= new_y < n and self.matrix[new_x][new_y] != 1 

   def train(self, agent ,_epoch = 0):
      """
      Train the agent using Q-learning for a specified number of epochs.

      Args:
          agent (object): The agent object that moves within the environment.
          _epoch (int, optional): The current epoch of the training. Defaults to 0.
      """
      if _epoch < self.epochs:
         self.explore(agent)
         agent.reset(0,0)
         agent.canvas.after(self.epochs_delay, self.train, agent, _epoch+1)  
         print(self.q_matrix)
      else:
         print(self.q_matrix)
         print("Success")
         agent.canvas.itemconfig(agent.icon, fill='#76ABAE')

   def explore(self, agent, done = False, state=None):
      """
      Explore the environment, allowing the agent to take actions and update its Q-values.

      Args:
          agent (object): The agent object that moves within the environment.
          done (bool, optional): Whether the agent has reached the goal. Defaults to False.
          state (int, optional): The current state of the agent. Defaults to None.
      """
      if state is None:
         n = len(self.matrix)
         state = agent.x * n + agent.y 

      if self.matrix[agent.x][agent.y] == -1:
        print("Agent has reached the finish state!")
        done = True  
        return  

      if done == False:
         action_id = self.choose_action(state)  
         action = list(self.actions.keys())[action_id] 

         new_x, new_y = agent.x, agent.y
         if action == 'up':
               new_x, new_y = agent.x - 1, agent.y
         elif action == 'down':
               new_x, new_y = agent.x + 1, agent.y
         elif action == 'left':
               new_x, new_y = agent.x, agent.y - 1
         elif action == 'right':
               new_x, new_y = agent.x, agent.y + 1

         if self.is_valid_move(new_x, new_y):
               current_state = agent.x * len(self.matrix) + agent.y
               reward = self.r_matrix[current_state, action_id]

               agent.move(new_x, new_y)

               new_state = new_x * len(self.matrix) + new_y

               self.update_q_value(current_state, action_id, reward, new_state) 

               state = new_state
               self.epsilon = max(0.01, self.epsilon * 0.99)

      agent.canvas.after(self.exploration_steps_delay, self.explore, agent, done, state) 

   def test_agent(self, agent, done = False, state = None):
      """
      Test the agent's learned policy after training by running it through the environment.

      Args:
          agent (object): The agent object that moves within the environment.
          done (bool, optional): Whether the agent has reached the goal. Defaults to False.
          state (int, optional): The current state of the agent. Defaults to None.
      """
      n = len(self.matrix)
      state = agent.x * n + agent.y

      if done == False:
         if self.matrix[agent.x][agent.y] == -1:
            print("Agent has reached the finish state during testing!")
            return

         action_id = np.argmax(self.q_matrix[state])
         action = list(self.actions.keys())[action_id] 

         new_x, new_y = agent.x, agent.y
         if action == 'up':
               new_x, new_y = agent.x - 1, agent.y
         elif action == 'down':
               new_x, new_y = agent.x + 1, agent.y
         elif action == 'left':
               new_x, new_y = agent.x, agent.y - 1
         elif action == 'right':
               new_x, new_y = agent.x, agent.y + 1

         if self.is_valid_move(new_x, new_y):
               agent.move(new_x, new_y)
               state = new_x * n + new_y

         agent.canvas.after(self.test_steps_delay, self.test_agent, agent, done, state)