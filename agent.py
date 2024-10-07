import tkinter as tk
import numpy as np
import random
from qLearning import QLearning

class Agent:
   def __init__(self, canvas, start_x, start_y, cell_size, matrix):
      self.canvas = canvas
      self.x = start_x
      self.y = start_y
      self.cell_size = cell_size
      self.icon = self.canvas.create_oval(self.x * self.cell_size + 10, 
                                          self.y * self.cell_size + 10,
                                          (self.x + 1) * self.cell_size - 10, 
                                          (self.y + 1) * self.cell_size - 10, 
                                          fill='#f26065')
      
      self.grid_matrix = matrix
      self.find_path = QLearning(matrix)
   
   def move(self, new_x, new_y):
      self.x = new_x
      self.y = new_y
      self.canvas.coords(self.icon, 
                        self.y * self.cell_size + 10, 
                        self.x * self.cell_size + 10,
                        (self.y + 1) * self.cell_size - 10, 
                        (self.x + 1) * self.cell_size - 10)
      
   def reset(self, start_x, start_y):
      self.x = start_x
      self.y = start_y
      self.canvas.coords(self.icon, 
                        self.y * self.cell_size + 10, 
                        self.x * self.cell_size + 10,
                        (self.y + 1) * self.cell_size - 10, 
                        (self.x + 1) * self.cell_size - 10)
      self.find_path.epsilon = 0.5
   
   def start(self):
      self.find_path.train(self)

   def test_agent(self):
      self.find_path.test_agent(self)
      
   
       

       
