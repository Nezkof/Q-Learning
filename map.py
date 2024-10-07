import tkinter as tk
import random
from agent import Agent

class Map:
   def __init__(self, master, n, custom_matrix=None):
        self.master = master
        self.n = n
        self.cell_size = 50

        self.grid_frame = tk.Frame(master)
        self.grid_frame.pack()

        self.canvas = tk.Canvas(self.grid_frame, width=n * self.cell_size, height=n * self.cell_size)
        self.canvas.pack()

        self.matrix = custom_matrix if custom_matrix is not None else self.initialize_matrix()
        self.draw_grid()

        self.agent = self.create_agent()


        self.create_buttons()

   def initialize_matrix(self):
        return [[random.choice([0, 1]) for _ in range(self.n)] for _ in range(self.n)]

   def draw_grid(self):
        for i in range(self.n):
            for j in range(self.n):
               if self.matrix[i][j] == 1:
                  color = '#31363F'
               elif self.matrix[i][j] == -1:
                  color = '#454a51'
               else:
                  color = '#222831'

               self.canvas.create_rectangle(j * self.cell_size, i * self.cell_size,
                                             (j + 1) * self.cell_size, (i + 1) * self.cell_size,
                                             fill=color, outline='#2d323b')

   def create_agent(self):
      return Agent(self.canvas, 0, 0, self.cell_size, self.matrix) 

   def create_buttons(self):
      self.button_frame = tk.Frame(self.master, bg='#2d323b')
      self.button_frame.pack(pady=20)

      btn1 = tk.Button(self.button_frame, text="Start", command=self.start_agent, bg='#2d323b', fg='white')
      btn2 = tk.Button(self.button_frame, text="Reset position", command=self.reset_agent_pos, bg='#2d323b', fg='white')
      btn3 = tk.Button(self.button_frame, text="Test", command=self.test_agent, bg='#2d323b', fg='white')

      btn1.pack(side=tk.LEFT, padx=10)
      btn2.pack(side=tk.LEFT, padx=10)
      btn3.pack(side=tk.LEFT, padx=10)

   def start_agent(self):
      self.agent.start()

   def reset_agent_pos(self):
      self.agent.reset(0,0)

   def test_agent(self):
      self.agent.test_agent()
