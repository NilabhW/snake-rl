import numpy as np
import random
from collections import deque
from game import SnakeGame
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000  # how many experiences to store
BATCH_SIZE = 1000     # how many to sample per training step
LR         = 0.001    # learning rate

class Agent:
    def __init__(self):
        self.n_games  = 0
        self.epsilon  = 0      # exploration rate (filled dynamically)
        self.gamma    = 0.9    # discount factor
        self.memory   = deque(maxlen=MAX_MEMORY)  # auto-drops oldest when full

        self.model   = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        """Store one experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Sample a batch from memory and train — called after every game."""
        if len(self.memory) < BATCH_SIZE:
            sample = list(self.memory)
        else:
            sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on the single most recent experience — called every step."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Epsilon-greedy: explore randomly at first,
        exploit the model more as training progresses.
        """
        self.epsilon = max(10, 80 - self.n_games)
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Random action (exploration)
            idx = random.randint(0, 2)
        else:
            # Model's best action (exploitation)
            state_t = __import__('torch').tensor(state, dtype=__import__('torch').float)
            prediction = self.model(state_t)
            idx = int(prediction.argmax().item())

        action[idx] = 1
        return action