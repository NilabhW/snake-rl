import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')

BLOCK  = 20   # pixel size of each grid cell
SPEED  = 40   # FPS during training (set higher = faster training)
WHITE  = (255, 255, 255)
RED    = (200, 0, 0)
GREEN1 = (0, 200, 100)
GREEN2 = (0, 150, 70)
BLACK  = (0, 0, 0)

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Called at the start of every episode."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK, self.head.y),
            Point(self.head.x - 2*BLOCK, self.head.y),
        ]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()  # retry if food lands on snake

    def play_step(self, action):
        """
        Takes an action, advances one frame.
        Returns: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # 1. Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            # Timeout prevents infinite loops when snake is lost
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # remove tail if no food eaten

        # 5. Render & return
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hit wall
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Hit self
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        """
        action = [1,0,0] straight
                 [0,1,0] right turn
                 [0,0,1] left turn
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]            # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # left turn

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK
        elif self.direction == Direction.LEFT:  x -= BLOCK
        elif self.direction == Direction.DOWN:  y += BLOCK
        elif self.direction == Direction.UP:    y -= BLOCK
        self.head = Point(x, y)

    def get_state(self):
        """Returns the 11-value state vector."""
        head = self.head
        pt_l = Point(head.x - BLOCK, head.y)
        pt_r = Point(head.x + BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(pt_r)) or
            (dir_l and self.is_collision(pt_l)) or
            (dir_u and self.is_collision(pt_u)) or
            (dir_d and self.is_collision(pt_d)),

            # Danger right
            (dir_u and self.is_collision(pt_r)) or
            (dir_d and self.is_collision(pt_l)) or
            (dir_l and self.is_collision(pt_u)) or
            (dir_r and self.is_collision(pt_d)),

            # Danger left
            (dir_d and self.is_collision(pt_r)) or
            (dir_u and self.is_collision(pt_l)) or
            (dir_r and self.is_collision(pt_u)) or
            (dir_l and self.is_collision(pt_d)),

            # Current direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def _update_ui(self):
        self.display.fill(BLACK)
        for i, pt in enumerate(self.snake):
            color = GREEN1 if i == 0 else GREEN2
            pygame.draw.rect(self.display, color,
                             pygame.Rect(pt.x, pt.y, BLOCK, BLOCK))
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK, BLOCK))
        font = pygame.font.SysFont('arial', 18)
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()