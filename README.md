# 🐍 Snake RL — Deep Q-Learning Agent

A Snake game AI trained from scratch using **Deep Q-Network (DQN)** reinforcement learning. The agent learns entirely through trial and error — no human gameplay data, no hard-coded rules — and progressively improves its score over thousands of games.

---

## 📺 Demo

> After training, the agent navigates the board, avoids walls and its own body, and efficiently hunts food.

Training progress is saved automatically to `progress.png` after each game:

![Training Progress](progress.png)

---

## 🧠 How It Works

This project implements the classic **DQN (Deep Q-Network)** algorithm:

```
State (11 values) ──► Neural Network ──► Q-values for 3 actions ──► Best action
```

### State Representation (11 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | Danger straight | Is there a wall/body directly ahead? |
| 2 | Danger right | Is there danger to the right? |
| 3 | Danger left | Is there danger to the left? |
| 4–7 | Current direction | One-hot: LEFT / RIGHT / UP / DOWN |
| 8 | Food left | Is the food to the left of the head? |
| 9 | Food right | Is the food to the right of the head? |
| 10 | Food up | Is the food above the head? |
| 11 | Food down | Is the food below the head? |

### Action Space (3 actions)
| Action | Meaning |
|--------|---------|
| `[1, 0, 0]` | Go straight |
| `[0, 1, 0]` | Turn right |
| `[0, 0, 1]` | Turn left |

Actions are **relative to the current direction**, preventing the snake from reversing into itself.

### Reward Structure
| Event | Reward |
|-------|--------|
| Eating food | **+10** |
| Dying (wall / self) | **−10** |
| Timeout (stuck in loop) | **−10** |

### Neural Network Architecture
```
Input (11)  →  Linear → ReLU  →  Hidden (256)  →  Linear → ReLU  →  Hidden (256)  →  Linear  →  Output (3)
```
A two-hidden-layer **MLP** with ReLU activations, trained with the **Adam** optimizer and **MSE loss**.

### Training Loop (per step)
1. Observe state `s`
2. Choose action via **ε-greedy** (random early on, model-driven later)
3. Execute action, receive reward `r`, observe next state `s'`
4. Store `(s, a, r, s', done)` in **replay buffer** (capacity: 100,000)
5. **Short-memory training**: train on this single transition immediately
6. On game over → **Long-memory training**: sample a random batch of 1,000 from the buffer and train (experience replay)
7. Save model if a new best score is achieved

### Bellman Update (Q-learning target)
```
Q_new = r + γ · max(Q(s'))    if not done
Q_new = r                      if done
```
Discount factor **γ = 0.9**, learning rate **lr = 0.001**.

### Exploration vs. Exploitation
```python
epsilon = max(10, 80 - n_games)
```
The agent explores randomly ~40% of the time during the first ~70 games, then increasingly relies on the learned model.

---

## 📁 Project Structure

```
snake-rl/
├── game.py        # SnakeGame environment (Pygame)
├── agent.py       # DQN Agent (memory, training, action selection)
├── model.py       # Neural network (Linear_QNet) & QTrainer (Bellman update)
├── train.py       # Main training loop + live plotting
├── reset.py       # Utility: deletes saved model & plot for a fresh run
├── model/
│   └── model.pth  # Best model checkpoint (auto-saved, git-ignored)
└── progress.png   # Training plot (auto-generated, git-ignored)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repo
```bash
git clone https://github.com/NilabhW/snake-rl.git
cd snake-rl
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install torch pygame numpy matplotlib
```

### 4. Start training
```bash
python train.py
```

A **Pygame window** will open showing the snake playing in real time.  
Training statistics are printed to the console and `progress.png` is updated after every game.

### 5. Reset and start fresh (optional)
```bash
python reset.py
```
This deletes `model/model.pth` and `progress.png` so you can start training from scratch.

---

## 📊 Console Output

```
  Game   Score    Best      Mean
-----------------------------------
     1       0       0      0.00
     2       1       1      0.50
     3       0       1      0.33
    ...
   150      24      24     11.47
```

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Max replay memory | 100,000 | `agent.py` |
| Batch size | 1,000 | `agent.py` |
| Learning rate | 0.001 | `agent.py` |
| Discount factor (γ) | 0.9 | `agent.py` |
| Hidden layer size | 256 | `agent.py` |
| Game speed (FPS) | 40 | `game.py` |
| Grid block size | 20 px | `game.py` |
| Window size | 640×480 | `game.py` |

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Neural network & backpropagation |
| `pygame` | Snake game environment & rendering |
| `numpy` | State vector computation |
| `matplotlib` | Training progress plots |

---

## 💡 Ideas for Improvement

- [ ] Add a **target network** for more stable training
- [ ] Tune epsilon decay schedule
- [ ] Experiment with **deeper / wider networks**
- [ ] Add **convolutional layers** using a pixel-based state
- [ ] Implement **Double DQN** or **Dueling DQN**
- [ ] Add a headless / no-render training mode for speed

---

## 📄 License

MIT — feel free to fork, experiment, and improve.
