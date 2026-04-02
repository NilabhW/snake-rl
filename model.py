import torch
import torch.nn as nn
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    def save(self, filename='model.pth'):
        folder = './model'
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, filename))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state      = torch.tensor(state,      dtype=torch.float)
        action     = torch.tensor(action,     dtype=torch.long)
        reward     = torch.tensor(reward,     dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Handle single sample (1D) by adding batch dimension
        if state.dim() == 1:
            state      = state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            done       = (done, )

        # Current Q-values from the model
        pred = self.model(state)

        # Bellman update: Q_new = r + γ * max(Q(s'))
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()