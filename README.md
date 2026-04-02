cat > README.md << EOF
# Snake RL

A Deep Q-Network (DQN) agent that learns to play Snake from scratch using reinforcement learning.

## Setup
\`\`\`
python3 -m venv venv
source venv/bin/activate
pip install pygame torch numpy matplotlib
\`\`\`

## Train
\`\`\`
python3 train.py
\`\`\`

## Stack
- Python 3.13
- PyTorch
- Pygame
- DQN with experience replay
EOF

git add README.md
git commit -m "Add README"
git push
