import matplotlib
matplotlib.use('Agg')   # works reliably on macOS
import matplotlib.pyplot as plt
from agent import Agent
from game import SnakeGame



def plot(scores, mean_scores):
    plt.clf()
    plt.title('Snake RL — Training Progress')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores,      label='Score',      alpha=0.4)
    plt.plot(mean_scores, label='Mean score', linewidth=2)
    plt.legend()
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1],  str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1],
                 f'{mean_scores[-1]:.1f}')
    plt.savefig('progress.png', dpi=100)

def train():
    scores      = []
    mean_scores = []
    total_score = 0
    best_score  = 0

    agent = Agent()
    game  = SnakeGame()

    print('Starting training — close the pygame window to stop.')
    print(f'{"Game":>6}  {"Score":>6}  {"Best":>6}  {"Mean":>8}')
    print('-' * 35)

    while True:
        # 1. Get current state
        state_old = game.get_state()

        # 2. Choose action
        action = agent.get_action(state_old)

        # 3. Perform action, get new state
        reward, done, score = game.play_step(action)
        state_new = game.get_state()

        # 4. Train short memory (one step)
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # 5. Store experience
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # 6. End of game — train on full replay buffer
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # 7. Save model if new best score
            if score > best_score:
                best_score = score
                agent.model.save()

            # 8. Track and plot progress
            total_score += score
            mean_score   = total_score / agent.n_games
            scores.append(score)
            mean_scores.append(mean_score)

            print(f'{agent.n_games:>6}  {score:>6}  {best_score:>6}  {mean_score:>8.2f}')
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()