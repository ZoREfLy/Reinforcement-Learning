import gym
import numpy as np
from tqdm import tqdm
from Models.DQN import DQN
import matplotlib.pyplot as plt


EPISODES = 5
ACTION_SPACE = 25
env = gym.make('Pendulum-v0')
env.seed(1)

Agent = DQN(env, ACTION_SPACE)


def train():
    rewards = np.array([0])
    for episode in tqdm(range(EPISODES)):
        t = 0
        state = env.reset().reshape([1, 3])
        while t < 5000:
            action = Agent.choose_action(state)
            force = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)

            next_state, reward, done, _ = env.step(np.array([force]))
            next_state = next_state.reshape([1, 3])
            Agent.remember(state[0], action, reward, 0, next_state[0])

            state = next_state

            if episode > 3:
                env.render()
                rewards = np.append(rewards, [reward])

            if t > Agent.memory_size:
                Agent.replay(3000)
            t += 1

    # Agent.save_model('test1')

    plt.figure(1)
    plt.plot(rewards)
    plt.xlabel('training step')
    plt.ylabel('accumulated reward')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    train()
