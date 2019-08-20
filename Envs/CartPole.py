import gym
from Models.DQN import DQN


EPISODES = 3500
ACTION_SPACE = 2

env = gym.make('CartPole-v0')
env.seed(1)

Agent = DQN(env, ACTION_SPACE, replace_itr=10, memory_size=1000)


def train():
    for episode in range(EPISODES):
        state = env.reset().reshape([1, 4])
        for t in range(500):
            action = Agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, 4])

            Agent.remember(state[0], action, reward, done, next_state[0])

            state = next_state

            if episode > 3490:
                env.render()

            if done:
                print('Episode: {}/{}, Score: {}'.format(episode, EPISODES, t))
                break

        Agent.replay(32)


if __name__ == '__main__':
    train()
