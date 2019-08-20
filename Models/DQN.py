import keras
import random
import numpy as np

from pdb import set_trace as bp
from keras import layers
from keras import optimizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQN:
    """Deep Q Network"""
    def __init__(self, env, action_space, epsilon=1.0, epsilon_min=0.01, memory_size=3000, model_name=None):
        self.gamma = 0.95
        self.epsilon_min = epsilon_min

        self.epsilon = epsilon
        self.epsilon_decay = 0.993
        self.learning_rate = 0.001

        self.action_space = action_space
        self.observation_space = env.observation_space.shape[0]

        self.memory_counter = 0
        self._memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.observation_space * 2 + 3))

        if not model_name:
            self.model = self.build_model()
        else:
            self.load_model(model_name)

    @property
    def memory_size(self):
        return self._memory_size

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.observation_space,), activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def remember(self, state, action, reward, done, next_state):
        index = self.memory_counter % self._memory_size
        transition = np.hstack((state, [action, reward], done, next_state))
        self.memory[index, :] = transition
        self.memory_counter += 1

    def replay(self, batch_size):
        random_index = np.random.choice(self._memory_size, size=batch_size)
        samples = self.memory[random_index, :]

        index = self.observation_space
        states = samples[:, :index]
        actions = samples[:, index]
        rewards = samples[:, index+1]
        next_states = samples[:, index+3:]

        targets = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)[:]
        target_f = self.model.predict(states)

        batch_index = np.arange(batch_size, dtype=np.int32)
        actions_index = actions.astype(int)
        target_f[batch_index, actions_index] = targets

        done_mask = samples[:, index+2] == True
        actions_mask = actions_index[done_mask]
        target_f[done_mask, actions_mask] = rewards[done_mask]

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = keras.models.load_model(name)
