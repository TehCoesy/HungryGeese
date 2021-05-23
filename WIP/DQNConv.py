import numpy as np
import random
from collections import deque

# Keras imports
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from keras.losses import Huber

# Hungry-Geese env imports
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, GreedyAgent

ACTIONS = ['NORTH', 'SOUTH', 'WEST', 'EAST']

class DQN:
    def __init__(self, config):
        # Environment
        self.config = config
        self.observation_space = self.config.rows * self.config.columns
        self.action_space = 4
        self.last_action = 0
        # Training parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.01
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        # Model
        self.memory = deque(maxlen=10000)
        self.model = self.createModel()

    def createModel(self):
        model = Sequential()

        # Conv2D
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(7, 11, 3)))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(Flatten())
        # Dense
        model.add(Dense(128,activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        # Output Q-Values of 4 available actions
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.summary()
        return model

    def saveModel(self):
        self.model.save("./Weights/DQNConv_Model_3.h5")

    # Training functions
    def epsilonDecay(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode)

    def resetEpsilon(self):
        self.epsilon = 1

    def addMemory(self, obs, action, reward, new_obs, done):
        encoded_obs = self.createInput(obs)
        new_encoded_obs = self.createInput(new_obs)
        self.memory.append([encoded_obs, action, reward, new_encoded_obs, done])

    def train(self):
        # Training parameters
        batch_size = 512

        # Calculate Q for states
        if len(self.memory) < batch_size:
            return

        x = []
        y = []

        mini_batch = random.sample(self.memory, batch_size)

        for obs, action, reward, new_obs, done in mini_batch:
            x.append(obs)
            max_q = reward
            q = [0]*4
            if not done:
                new_q = self.model.predict(np.array([new_obs]))[0]
                max_q += self.discount_factor * np.max(new_q)

            q[action] = (1 - self.learning_rate) * q[action] + self.learning_rate * max_q
            y.append(q)

        #print(np.array(x).shape)
        #print(np.array(y).shape)

        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0)
        #print("Training done")

    # Input / Output
    def createInput(self, obs):
        # Convert agent's observation to my format
        food = obs['food']
        geese = obs['geese']
        player = obs['index']
        player_goose = obs['geese'][player]

        # Geese
        geese_one_hot = np.zeros(self.config.rows * self.config.columns)
        for g in np.concatenate(geese):
            geese_one_hot[int(g)] = 1

        #geese_one_hot = geese_one_hot.reshape(-1, 7, 11)

        # Player's head
        player_one_hot = np.zeros(self.config.rows * self.config.columns)
        if (len(player_goose) > 0):
            player_head = player_goose[0]
            player_one_hot[player_head] = 1

        #player_one_hot = player_one_hot.reshape(-1, 7, 11)

        # Food
        food_one_hot = np.zeros(self.config.rows * self.config.columns)
        for f in food:
            food_one_hot[int(f)] = 1

        #food_one_hot = food_one_hot.reshape(-1, 7, 11)

        #input = np.vstack((geese_one_hot, player_one_hot, food_one_hot))
        #input = input.reshape(7, 11, 3)

        input = np.zeros((7, 11, 3))
        for i in range(7):
            for k in range(11):
                if geese_one_hot[i * 11 + k] == 1:
                    input[i][k][0] = 1
                if player_one_hot[i * 11 + k] == 1:
                    input[i][k][1] = 1
                if food_one_hot[i * 11 + k] == 1:
                    input[i][k][2] = 1
        return input

    def getAction(self, obs):
        encoded_obs = self.createInput(obs)
        # Exploration (Greedy strategy)
        if np.random.rand() <= self.epsilon:
            #return np.random.choice(self.action_space)
            g_agent = GreedyAgent(Configuration(self.config))
            action = g_agent(Observation(obs))
            return ACTIONS.index(action)
        # Exploitation
        q = self.model.predict(np.array([encoded_obs]))
        return np.argmax(q[0])

if __name__=="__main__":
    env = make("hungry_geese")
    # Train with 3 Greedy agents
    trainer = env.train([None, "greedy", "greedy", "greedy"])

    # Training configuration
    episodes = 1000
    training_steps = 40

    # Training stuff
    config = env.configuration
    agent = DQN(config)
    
    for ep in range(episodes):
        obs = trainer.reset()
        done = False

        episode_reward = 0 # Culmulative episode reward

        while not done:
            action_int = agent.getAction(obs)
            action = ACTIONS[action_int]
            new_obs, reward, done, info = trainer.step(action)

            agent.addMemory(obs, action_int, reward, new_obs, done)
            obs = new_obs

            if done and reward == 0:
                reward = -100

            episode_reward += reward

            if done and ep % 5 == 0:
                print("Episode: ", ep, " with total reward ", episode_reward, " in ", obs['step'], " steps")
                agent.train()

        agent.epsilonDecay(ep)

    agent.saveModel()