import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt

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
        self.discount_factor = 0.4
        self.epsilon = 1
        self.epsilon_decay = 0.01
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        # Model
        self.memory = deque(maxlen=10000)
        self.model = self.createModel()
        
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())

    def createModel(self):
        model = Sequential()

        # Conv2D
        model.add(Input(shape=(7,11,17)))
        model.add(Conv2D(filters=128, kernel_size=(3, 5), activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(Flatten())
        # Dense
        model.add(Dense(128,activation='relu'))
        model.add(Dense(32, activation='relu'))

        # Output Q-Values of 4 available actions
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(), metrics=['accuracy'])
        model.summary()
        return model

    def saveModel(self):
        self.model.save("./Weights/DDQNConv_Model_10.h5")

    def plot(self, arr):
        fig = plt.figure()
        plt.plot(arr)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.show()
        fig.savefig('DDQNConv_LR_10_DF_40.jpg')

    def save_rewards(self, arr):
        file = open("./DDQN_Reward_LR_10_DF_40.txt", "w+")
        for reward in arr:
            file.write(str(reward)+ '\n')
        file.close()

    # Training functions
    def epsilonDecay(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * episode)

    def resetEpsilon(self):
        self.epsilon = 1

    def addMemory(self, obs, action, reward, new_obs, done):
        self.memory.append([encoded_obs, action, reward, new_encoded_obs, done])
    
    def train(self):
        # Training parameters
        batch_size = 1024

        # Calculate Q for states
        if len(self.memory) < 3072:
            return

        mini_batch = random.sample(self.memory, batch_size)
        
        #current_states = np.array([transition[0] for transition in mini_batch])
        #current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            #current_qs = current_qs_list[index]
            current_qs = [0]*4
            current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)

        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def updateTarget(self):
        self.target_model.set_weights(self.model.get_weights())

    # Input / Output
    def centerize(self, b):
        dy, dx = np.where(b[0])
        centerize_y = (np.arange(0,7)-3+dy[0])%7
        centerize_x = (np.arange(0,11)-5+dx[0])%11
        
        b = b[:, centerize_y,:]
        b = b[:, :,centerize_x]
        
        return b
        
    def createInput(self, obses):
        b = np.zeros((17, 7 * 11), dtype=np.float32)
        obs = obses[-1]

        for p, pos_list in enumerate(obs['geese']):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - obs['index']) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - obs['index']) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - obs['index']) % 4, pos] = 1
                
        # previous head position
        if len(obses) > 1:
            obs_prev = obses[-2]
            for p, pos_list in enumerate(obs_prev['geese']):
                for pos in pos_list[:1]:
                    b[12 + (p - obs['index']) % 4, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1
            
        b = b.reshape(-1, 7, 11)
        b = self.centerize(b)
        b = np.transpose(b, (1,2,0))

        return b

    def getAction(self, obs):
        # Exploration (Greedy strategy)
        if np.random.rand() <= self.epsilon or len(self.memory) < 5000:
            #return np.random.choice(self.action_space)
            g_agent = GreedyAgent(Configuration(self.config))
            action = g_agent(Observation(obs[-1]))
            return ACTIONS.index(action)
        # Exploitation
        encoded_obs = self.createInput(obs)
        encoded_reshaped = encoded_obs.reshape(-1, 7, 11, 17)
        q = self.model.predict(encoded_reshaped)
        return np.argmax(q)

if __name__=="__main__":
    env = make("hungry_geese")
    # Train with 3 Greedy agents
    trainer = env.train([None, "greedy", "greedy", "greedy"])

    # Training configuration
    episodes = 1000

    # Training stuff
    config = env.configuration
    agent = DQN(config)
    
    update_step = 0

    obs_list = []
    total_rewards = []

    for ep in range(episodes):
        obs = trainer.reset()
        obs_list = []
        episode_reward = 0 # Culmulative episode reward

        done = False

        while not done:
            obs_list.append(obs)
            encoded_obs = agent.createInput(obs_list)

            update_step += 1

            action_int = agent.getAction(obs_list)
            action = ACTIONS[action_int]
            new_obs, reward, done, info = trainer.step(action)

            if done and reward == 0:
                reward = -1000

            new_encoded_obs = agent.createInput(obs_list)
            agent.addMemory(encoded_obs, action_int, reward, new_encoded_obs, done)

            if update_step % 4 == 0 or done:
                agent.train()

            obs = new_obs
            episode_reward += reward

            if done:
                print("Episode: ", ep, " with total reward ", episode_reward, " in ", obs['step'], " steps")
                total_rewards.append(episode_reward)

                if update_step >= 100:
                    print('Copying main network weights to the target network weights')
                    update_step = 0
                    agent.updateTarget()
                break

        agent.epsilonDecay(ep)

    agent.saveModel()
    agent.plot(total_rewards)
    agent.save_rewards(total_rewards)