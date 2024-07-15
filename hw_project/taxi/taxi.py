import gym
from collections import deque
import sys
from collections import defaultdict
import numpy as np
from agent import Agent

env = gym.make('Taxi-v3')

action_size = env.action_space.n
print("Action Space", env.action_space.n)
print("State Space", env.observation_space.n)


def model_free_RL(Q, mode):
    agent = Agent(Q, mode)
    num_episodes = 100000
    last_100_episode_rewards = deque(maxlen=100)
    for i_episode in range(1, num_episodes+1):

        state = env.reset()
        episode_rewards = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            episode_rewards += reward
            if done:
                last_100_episode_rewards.append(episode_rewards)
                break

            state = next_state

        if (i_episode >= 100):
            last_100_episode_rewards.append(episode_rewards)
            avg_reward = sum(last_100_episode_rewards) / len(last_100_episode_rewards)
            print("\rEpisode {}/{} || Best average reward {}\n".format(i_episode, num_episodes, avg_reward), end="")

    print()


def testing_after_learning(Q, mode):
    agent = Agent(Q, mode)
    n_tests = 100
    total_test_rewards = []
    for episode in range(n_tests):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                total_test_rewards.append(episode_reward)
                break

            state = new_state

    print("avg: " + str(sum(total_test_rewards) / n_tests))


Q = defaultdict(lambda: np.zeros(action_size))

while True:
    print("1. MC-control")
    print("2. Q-learning")
    print("3. Testing after learning")
    print("4. Exit")
    menu = int(input("select: "))    
    
    if menu == 1:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "mc_control")
    elif menu == 2:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "q_learning")
    elif menu == 3:
        testing_after_learning(Q, "test_mode")
    elif menu == 4:
        break
    else:
        print("wrong input!")
