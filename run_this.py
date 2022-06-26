import math

import torch

from field_env import Flow_Field
from RL_brain import DeepQNetwork
import numpy as np
#import random

MAX_EPISODE = 4


def start_swim():
    action_space = ['plus5', 'plus10', 'minus5', 'minus10', 'zero']
    action_his = open('action_history.txt', 'w+')
    initial_k = 20363359375.000004
    with open('total_reward.txt', 'w+') as f:
        for episode in range(MAX_EPISODE):
            # 初始物体状态
            vertex = open('swimmer.vertex', 'r')
            n = int(vertex.readline())
            Lag = []
            for i in range(n):
                tmp = vertex.readline().split()
                Lag.append([float(tmp[0]), float(tmp[1])])
            Lag = np.array(Lag, dtype=np.float64)
            x, y = np.mean(Lag, axis=0)
            observation = [x, y, 0, 0, math.log(initial_k)]
            observation = np.array(observation)
            vertex.close()
            # 初始环境
            env = Flow_Field()
            env.episode = episode
            done = False
            step = 0 #记录步数d
            while not done:
                # RL choose action based on observation
                action = RL.choose_action(observation)
                action = int(action)
                print('========action = ', action_space[action], '==========\n')
                action_his.write('episode =' + str(episode) + ', action= ' + action_space[action] + '\n')
                # action = random.choice(env.action_space)
                # print('========action = ', action, '==========\n')


                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action_space[action])
                #observation_, reward, done = env.step(action)

                env.total_reward += reward

                #RL.store_transition(torch.from_numpy(observation), torch.tensor([action]), torch.tensor([reward]), torch.from_numpy(observation_))
                RL.store_transition(torch.from_numpy(observation), action, reward, torch.from_numpy(observation_))

                if (step > 5) and (step % 5 == 0):
                    RL.learn()
                #RL.learn()
                # swap observation
                observation = observation_

                step += 1
            print('=========================Episode ', episode, ' Total Reward = ', env.total_reward, '================================\n')
            f.write('Episode ' + str(episode) + ' Total Reward = ' + str(env.total_reward) + '\n')
            action_his.write('\n')
    # 训练结束
    print('train finished!')
    action_his.close()
    #print(RL.memory)
    # print(RL.policy_net.state_dict())
    # env.destroy()


if __name__ == "__main__":
    # 强化学习推进策略
    n_actions = 5
    n_features = 5
    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=200,
                      )
    start_swim()
    RL.plot_cost()
"""
RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )
                      
RL.plot_cost()
"""