import random
import numpy as np
from collections import namedtuple,deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        #在这里设置神经网络的具体形状
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, n_actions),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=30,
            memory_size=200,
            batch_size=20,
            e_greedy_increment=None,
    ):
        # 神经网络输入输出参数
        self.n_actions = n_actions # 输出个数
        self.n_features = n_features # 神经网络输入个数

        #学习参数
        self.lr = learning_rate
        self.gamma = reward_decay

        # epsilon参数
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 记忆库初始化和记忆库和参数，记忆库的每个元素为[s, a, r, s_]，代表在状态s采取行动a得到了奖励r和下一个状态s_
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = deque([], maxlen=self.memory_size)


        # 总学习次数
        self.learn_step_counter = 0

        # 建立两个神经网络进行训练
        #torch.set_default_tensor_type(torch.float)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.n_features, self.n_actions).to(self.device).double()
        self.target_net = DQN(self.n_features, self.n_actions).to(self.device).double()


        # 用于存储历史的cost
        self.cost_his = []

        # 优化器
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        # policy每隔这么多次更新就把自身的参数传入target
        self.replace_target_iter = replace_target_iter

        # 记忆库超过batch_size个后再开始学习，一次随机抓batch_size个样本
        self.batch_size = batch_size

    # 将状态存储于记忆库中
    def store_transition(self, *args):
        self.memory.append(self.Transition(*args))
        #transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        #index = self.memory_counter % self.memory_size
        #self.memory[index, :] = transition
        #self.memory_counter += 1

    # 根据当前observation给出action,这里给出一个序号,然后再对应action_space中的动作
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        #observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # 需要将状态输入神经网络然后给出一个n_actions维的向量,取其中最大的元素的index作为返回值
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                observation = torch.from_numpy(observation)
                #observation = observation.double()
                observation = observation.to(torch.float64)

                temp = self.policy_net(observation)
                return np.argmax(temp)
                #return int(self.policy_net(observation).max(1)[1].view(1, 1)[0][0])
        else:
            return np.random.randint(0, self.n_actions)

    # 该函数从memory中抓取batch_size个[state,action,reward,next_state]形式的样本
    # 将batch_size个state输入policy_net，next_state输入target_net
    # 前者返回了一个n_actions维的向量，代表每个action的value,记为Q(s,a)
    # 后者将返回得到的向量中最大的元素取出，组合出期望value = reward+gamma*该元素
    # 收敛时，要求这两个value相等
    def learn(self):
        #一次抓batch_size个记忆库中的样本，不够就先不学
        if len(self.memory) < self.batch_size:
            return

        #policy更新replace_target_iter次后将参数传入target网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('\ntarget_params_replaced\n')

        #在记忆库取出batch_size个记忆样本，每个样本的形式是[state,action,reward,next_state]
        transitions = random.sample(self.memory, self.batch_size)

        # *transitions 将transitions中的每个样本拆开成单元，在将其zip起来,
        # 即 zip(*transitions) 变成[states[],actions[],rewards[],next_state[]]的结构
        # Transition(*zip(*transitions)) 再将这个结构的单元拆出来填入Transition结构
        batch = self.Transition(*zip(*transitions))

        # 取出batch中的每个分量的集合
        state_batch = torch.stack(batch.state)
        next_state_batch = torch.stack(batch.next_state)
        action_batch = np.array(batch.action, dtype=np.int64).reshape(len(batch.action), 1)
        action_batch = torch.from_numpy(action_batch)
        reward_batch = np.array(batch.reward, dtype=np.float64).reshape(len(batch.reward), 1)
        reward_batch = torch.from_numpy(reward_batch)

        # 根据每个state得出每个action的Q(s,a),再挑出这个state实际采取的action的Q(s,a*)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 将每个样本的next_state输入target网络,得出每个action的Q(s,a),再挑取最大的
        next_state_values = self.target_net(next_state_batch).max(1)[0]

        # 计算实际期望值
        expected_state_action_values = (next_state_values.reshape(len(next_state_values), 1)
                                        * self.gamma) + reward_batch

        """
        state_batch = torch.cat(batch.state)
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = torch.cat(batch.next_state)
        # 对state集合中的每个状态返回policy网络计算得到的对应的action_value，并且按action_batch排序？
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 对next_state集合每个状态计算target网络得到的state_value中最大的值,并且detach
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # 计算期望，最终需要梯度下降法使得这两个值相等
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        """

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.cost_his.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 记录policy网络已学习的步数
        self.learn_step_counter += 1

        """
        tensorflow的原代码，上面的pytorch版本应该是实现了相同的功能？
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        """


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()
