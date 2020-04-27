"""
Utilities for training a DQN
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import gym
import numpy as np

from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

# replay buffer from and some code based on https://github.com/sfujim/TD3
# create replay buffer of tuples of (state, next_state, action, reward, done)
class ReplayBuffer():
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        # print("testing sample shapes")
        # print(np.shape(x))
        # print("after x....")
        # print(np.shape(y))
        # print(np.shape(u))
        # print(np.shape(r))
        # print(np.shape(d))
        return np.array(x), np.array(y), np.array(u).reshape(-1,1), np.array(r).reshape(-1,1), np.array(d).reshape(-1,1)


# create Deep Q Network Class by inheriting from torch.nn.Module
# based on Nature CNN from OpenAI baselines: https://github.com/openai/baselines/blob/1b092434fc51efcb25d6650e287f07634ada1e08/baselines/common/models.py
class DeepQNetwork(nn.Module):
    def __init__(self, action_size, hidden_size):
        super(DeepQNetwork, self).__init__()
        #self.conv_layer_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_layer_1 = nn.Conv2d(1, 32, kernel_size=8, stride=4) #using black and white and frame velocity, only 1 channel needed
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense_layer = nn.Linear(1024, hidden_size)
        self.out_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x / 255.  # image data is stored as ints in 0 to 255 range. Divide to scale to 0 to 1 range
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = F.relu(self.dense_layer(x.view(32, -1)))
        return self.out_layer(x)

class DuelingDQN(nn.Module):
    def __init__(self, action_size, hidden_size):
        super(DuelingDQN, self).__init__()
        #self.conv_layer_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_layer_1 = nn.Conv2d(1, 32, kernel_size=8, stride=4) #using black and white and frame velocity, only 1 channel needed
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # V(s) value of the state
        self.dueling_value_1 = nn.Linear(1024, hidden_size)
        self.dueling_value_2 = nn.Linear(hidden_size, 1)
        # Q(s,a) Q values of the state-action combination
        self.dueling_action_1 = nn.Linear(1024, hidden_size)
        self.dueling_action_2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x / 255.  # image data is stored as ints in 0 to 255 range. Divide to scale to 0 to 1 range
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        # advantage from action net
        advantage = F.relu(self.dueling_action_1(x))
        advantage = self.dueling_action_2(advantage)
        # value from value function net
        value = F.relu(self.dueling_value_1(x))
        value = self.dueling_value_2(value)

        x = advantage - advantage.mean(dim=1, keepdim=True) + value
        return x


class IMPALANet(nn.Module):
    # Based on IMPALA Net from IMPALA paper and IMPALA Net implementations on CoinRun env and Facebook IMPALA Net
    # https://github.com/openai/coinrun
    # https://github.com/facebookresearch/torchbeast
    def __init__(self, action_size, hidden_size, input_channels, dropout_p=0.1, is_large_net=False):
        super(IMPALANet, self).__init__()
        self.is_large_net = is_large_net
        self.dropout = nn.Dropout(p=0.1)

        if is_large_net:
            depth_list = [32, 64, 64, 64, 64]  # larger CoinRun net size
        else:
            depth_list = [16, 32, 32]  # original impala net size

        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=depth_list[0], kernel_size=3,
                                stride=1, padding=1)
        num_ch = depth_list[0]
        self.bn_1a = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_conv_1_a1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_1b = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_1_a2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_1c = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_1_b1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_1d = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_1_b2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_1e = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)

        self.conv_2 = nn.Conv2d(in_channels=depth_list[0], out_channels=depth_list[1], kernel_size=3,
                                stride=1, padding=1)
        num_ch = depth_list[1]
        self.bn_2a = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_conv_2_a1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_2b = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_2_a2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_2c = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_2_b1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_2d = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_2_b2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_2e = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)

        self.conv_3 = nn.Conv2d(in_channels=depth_list[1], out_channels=depth_list[2], kernel_size=3,
                                stride=1, padding=1)
        num_ch = depth_list[2]
        self.bn_3a = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_conv_3_a1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_3b = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_3_a2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_3c = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_3_b1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_3d = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        self.res_conv_3_b2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                       stride=1, padding=1)
        self.bn_3e = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)

        if is_large_net:
            self.conv_4 = nn.Conv2d(in_channels=depth_list[2], out_channels=depth_list[3], kernel_size=3,
                                    stride=1, padding=1)
            num_ch = depth_list[3]
            self.bn_4a = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.res_conv_4_a1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_4b = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_4_a2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_4c = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_4_b1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_4d = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_4_b2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_4e = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)

            self.conv_5 = nn.Conv2d(in_channels=depth_list[3], out_channels=depth_list[4], kernel_size=3,
                                    stride=1, padding=1)
            num_ch = depth_list[4]
            self.bn_5a = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.res_conv_5_a1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_5b = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_5_a2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_5c = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_5_b1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_5d = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
            self.res_conv_5_b2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3,
                                           stride=1, padding=1)
            self.bn_5e = nn.BatchNorm2d(num_ch, momentum=0.001, eps=0.001, affine=True)
        # uses dueling architecture
        # V(s) value of the state
        self.dueling_value_1 = nn.Linear(2048, hidden_size)
        self.dueling_value_2 = nn.Linear(hidden_size, 1)
        # Q(s,a) Q values of the state-action combination
        self.dueling_action_1 = nn.Linear(2048, hidden_size)
        self.dueling_action_2 = nn.Linear(hidden_size, action_size)


    def forward(self, x):
        x = x / 255.  # image data is stored as ints in 0 to 255 range. Divide to scale to 0 to 1 range
        # 3 Blocks: conv, pool, res block 1, res block 2
        # at start of each res block, split paths:
        # path 1 does relu, conv, relu, conv
        # path 2 goes straight to output
        # output = path_1_output + path_2_output
        # dropout after every conv
        # different batch norm layer after every dropout
        # Block 1
        x = self.bn_1a(self.dropout(self.conv_1(x)))
        x = self.pool_1(x)
        res_x = F.relu(x)  # res block 1
        res_x = F.relu(self.bn_1b(self.dropout(self.res_conv_1_a1(res_x))))
        res_x = self.bn_1c(self.dropout(self.res_conv_1_a2(res_x)))
        x = res_x + x
        res_x = F.relu(x)  # res block 2
        res_x = F.relu(self.bn_1d(self.dropout(self.res_conv_1_b1(res_x))))
        res_x = self.bn_1e(self.dropout(self.res_conv_1_b2(res_x)))
        x = res_x + x
        # Block 2
        x = self.bn_2a(self.dropout(self.conv_2(x)))
        x = self.pool_2(x)
        res_x = F.relu(x)  # res block 1
        res_x = F.relu(self.bn_2b(self.dropout(self.res_conv_2_a1(res_x))))
        res_x = self.bn_2c(self.dropout(self.res_conv_2_a2(res_x)))
        x = res_x + x
        res_x = F.relu(x)  # res block 2
        res_x = F.relu(self.bn_2d(self.dropout(self.res_conv_2_b1(res_x))))
        res_x = self.bn_2e(self.dropout(self.res_conv_2_b2(res_x)))
        x = res_x + x
        # Block 3
        x = self.bn_3a(self.dropout(self.conv_3(x)))
        x = self.pool_3(x)
        res_x = F.relu(x)  # res block 1
        res_x = F.relu(self.bn_3b(self.dropout(self.res_conv_3_a1(res_x))))
        res_x = self.bn_3c(self.dropout(self.res_conv_3_a2(res_x)))
        x = res_x + x
        res_x = F.relu(x)  # res block 2
        res_x = F.relu(self.bn_3d(self.dropout(self.res_conv_3_b1(res_x))))
        res_x = self.bn_3e(self.dropout(self.res_conv_3_b2(res_x)))
        x = res_x + x
        # large Impala Net does 2 more blocks
        if self.is_large_net:
            # Block 4
            x = self.bn_4a(self.dropout(self.conv_4(x)))
            x = self.pool_4(x)
            res_x = F.relu(x)  # res block 1
            res_x = F.relu(self.bn_4b(self.dropout(self.res_conv_4_a1(res_x))))
            res_x = self.bn_4c(self.dropout(self.res_conv_4_a2(res_x)))
            x = res_x + x
            res_x = F.relu(x)  # res block 2
            res_x = F.relu(self.bn_4d(self.dropout(self.res_conv_4_b1(res_x))))
            res_x = self.bn_4e(self.dropout(self.res_conv_4_b2(res_x)))
            x = res_x + x
            # Block 5
            x = self.bn_5a(self.dropout(self.conv_5(x)))
            x = self.pool_5(x)
            res_x = F.relu(x)  # res block 1
            res_x = F.relu(self.bn_5b(self.dropout(self.res_conv_5_a1(res_x))))
            res_x = self.bn_5c(self.dropout(self.res_conv_5_a2(res_x)))
            x = res_x + x
            res_x = F.relu(x)  # res block 2
            res_x = F.relu(self.bn_5d(self.dropout(self.res_conv_5_b1(res_x))))
            res_x = self.bn_5e(self.dropout(self.res_conv_5_b2(res_x)))
            x = res_x + x

        x = x.view(32, -1)

        # dueling architecture
        # get advantage from advantage net
        advantage = F.relu(self.dueling_action_1(x))
        advantage = self.dueling_action_2(advantage)
        # value from value function net
        value = F.relu(self.dueling_value_1(x))
        value = self.dueling_value_2(value)
        x = advantage - advantage.mean(dim=1, keepdim=True) + value
        return x


class DQNAgent():
    def __init__(self, action_size, hidden_size, learning_rate, is_dueling=True, is_impala_net=True):
        # check and use GPU if available if not use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        if is_dueling:
            if is_impala_net:
                self.train_net = IMPALANet(action_size, hidden_size, 1).to(self.device)
                self.target_net = IMPALANet(action_size, hidden_size, 1).to(self.device)
            else:
                self.train_net = DeepQNetwork(action_size, hidden_size).to(self.device)
                self.target_net = DeepQNetwork(action_size, hidden_size).to(self.device)
        else:
            self.train_net = DuelingDQN(action_size, hidden_size).to(self.device)
            self.target_net = DuelingDQN(action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.optimizer = optim.Adam(self.train_net.parameters(), lr=learning_rate)

    def select_action(self, s, eps, env, batch_size):
        # select action according to epsilon-greedy method
        if np.random.rand() <= eps:
            a = env.action_space.sample()
        else:
            # greedy action is the largest Q value from the train network based on the input
            with torch.no_grad():
                # tiling the state to batch size due to how pysyft doesn't play well with certain forward arguments
                s = np.array(s)
                s = np.tile(s, (batch_size,1,1,1))
                input_state = torch.FloatTensor(s).to(self.device)
                a = self.train_net(input_state).max(1)[1]  # .view(1, 1)#.detach().cpu().numpy()[0]
                a = int(a[0])
        return np.array(a,dtype=np.int32).reshape((1,))

    def train(self, replay_buffer, batch_size, discount):
        # train the training network
        # sample a batch from the replay buffer
        x0, x1, a, r, d = replay_buffer.sample(batch_size)
        # turn batches into tensors and attack to GPU if available
        state_batch = torch.FloatTensor(x0).to(self.device)
        next_state_batch = torch.FloatTensor(x1).to(self.device)
        action_batch = torch.LongTensor(a).to(self.device)
        reward_batch = torch.FloatTensor(r).to(self.device)
        done_batch = torch.FloatTensor(1. - d).to(self.device)

        # get train net Q values
        train_q = self.train_net(state_batch).gather(1, action_batch)

        # get target net Q values
        with torch.no_grad():
            # Vanilla DQN: get target values from target net
            #             target_net_q = reward_batch + done_batch * discount * \
            #                      torch.max( self.target_net(next_state_batch).detach(), dim=1)[0].view(batch_size, -1)

            # Double DQN: get argmax values from train network, use argmax in target network
            train_argmax = self.train_net(next_state_batch).max(1)[1].view(batch_size, 1)
            target_net_q = reward_batch + done_batch * discount * \
                           self.target_net(next_state_batch).gather(1, train_argmax)

        # get loss between train q values and target q values
        # DQN implementations typically use MSE loss or Huber loss (smooth_l1_loss is similar to Huber)
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(train_q, target_net_q)
        loss = F.smooth_l1_loss(train_q, target_net_q)

        # optimize the parameters with the loss
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.train_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # we return the loss so we can monitor loss and debug the network if necessary
        return loss.detach().cpu().numpy()

    def update_target_network(self, num_iter, update_every):
        # update target network every so often
        # hard target network update: updates target network fully with train network params
        if num_iter % update_every == 0:
            # print("Updating target network parameters")
            self.target_net.load_state_dict(self.train_net.state_dict())

    def update_target_network_soft(self, num_iter, update_every, update_tau=0.001):
        # soft target network update: update target network with mixture of train and target
        if num_iter % update_every == 0:
            for target_var, var in zip(self.target_net.parameters(), self.train_net.parameters()):
                target_var.data.copy_((1. - update_tau) * target_var.data + (update_tau) * var.data)


# from: https://github.com/openai/baselines/baselines/common/atari_wrappers.py
# from: https://github.com/Officium/RL-Experiments/blob/master/src/common/wrappers.py 

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames so it's important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        shape = (2,) + env.observation_space.shape
        self._obs_buffer = np.zeros(shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    @staticmethod
    def reward(reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        shape = (1 if self.grayscale else 3, self.height, self.width)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        size = (self.width, self.height)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame.transpose((2, 0, 1))


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also `LazyFrames`
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = (shp[0] * k,) + shp[1:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.asarray(self._get_ob()), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can be
        huge for DQN's 1M frames replay buffers.
        This object should only be converted to numpy array before being passed
        to the model. You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-3)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    # if episode_life:
    #     env = EpisodicLifeEnv(env)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    #env = WarpFrame(env, width=84, height=84) # no need to warp. observation size is fine
    env = MaxAndSkipEnv(env, skip=4)
    # if scale:
    #     env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class Scalarize:
    """
    Convert a VecEnv into an Env
    There is a minor difference between this and a normal Env, which is that
    the final observation (when done=True) does not exist, and instead you will
    receive the second to last observation a second time due to how the VecEnv
    interface handles resets.  In addition, you are cannot step this
    environment after done is True, since that is not possible for VecEnvs.
    """

    def __init__(self, venv) -> None:
        assert venv.num_envs == 1
        self._venv = venv
        self._waiting_for_reset = True
        self._previous_obs = None
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space
        self.metadata = self._venv.metadata
        #self.spec = self._venv.spec
        self.reward_range = self._venv.reward_range

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            # dict space
            scalar_obs = {}
            for k, v in obs.items():
                scalar_obs[k] = v[0]
            return scalar_obs
        else:
            return obs[0]

    def reset(self):
        self._waiting_for_reset = False
        obs = self._venv.reset()
        self._previous_obs = obs
        return self._process_obs(obs)

    def step(self, action):
        assert not self._waiting_for_reset
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = np.array([action], dtype=self._venv.action_space.dtype)
        else:
            action = np.expand_dims(action, axis=0)
        obs, rews, dones, infos = self._venv.step(action)
        if dones[0]:
            self._waiting_for_reset = True
            obs = self._previous_obs
        else:
            self._previous_obs = obs
        return self._process_obs(obs), rews[0], dones[0], infos[0]

    def render(self, mode="human"):
        if mode == "human":
            return self._venv.render(mode=mode)
        else:
            return self._venv.get_images(mode=mode)[0]

    def close(self):
        return self._venv.close()

    def seed(self, seed=None):
        return self._venv.seed(seed)

    @property
    def unwrapped(self):
        # it might make more sense to return the venv.unwrapped here
        # except that the interface is different for a venv so things are unlikely to work
        return self

    def __repr__(self):
        return f"<Scalarize venv={self._venv}>"
