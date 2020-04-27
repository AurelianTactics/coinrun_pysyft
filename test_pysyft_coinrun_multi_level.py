"""
Test a trained agent
"""

import time
from mpi4py import MPI
#import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, dqn_utils
from coinrun.config import Config
import numpy as np
import torch
from PIL import Image
import pickle

def main():
    # set seed
    args = setup_utils.setup_and_load()
    print("args are")
    print(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = 0
    set_global_seeds(seed * 100 + rank)

    # Initialize env
    nenvs = 1
    env_init_size = nenvs
    env = utils.make_general_env(nenvs, seed=rank)

    # check and use GPU if available if not use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))

    # wrap env (not needed with Coinrun options)
    #env = dqn_utils.wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=False)
    action_size = env.action_space.n

    #env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # hyperparameters
    timesteps = 1000000 #2000000 #1000#2000000  # run env for this many time steps
    hidden_size = 512  # side of hidden layer of FFNN that connects CNN to outputs
    learning_rate = 0.0001  # learning rate of optimizer
    batch_size = 32  # size of batch trained on
    #start_training_after = 50 #10000  # start training NN after this many timesteps
    discount = 0.99  # discount future states by
    is_dueling = True
    is_impala_net = False

    frame_skip = 4 #hold action for this many frames

    # create DQN Agent
    dqn_agent = dqn_utils.DQNAgent(action_size, hidden_size, learning_rate, is_dueling, is_impala_net)

    train_time_id = 1587836568
    test_time_id = int(time.time())
    load_array = []
    for i in range(20000,250000,20000):
       load_array.append(i)
    load_array.append('FINAL')
    # load agent
    for i in load_array:
        model_number = i
        PATH = "saved_models/dqn_model_{}_{}.pt".format(train_time_id, model_number)
        dqn_agent.train_net.load_state_dict(torch.load(PATH))
        dqn_agent.train_net.eval()

        # training loop
        stats_rewards_list = []  # store stats for plotting in this
        stats_every = 1  # print stats every this many episodes
        total_reward = 0.
        episode = 1
        episode_length = 0
        stats_loss = 0.
        epsilon = 0.02
        # can't call env.reset() in coinrun, so start each episode with a no acton
        state_list, _, _, _ = env.step(np.array([0], dtype=np.int32))

        for ts in range(timesteps):
            # select an action from the agent's policy
            action = dqn_agent.select_action(state_list[0].squeeze(axis=-1), epsilon, env, batch_size)

            # enter action into the env
            for _ in range(frame_skip):
                next_state_list, reward_list, done_list, _ = env.step(action)
                total_reward += reward_list[0]
                if done_list[0]:
                    break
            done = done_list[0]
            episode_length += 1

            if done:
                state_list = env.reset()

                stats_rewards_list.append((episode, total_reward, episode_length))
                episode += 1
                if episode >= 201:
                    break
                if episode % stats_every == 0:
                    print('Episode: {}'.format(episode),
                          'Timestep: {}'.format(ts),
                          'Episode reward {}'.format(total_reward),
                          'Episode len {}'.format(episode_length),
                          'Mean reward: {:.1f}'.format(np.mean(stats_rewards_list, axis=0)[1]),
                          'Mean length: {:.1f}'.format(np.mean(stats_rewards_list, axis=0)[2]))
                    stats_loss = 0.

                total_reward = 0
                episode_length = 0
            else:
                state_list = next_state_list

        #save final stats
        stats_save_string = "saved_models/test_env_stats_{}_{}_{}.pickle".format(train_time_id, model_number, test_time_id)
        with open(stats_save_string, 'wb') as handle:
            pickle.dump(stats_rewards_list, handle)


if __name__ == '__main__':
    main()

