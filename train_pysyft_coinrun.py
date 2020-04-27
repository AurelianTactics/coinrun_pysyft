"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
from mpi4py import MPI
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, dqn_utils
from coinrun.config import Config
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
import pickle
from PIL import Image

def main():
    # check and use GPU if available if not use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))

    # arguments
    args = setup_utils.setup_and_load()
    print("Arguments are")
    print(args)

    # set seed
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    time_int = int(time.time())
    seed = time_int % 10000
    set_global_seeds(seed * 100 + rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if Config.NUM_ENVS > 1:
        print("To do: add multi env support")
    nenvs = 1 #Config.NUM_ENVS
    env = utils.make_general_env(nenvs, seed=rank)

    # wrap env (not needed with Coinrun options)
    #env = dqn_utils.wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=False)
    action_size = env.action_space.n

    # set up pysyft workers
    num_workers = 2
    hook = sy.TorchHook(torch)
    worker_1 = sy.VirtualWorker(hook, id='worker_1')
    worker_2 = sy.VirtualWorker(hook, id='worker_2')
    secure_worker = sy.VirtualWorker(hook, id='secure_worker')
    worker_list = []
    worker_list.append(worker_1)
    worker_list.append(worker_2)

    # Training hyperparameters
    timesteps = 250000 #2000000 #1000#2000000  # run env for this many time steps
    hidden_size = 512  # side of hidden layer of FFNN that connects CNN to outputs
    is_dueling = True
    is_impala_net = False
    learning_rate = 0.0001  # learning rate of optimizer
    batch_size = 32  # size of batch trained on
    start_training_after = 10000  # start training NN after this many timesteps
    discount = 0.99  # discount future states by

    epsilon_start = 1.0  # epsilon greedy start value
    epsilon_min = 0.02  # epsilon greedy end value
    epsilon_decay_steps = timesteps * .5  # decay epsilon over this many timesteps
    epsilon_step = (epsilon_start - epsilon_min) / (epsilon_decay_steps)  # decrement epsilon by this amount every timestep

    update_target_every = 1  # update target network every this steps
    tau = 0.001 # soft target updating amount

    frame_skip = 4 #hold action for this many frames
    save_every = 10000 #timesteps to save model after
    train_every = 1 # number of times to train

    # create replay buffer
    replay_size = 50000  # size of replay buffer
    replay_buffer_list = []
    for i in range(num_workers):
        replay_buffer = dqn_utils.ReplayBuffer(max_size=replay_size)
        replay_buffer_list.append(replay_buffer)

    # create DQN Agent
    dqn_agent = dqn_utils.DQNAgent(action_size, hidden_size, learning_rate, is_dueling, is_impala_net)

    # create states for every env
    stats_every = 10 #10  # print stats every this many episodes
    stats_list = [] # store stats for each env init here
    for i in range(num_workers):
        temp_dict = {}
        temp_dict['episode'] = 0
        temp_dict['mean_reward_total'] = 0.
        temp_dict['mean_ep_length_total'] = 0.
        temp_dict['mean_reward_recent'] = 0.
        temp_dict['mean_ep_length_recent'] = 0.
        temp_dict['episode_loss'] = 0.
        temp_dict['episode_reward'] = 0.
        temp_dict['episode_length'] = 0.
        stats_list.append(temp_dict)

    # training loop
    epsilon = epsilon_start
    # take no_action on first step to get state
        # use state to tell which level
        # env.reset() does not produce and observation in CoinRun until an action is taken
    no_action = np.zeros((nenvs,),dtype=np.int32)
    state_list, _, _, _ = env.step(no_action)

    # assign each level to a worker
        # coinrun doesn't have a way to tell the current level so take mean of first screen of level and use dictionary to assign levels
        # worker_level is used to tell which replay buffer to put data into (ie which worker is training)
    level_worker_dict = {}
    levels_assigned = 0
    def get_worker_level(state, lw_dict, la, nw):
        temp_key = int(1000*np.mean(state))
        if temp_key not in lw_dict:
            la += 1
            lw_dict[temp_key] = la % nw
            print("Adding new key to level_worker_dict. current size is: {}".format(len(lw_dict)))
            print(lw_dict)
        return lw_dict[temp_key], lw_dict, la

    worker_level, level_worker_dict, levels_assigned = get_worker_level(state_list[0], level_worker_dict, levels_assigned, num_workers)

    for ts in range(timesteps):
        # decay epsilon
        epsilon -= epsilon_step
        if epsilon < epsilon_min:
            epsilon = epsilon_min

        # select an action from the agent's policy
        action = dqn_agent.select_action(state_list[0].squeeze(axis=-1), epsilon, env, batch_size)

        # enter action into the env
        reward_frame_skip = 0.
        for _ in range(frame_skip):
            next_state_list, reward_list, done_list, _ = env.step(action)
            stats_list[worker_level]['episode_reward'] += reward_list[0]
            reward_frame_skip += reward_list[0]
            if done_list[0]:
                break
        done = done_list[0]
        stats_list[worker_level]['episode_length'] += 1

        # add experience to replay buffer
        replay_buffer_list[worker_level].add((state_list[0].squeeze(axis=-1), next_state_list[0].squeeze(axis=-1), action, reward_frame_skip, float(done)))

        if done:
            # env.reset doesn't reset the coinrun env but does produce image of first frame, which we can use get the worker_level
            state_list = env.reset()
            worker_level, level_worker_dict, levels_assigned = get_worker_level(state_list[0],
                                                                                        level_worker_dict,
                                                                                        levels_assigned, num_workers)

            #update stats
            stats_list[worker_level]['episode'] += 1
            #overall averages
            stats_list[worker_level]['mean_reward_total'] = (stats_list[worker_level]['mean_reward_total'] * (
                        stats_list[worker_level]['episode'] - 1) + stats_list[worker_level]['episode_reward']) / \
                                                            stats_list[worker_level]['episode']
            stats_list[worker_level]['mean_ep_length_total'] = (stats_list[worker_level]['mean_ep_length_total'] * (
                        stats_list[worker_level]['episode'] - 1) + stats_list[worker_level]['episode_length']) / \
                                                               stats_list[worker_level]['episode']
             # keep running average of last stats_every episodes
            if stats_list[worker_level]['episode'] >= stats_every:
                temp_episodes_num = stats_every
            else:
                temp_episodes_num = stats_list[worker_level]['episode']
            stats_list[worker_level]['mean_reward_recent'] = (stats_list[worker_level]['mean_reward_recent'] * (temp_episodes_num - 1) +
                                                   stats_list[worker_level]['episode_reward']) / temp_episodes_num
            stats_list[worker_level]['mean_ep_length_recent'] = (stats_list[worker_level]['mean_ep_length_recent'] * (temp_episodes_num - 1) +
                                                      stats_list[worker_level]['episode_length']) / temp_episodes_num
            # reset episode stats
            stats_list[worker_level]['episode_reward'] = 0.
            stats_list[worker_level]['episode_length'] = 0

            # print stats
            if stats_list[worker_level]['episode'] % stats_every == 0:
                print('w: {}'.format(worker_level),
                      'epi: {}'.format(stats_list[worker_level]['episode']),
                      't: {}'.format(ts),
                      'r: {:.1f}'.format(stats_list[worker_level]['mean_reward_total']),
                      'l: {:.1f}'.format(stats_list[worker_level]['mean_ep_length_total']),
                      'r r: {:.1f}'.format(stats_list[worker_level]['mean_reward_recent']),
                      'r l: {:.1f}'.format(stats_list[worker_level]['mean_ep_length_recent']),
                      'eps: {:.2f}'.format(epsilon),
                      'loss: {:.1f}'.format(stats_list[worker_level]['episode_loss']))

                stats_list[worker_level]['episode_loss'] = 0.
        else:
            state_list = next_state_list

        if ts > start_training_after:
            # train the agent
            # typical DQN gather experiences and trains once every iteration
                # train_every can modify that to 'train_every' many times every 'train_every'th iteration
                # example: if train_every=10 then train 10 times every 10th iteration
            if ts % train_every == 0:
                # pysyft federated learning training
                    # copy model to each worker
                    # each worker trains on its own data from its own replay buffer
                    # updated models from each worker sent to a secure worker who updates the new model
                worker_dqn_list = []
                worker_dqn_target_list = []
                worker_opt_list = []
                for i in range(num_workers):
                    worker_dqn_list.append(dqn_agent.train_net.copy().send(worker_list[i]))
                    worker_dqn_target_list.append(dqn_agent.target_net.copy().send(worker_list[i]))
                    worker_opt_list.append(optim.Adam(params=worker_dqn_list[i].parameters(), lr=learning_rate))

                for i in range(num_workers):
                    for _ in range(train_every):
                        # sample a batch from the replay buffer
                        x0, x1, a, r, d = replay_buffer_list[i].sample(batch_size)
                        # turn batches into tensors and attack to GPU if available
                        state_batch = torch.FloatTensor(x0).to(device)
                        state_batch = torch.unsqueeze(state_batch,dim=1)
                        next_state_batch = torch.FloatTensor(x1).to(device)
                        next_state_batch = torch.unsqueeze(next_state_batch, dim=1)
                        action_batch = torch.LongTensor(a).to(device)
                        reward_batch = torch.FloatTensor(r).to(device)
                        done_batch = torch.FloatTensor(1. - d).to(device)

                        # send data to worker
                        worker_state_batch = state_batch.send(worker_list[i])
                        worker_next_state_batch = next_state_batch.send(worker_list[i])
                        worker_action_batch = action_batch.send(worker_list[i])
                        worker_reward_batch = reward_batch.send(worker_list[i])
                        worker_done_batch = done_batch.send(worker_list[i])

                        train_q = worker_dqn_list[i](worker_state_batch).gather(1, worker_action_batch)

                        with torch.no_grad():
                            # Double DQN: get argmax values from train network, use argmax in target network
                            train_argmax = worker_dqn_list[i](worker_next_state_batch).max(1)[1].view(batch_size, 1)
                            target_net_q = worker_reward_batch + worker_done_batch * discount * \
                                            worker_dqn_target_list[i](worker_next_state_batch).gather(1, train_argmax)

                        # get loss between train q values and target q values
                        # DQN implementations typically use MSE loss or Huber loss (smooth_l1_loss is similar to Huber)
                        # loss_fn = nn.MSELoss()
                        # loss = loss_fn(train_q, target_net_q)
                        loss = F.smooth_l1_loss(train_q, target_net_q)

                        # optimize the parameters with the loss
                        worker_opt_list[i].zero_grad()
                        loss.backward()
                        for param in worker_dqn_list[i].parameters():
                            param.grad.data.clamp_(-1, 1)
                        worker_opt_list[i].step()
                        # get loss stats
                        #print("loss is {}".format(loss))
                        temp_loss = loss.get()
                        #print("loss get is {}".format(temp_loss))
                        stats_list[i]['episode_loss'] += temp_loss.detach().cpu().numpy()

                    # move the worker trained model to secure worker for updating the centralized DQN
                    worker_dqn_list[i].move(secure_worker)
                    with torch.no_grad():
                        # first worker replaces centralized DQN parameters, then do keep a running average as each new worker's params are found
                        if i == 0:
                            dqn_agent.train_net.load_state_dict(worker_dqn_list[i].get().state_dict())
                        else:
                            tau = 1. / (1 + i)
                            temp_net = worker_dqn_list[i].get()
                            for dqn_var, temp_var in zip(dqn_agent.train_net.parameters(), temp_net.parameters()):
                                dqn_var.data.copy_((1. - tau) * dqn_var.data + (tau) * temp_var.data)

            # save the network
            if ts % save_every == 0:
                save_string = "saved_models/dqn_model_{}_{}.pt".format(time_int, ts)
                torch.save(dqn_agent.train_net.state_dict(), save_string)
                stats_save_string = "saved_models/stats_{}_{}.pickle".format(time_int, ts)
                with open(stats_save_string, 'wb') as handle:
                    pickle.dump(stats_list, handle)
            # update the target network
            dqn_agent.update_target_network_soft(ts, update_target_every, tau)

    print("save final model")
    save_string = "saved_models/dqn_model_{}_FINAL.pt".format(time_int)
    torch.save(dqn_agent.train_net.state_dict(), save_string)
    stats_save_string = "saved_models/stats_{}_FINAL.pickle".format(time_int)
    with open(stats_save_string, 'wb') as handle:
        pickle.dump(stats_list, handle)



if __name__ == '__main__':
    main()

