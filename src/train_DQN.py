import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model_DQN import QNet
from memory import Memory_DQN as Memory
from tensorboardX import SummaryWriter

from config_DQN import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length
from collections import deque

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

def get_action(state_series, target_net, epsilon, env):
    if np.random.rand() <= epsilon or len(state_series) < sequence_length:
        return env.action_space.sample()
    else:
        return target_net.get_action(torch.stack(list(state_series)))

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())

def state_to_partial_observability(state):
    state = state[[0, 2]]
    return state

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = 2
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    N_EPISODES = 5000
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    for e in range(N_EPISODES):
        done = False

        state_series = deque(maxlen=sequence_length)
        next_state_series = deque(maxlen=sequence_length)
        score = 0
        state = env.reset()
        
        state = state_to_partial_observability(state)
        state = torch.Tensor(state).to(device)
        
        next_state_series.append(state)
        while not done:
            steps += 1
            state_series.append(state)
            action = get_action(state_series, target_net, epsilon, env)
            next_state, reward, done, _ = env.step(action)

            next_state = state_to_partial_observability(next_state)
            next_state = torch.Tensor(next_state).to(device)


            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            if len(state_series) >= sequence_length:
                memory.push(state_series, next_state_series, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.000005
                epsilon = max(epsilon, 0.01)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | loss: {:.5f} | epsilon: {:.2f}'.format(
                e, running_score, loss, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break

if __name__=="__main__":
    main()