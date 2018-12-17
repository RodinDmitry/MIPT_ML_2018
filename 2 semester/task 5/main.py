import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.multiprocessing as mp
import time
from collections import defaultdict


class A3C_Model(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(num_inputs, 256), nn.LeakyReLU(0.1),
                                nn.Linear(256,256), nn.LeakyReLU(0.1),
                                nn.Linear(256,128), nn.LeakyReLU(0.1),
                                nn.Linear(128,128), nn.LeakyReLU(0.1),
                                nn.Linear(128,128), nn.LeakyReLU(0.1))

        self.lstm = nn.LSTMCell(128, 128)
        num_outputs = action_space
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = self.fc(x)
        x = x.view(1, 128)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)


def normal(x, mu, sigma):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        else:
            shared_param._grad = param.grad


class SharedAdam(optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-3,
                 weight_decay=0, amsgrad=True):
        defaults = defaultdict(lr=lr, betas=betas, eps=eps,
                               weight_decay=weight_decay, amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new(
                ).resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * \
                            math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def train(shared_model, optimizer, lr, steps=20, gamma=0.99):
    env = gym.make('BipedalWalker-v2')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    model = A3C_Model(state_dim, action_dim)
    model.load_state_dict(shared_model.state_dict())
    model.train(True)
    while True:
        i = 0
        obs = env.reset()
        is_done = False
        entropies = []
        log_probs = []
        values = []
        rewards = []
        obs = torch.from_numpy(obs).float()
        cx = Variable(torch.zeros(1, 128))
        hx = Variable(torch.zeros(1, 128))
        while not is_done:

            value, mu, sigma, (hx, cx) = model(
                (Variable(obs), (hx, cx)))
            mu = torch.clamp(mu, -1.0, 1.0)
            sigma = F.softplus(sigma) + 1e-5
            eps = torch.randn(mu.size())
            pi = np.array([math.pi])
            pi = torch.from_numpy(pi).float()
            eps = Variable(eps)
            pi = Variable(pi)

            action = (mu + sigma.sqrt() * eps).data
            act = Variable(action)
            prob = normal(act, mu, sigma)
            action = torch.clamp(action, -1.0, 1.0)
            entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
            entropies.append(entropy)
            log_prob = (prob + 1e-6).log()
            log_probs.append(log_prob)
            obs, reward, is_done, _ = env.step(
                action.cpu().numpy()[0])
            reward = max(min(float(reward), 1.0), -1.0)
            obs = torch.from_numpy(obs).float()
            i += 1
            values.append(value)
            rewards.append(reward)

            if i % steps == steps - 1 or is_done:
                if is_done:
                    R = torch.zeros(1, 1)
                else:
                    value, _, _, _ = model(
                        (Variable(obs), (hx, cx)))
                    R = value.data
                values.append(Variable(R))
                policy_loss = 0
                value_loss = 0
                R = Variable(R)
                gae = torch.zeros(1, 1)
                for i in reversed(range(len(rewards))):
                    R = gamma * R + rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data

                    gae = gae * gamma + delta_t

                    policy_loss = policy_loss - (log_probs[i].sum() * Variable(gae)) - (0.01 * entropies[i].sum())

                model.zero_grad()
                (policy_loss + 0.5 * value_loss).backward(retain_graph=True)
                ensure_shared_grads(model, shared_model)
                optimizer.step()
                model.load_state_dict(shared_model.state_dict())
                entropies = []
                log_probs = []
                values = []
                rewards = []


def eval_run(model):
    max_rew = -300

    env = gym.make('BipedalWalker-v2')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    actor = A3C_Model(state_dim, action_dim)
    num_runs = 50

    while True:
        rewards = []
        time.sleep(5)
        actor.load_state_dict(model.state_dict())
        actor.eval()
        actor.train(False)
        for _ in range(num_runs):
            total_reward = 0
            new_obs = env.reset()
            cx = Variable(torch.zeros(1, 128))
            hx = Variable(torch.zeros(1, 128))
            while True:
                obs = torch.from_numpy(new_obs).float()
                value, mu, sigma, (hx, cx) = actor((Variable(obs), (hx, cx)))
                mu = torch.clamp(mu.data, -1.0, 1.0)
                action = mu.cpu().numpy()[0]
                new_obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        total_reward = np.mean(rewards)
        print('Eval reward:', total_reward, 'Max reward:', max_rew)
        if max_rew < total_reward:
            max_rew = total_reward
            torch.save(model.state_dict(), 'best_model')


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    env.reset()
    mp.set_start_method('spawn')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    shared_model = A3C_Model(state_dim, action_dim)
    shared_model.share_memory()
    processes = []
    p = mp.Process(target=eval_run, args=(shared_model,))
    p.start()
    processes.append(p)
    workers = 3
    lr = 1e-4
    optimizer = SharedAdam(shared_model.parameters(), lr, amsgrad=True)
    optimizer.share_memory()
    for rank in range(0, workers):
        p = mp.Process(target=train, args=(shared_model, optimizer, lr, 20, 0.99))
        p.start()
        processes.append(p)
    for p in processes:
        time.sleep(0.1)
        p.join()
