import argparse
from collections import defaultdict
import time
from tqdm import tqdm, trange
from envs.tsp_bsp import TspSplittingEnv
from envs.tsp_ori import TspEnv

import pickle
import torch
import numpy as np

from utils import rollout_with_agents
from splitting_solver import get_minmax_length
from data_augment import augment, augment_xy_data_by_N_fold
from utils import set_seed


# parser = argparse.ArgumentParser(description='start training')
# parser.add_argument('--problem', type=str, required=True)
# parser.add_arlocalhostgument('--model', type=str, required=True)
# args = parser.parse_args()

device = 'cuda:0'
hist = 0

# NM = [(100, i) for i in range(2, 11)]
NM = [(100, i) for i in range(2, 11)]
# NM = [(200, i) for i in range(2, 21)]
# NM = [(500, i) for i in range(10, 41)]
# NM = [(2000, 100)]
n_aug =  8
n_samp = 16
div_n = 1
testcases = 100
model_ckpt = 'models/mTSP100/99.pt'
problem = 'mtsp'
exploration = 0
temperature = 1.
ft = False

if hist == 0:
    from net.tsp_lstm import Encoder, Decoder
else:
    assert False


def get_dist(locs, acts):
    prev = locs[0]
    dist = 0
    for a in acts:
        dist += np.linalg.norm(locs[a] - prev, ord=2)
        prev = locs[a]
    dist += np.linalg.norm(prev - locs[0], ord=2)
    return dist.item()


def parallel_eval(encoder, decoder, data, m_agents, n_aug, n_samp, div_n=1):
    locs = torch.from_numpy(data).float().to(device)
    num_instances, n_nodes = locs.shape[0], locs.shape[1]
    locs_aug = augment(locs, n_aug)
    locs_aug = locs_aug.repeat((n_samp, 1, 1))

    assert num_instances * n_aug * n_samp % div_n == 0

    batch_size = num_instances * n_aug * n_samp // div_n
    if problem == 'mtsp':
        env = TspSplittingEnv(n_nodes, m_agents, m_agents, batch_size, device)
    elif problem == 'mtspo':
        env = TspEnv(n_nodes, m_agents, m_agents, batch_size, device)
    else:
        assert False
    cost_cat = []
    acts_cat = []
    for z in trange(div_n):
        this_locs = locs_aug[z * batch_size: z * batch_size + batch_size]
        this_agents = torch.ones((batch_size, ), device=device) * m_agents
        rews, acts = rollout_with_agents(env, encoder, decoder, this_locs, this_agents, device, 'sampling' if n_samp != 1 else 'greedy', temperature=temperature)
        costs = -rews
        for t in range(batch_size // num_instances):
            cost_cat.append(costs[t * num_instances : t * num_instances + num_instances])
            acts_cat.append(acts[t * num_instances : t * num_instances + num_instances])

    result = torch.stack(cost_cat)
    result_act = torch.stack(acts_cat)
    # print(result.shape)
    final_cost, final_indices = result.min(dim=0)
    result_act = result_act[final_indices, torch.arange(result_act.shape[1], device=device), :]
    # print(final_cost.shape)
    return final_cost.mean().item(), result_act.cpu().numpy()


def main():
    set_seed(0)
    encoder = Encoder().to(device)
    decoder = Decoder(ft=ft, tanh_alpha=10).to(device)

    models = torch.load(model_ckpt, map_location='cpu')
    if isinstance(models, dict):
        encoder.load_state_dict(models['encoder'])
        decoder.load_state_dict(models['decoder'])
    else:
        encoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[0].items()})
        decoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[1].items()})
    print(model_ckpt)
    data_seed = 3333
    z_format = []
    for idx, (n, m) in enumerate(NM):
        # with open('testdata/tsp200_gaussian.pkl', 'rb') as fin:
        with open(f'testdata/mtsp_{n}_{data_seed}.pkl', 'rb') as fin:
            data = np.array(pickle.load(fin))[:testcases, :n]
        # print(data[0])
        diff = data - data[:, 0:1, :]
        print(diff.shape)
        dist = np.linalg.norm(diff, axis=2)
        print(dist.shape)
        print((dist.max(axis=1) * 2).mean())
        
        # locs = torch.from_numpy(data).float().to(device)
        
        res, res_act = parallel_eval(encoder, decoder, data, m, n_aug=n_aug, n_samp=n_samp, div_n=div_n)
        print(f'TSP_AVG @ {n}/{m}: {np.mean(res):.4f}')
        z_format.append(res)

        # model_name = model_ckpt.split('/')[-1].split('.')[0]

        # with open(f'testdata/{problem}_{n}_{m}_sol_{model_name}.pkl', 'wb') as fout:
        #     pickle.dump(res_act, fout)
        # x_min[idx] = min(x_min[idx], res)
    print('\n'.join(map(lambda x: f'{x:.4f}', z_format)))
    

if __name__ == '__main__':
    main()
