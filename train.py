import os
import argparse
import numpy as np
from tensordict import TensorDict
from envs.tsp_bsp import TspSplittingEnv
from envs.tsp_ori import TspEnv
from utils import rollout_with_agents, set_seed, save_args_to_file
import torch
torch.set_float32_matmul_precision('high')
import torch.optim as opt
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from data_augment import augment
import pickle


parser = argparse.ArgumentParser(description='start training')
parser.add_argument('--problem', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--epoch_size', type=int, required=True)
parser.add_argument('--nodes', type=int, required=True, help='N nodes')
parser.add_argument('--agents_min', type=int, required=True, help='M agents MAX')
parser.add_argument('--agents_max', type=int, required=True, help='M agents MAX')
parser.add_argument('--folder', type=str, required=True, help='folder to save logs & models')
parser.add_argument('--aug', type=int, default=1, help='aug during training')
parser.add_argument('--batch', type=int)
parser.add_argument('--history', type=int, default=0)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--ft', type=str)
parser.add_argument('--rescale', type=int, default=0)
parser.add_argument('--div', type=int, default=1)
parser.add_argument('--miniloop', type=int, default=1)
parser.add_argument('--resume', type=str)
parser.add_argument('--port', type=int, required=True)

reuse_encoder = None # 'logs/new_50-2-10/99.pt'

args = parser.parse_args()

folder = f'./logs/{args.folder}'

assert args.batch % args.div == 0

if args.history == 0:
    from net.tsp_lstm import Encoder, Decoder
else:
    assert False

# torch.autograd.set_detect_anomaly(True)

def adv_rescale(m_agents, adv):
    with torch.no_grad():
        num_classes = m_agents.max().item() + 1
        x_sum = torch.bincount(m_agents, adv)
        x_cnt = torch.bincount(m_agents)
        x_avg = x_sum / x_cnt.clamp(min=1)
        max_per_class = torch.full((num_classes,), float('-inf'), dtype=adv.dtype, device=adv.device)
        max_per_class = torch.scatter_reduce(
            max_per_class, 0, m_agents, adv, reduce='amax', include_self=True
        )
        min_per_class = torch.full((num_classes,), float('inf'), dtype=adv.dtype, device=adv.device)
        min_per_class = torch.scatter_reduce(
            min_per_class, 0, m_agents, adv, reduce='amin', include_self=True
        )

        sum_x2 = torch.zeros(num_classes, device=adv.device).scatter_add_(0, m_agents, adv**2)
        std = torch.sqrt((sum_x2 / x_cnt.clamp(min=1)) - x_avg ** 2)
    return {
        'avg': x_avg,
        'min': min_per_class,
        'max': max_per_class,
        'rng': max_per_class - min_per_class,
        'std': std
    }

def train_loop(rank, world_size):
    init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    assert args.div == world_size
    set_seed(args.seed + rank)

    device = f'cuda:{rank}'
    if args.problem == 'mtsp':
        env = TspSplittingEnv(args.nodes, args.agents_min, args.agents_max, args.batch * args.aug // args.div // args.miniloop, device)
    elif args.problem == 'mtspo':
        env = TspEnv(args.nodes, args.agents_min, args.agents_max, args.batch * args.aug // args.div // args.miniloop, device)
    else:
        assert False, 'not impl'
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    if args.ft:
        models = torch.load(args.ft, map_location='cpu')
        if isinstance(models, dict):
            encoder.load_state_dict(models['encoder'])
            decoder.load_state_dict(models['decoder'], strict=False)
        else:
            encoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[0].items()})
            decoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[1].items()})
        decoder.ft = True
        decoder.context_trans = nn.Sequential(
                nn.Linear(128, 8 * 128),
                nn.ReLU(),
                nn.Linear(8 * 128, 128)
            ).to(device)
        optimizer = opt.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)
    else:
        optimizer = opt.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    epoch_start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if isinstance(ckpt, dict):
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['decoder'])
            optimizer.load_state_dict(ckpt['optimizer'])
            epoch_start = ckpt['epoch'] + 1
            # torch.set_rng_state(ckpt['rng_state'])
        else:
            encoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in ckpt[0].items()})
            decoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in ckpt[1].items()})
    if reuse_encoder:
        ckpt = torch.load(reuse_encoder, map_location='cpu')
        encoder.load_state_dict(ckpt[0])

    # encoder = torch.compile(encoder)
    # decoder = torch.compile(decoder)
    encoder = DistributedDataParallel(encoder, device_ids=[rank])
    decoder = DistributedDataParallel(decoder, device_ids=[rank])

    scaler = GradScaler(device='cuda')

    # with open('testdata/mtsp_200_3333.pkl', 'rb') as fin:
    #     test_data = pickle.load(fin)
    #     test_data = np.array(test_data)
    for epoch in range(epoch_start, args.epochs):
        # one epoch
        assert args.epoch_size % args.batch == 0
        n_batchs = args.epoch_size // args.batch
        tr = trange(n_batchs, desc=f'Epoch={epoch}') if rank == 0 else range(n_batchs)
        for this_it in tr:
            with autocast(device_type='cuda'):
                # for single GPU
                if args.problem == 'mpdp':
                    n_nodes = args.nodes + 1
                else:
                    n_nodes = args.nodes
                for mini in range(args.miniloop):

                    this_locs = torch.rand((args.batch // args.div // args.miniloop, n_nodes, 2), device=device)
                    this_agents = torch.randint(args.agents_min, args.agents_max + 1, (args.batch // args.div // args.miniloop, ), device=device)

                    this_locs = augment(this_locs, args.aug)
                    this_agents = this_agents.repeat((args.aug, ))

                    td = TensorDict({
                        'locs': this_locs,
                        'agents': this_agents
                    }, batch_size=(args.batch * args.aug // args.div // args.miniloop, ), device=device)

                    td = env.reset(td)
                    node_embeddings, graph_embeddings, scale_embeddings = encoder(td)
                    decoder.module.build_cache(node_embeddings)

                    log_probs = []
                    while not env.is_done():
                        logits = decoder(td, node_embeddings, graph_embeddings, scale_embeddings)
                        dist = torch.distributions.Categorical(logits=logits)
                        acts = dist.sample()

                        log_probs.append(dist.log_prob(acts))

                        td['action'] = acts
                        td = env.step(td)
                    rewards = env.get_reward()

                    log_p = torch.stack(log_probs).transpose(0, 1)
                    bl = rewards.view(args.aug, args.batch // args.div // args.miniloop).mean(axis=0)
                    bl = bl.repeat(args.aug)
                    adv = rewards - bl

                    stat = adv_rescale(td['agents'], adv)

                    if args.rescale == 1:
                        adv = (1. / stat['std'])[td['agents']] * adv
                    loss = -log_p * adv.view(-1, 1)
                    loss = loss.mean() / args.miniloop

                    if mini != args.miniloop - 1:
                        with decoder.no_sync():
                            scaler.scale(loss).backward()
                    else:
                        scaler.scale(loss).backward()
                # single gpu end
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if rank == 0 and (epoch % 10 == 9 or args.ft):
            # torch.save((encoder.state_dict(), decoder.state_dict()), os.path.join(folder, f'{epoch}.pt'))
            torch.save({
                'epoch': epoch,
                'encoder': encoder.module.state_dict(),
                'decoder': decoder.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state()
            }, os.path.join(folder, f'{epoch}.pt'))
    destroy_process_group()


def main():
    # torch.autograd.set_detect_anomaly(True)
    # sw = SummaryWriter(folder)
    os.makedirs(os.path.join('./logs', args.folder), exist_ok=True)
    save_args_to_file(args, os.path.join('./logs', args.folder, 'cfg.json'))
    world_size = args.div
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    mp.spawn(train_loop, (world_size, ), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()

