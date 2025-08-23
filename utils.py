import json
import torch
from tensordict import TensorDict
import numpy as np
import random
from splitting_solver import get_minmax_length


def set_seed(seed):
    """
    Set the random seed for reproducibility in PyTorch and other libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch on CPU
    torch.cuda.manual_seed(seed)  # For PyTorch on the current GPU
    torch.cuda.manual_seed_all(seed)  # For PyTorch on all GPUs
    # torch.backends.cudnn.deterministic = True  # Ensure deterministic results with cuDNN
    # torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility


def save_args_to_file(args, file_path):
    """
    保存 argparse 的参数到文件。
    :param args: argparse.Namespace 对象
    :param file_path: 保存文件的路径
    """
    args_dict = vars(args)  # 转换为字典
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)  # 保存为 JSON 格式，带缩进
    print(f"Arguments saved to {file_path}")


def select_node_embedding(node_embeddings, index):
    '''
    node_embeddings: tensor (batch_size, n_nodes, embedding_size)
    index: tensor (batch_size, )

    ret: tensor (batch_size, embedding_size)
    '''
    batch_size = node_embeddings.shape[0]
    return node_embeddings[torch.arange(batch_size), index, :]


def recover_with_zeros(loc, act, m):
    EPSILON = 1e-5
    cost = get_minmax_length(loc[np.newaxis, ...], act[np.newaxis, ...], m)[0]
    now_tot = 0
    prev_node = 0
    n = loc.shape[0]
    result = []
    for i in range(n - 1):
        this_seg = np.linalg.norm(loc[prev_node] - loc[act[i]])
        back = np.linalg.norm(loc[0] - loc[act[i]])
        if now_tot + this_seg + back > cost + EPSILON:
            ## split
            # prev = now_tot + np.linalg.norm(loc[0] - loc[prev_node]) # ??
            # assert prev <= cost + 1e-5
            # print(prev)
            result.append(0)
            result.append(act[i])
            now_tot = back
        else:
            result.append(act[i])
            now_tot += this_seg
        prev_node = act[i]
    result.append(0)
    return result, cost


def check_minmax_length(locs, acts, m):
    assert acts[0] != 0 and acts[-1] == 0
    prev_node = 0
    dist = 0
    result = []
    for a in acts:
        dist += np.linalg.norm(locs[a] - locs[prev_node])
        if a == 0:
            result.append(dist)
            dist = 0
        prev_node = a
    # print(len(result))
    assert len(result) <= m
    return np.max(result)


def rollout(env, encoder, decoder, locs, device, rollout_type, temperature=1):
    td = TensorDict(
        {'locs': locs}, batch_size=(locs.shape[0], ), device=device
    )
    td = env.reset(td)
    actions = []
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        node_embeddings = encoder(td)
        decoder.build_cache(node_embeddings)
        while not env.is_done():
            logits = decoder(td, node_embeddings)
            if rollout_type == 'greedy':
                acts = torch.argmax(logits, dim=1)
            else:
                acts = torch.distributions.Categorical(logits=logits / temperature).sample()
            td['action'] = acts
            actions.append(acts)
            td = env.step(td)
        rewards = env.get_reward()
    actions = torch.stack(actions).transpose(0, 1).contiguous()
    encoder.train()
    decoder.train()
    return rewards, actions


def rollout_with_agents(env, encoder, decoder, locs, agents, device, rollout_type, temperature=1):
    td = TensorDict(
        {'locs': locs, 'agents': agents}, batch_size=(locs.shape[0], ), device=device
    )
    td = env.reset(td)
    actions = []
    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        node_embeddings, graph_embeddings, scale_embeddings = encoder(td)
        decoder.build_cache(node_embeddings)
        while not env.is_done():
            logits = decoder(td, node_embeddings, graph_embeddings, scale_embeddings)
            if rollout_type == 'greedy':
                acts = torch.argmax(logits, dim=1)
            else:
                acts = torch.distributions.Categorical(logits=logits / temperature).sample()
            td['action'] = acts
            actions.append(acts)
            td = env.step(td)
        rewards = env.get_reward()
    actions = torch.stack(actions).transpose(0, 1).contiguous()
    encoder.train()
    decoder.train()
    return rewards, actions