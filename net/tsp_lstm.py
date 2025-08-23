import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import select_node_embedding
from einops import rearrange


class GATLayer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mha = nn.MultiheadAttention(128, 8, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.mha(x, x, x, need_weights=False)[0] + x
        x = self.bn1(rearrange(x, 'b n e -> (b n) e')).view(x.size())
        x = self.mlp(x) + x
        x = self.bn2(rearrange(x, 'b n e -> (b n) e')).view(x.size())
        return x


class Encoder(nn.Module):

    """
    Input: Graphs (batch_size, n_nodes, 2)
    Output: Node embeddings (batch_size, n_nodes, 128)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_init = nn.Linear(2, 128)
        self.depot_init = nn.Linear(2, 128)
        self.scale_init = nn.Linear(1, 128)

        self.net = nn.Sequential(
            *[GATLayer() for _ in range(3)]
        )

    def forward(self, td):
        locs = td['locs']

        n = locs.shape[1]
        m = td['agents']
        scale_embedding = self.scale_init((n / m).view(-1, 1))

        node_x = locs[..., 1:, :]
        depot_x = locs[..., 0:1, :]

        node_init = self.node_init(node_x)
        depot_init = self.depot_init(depot_x)

        init_embedding = torch.cat([depot_init, node_init], dim=-2)
        node_embedding = self.net(init_embedding)
        graph_embedding = node_embedding.mean(1)
        return node_embedding, graph_embedding, scale_embedding


class Decoder(nn.Module):

    def __init__(self, ft=False, tanh_alpha=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ft = ft
        self.tanh_alpha = tanh_alpha
        self.linear_first = nn.Linear(128 * 4, 128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        # self.norm = nn.LayerNorm(128)
        if self.ft:
            self.context_trans = nn.Sequential(
                nn.Linear(128, 8 * 128),
                nn.ReLU(),
                nn.Linear(8 * 128, 128)
            )
        self.Wq = nn.Linear(128, 128, bias=False)
        self.Wk = nn.Linear(128, 128, bias=False)
        self.Wv = nn.Linear(128, 128, bias=False)
        self.linear_proj = nn.Linear(128, 128, bias=False)
        self.num_heads = 8

        self.logitk = nn.Linear(128, 128)
        self.cache = None
        self.now_step = 0

    def build_cache(self, node_embeddings):
        batch_size = node_embeddings.shape[0]
        self.prev_h = torch.zeros(1, batch_size, 128).to(self.logitk.weight.device)
        self.prev_c = torch.zeros(1, batch_size, 128).to(self.logitk.weight.device)

        self.cache = (
            self.Wk(node_embeddings),
            self.Wv(node_embeddings),
            self.logitk(node_embeddings)
        )
        self.now_step = 0

    def forward(self, td, node_embeddings, graph_embeddings, scale_embeddings):
        self.now_step += 1
        if self.now_step % 100 == 0:
            # assert False
            self.prev_c = self.prev_c.detach()
            self.prev_h = self.prev_h.detach()

        first_node_emb = select_node_embedding(node_embeddings, td['first_node'])
        cur_node_emb = select_node_embedding(node_embeddings, td['current_node'])
        
        context = torch.cat([graph_embeddings, first_node_emb, cur_node_emb, scale_embeddings], dim=1)
        context = self.linear_first(context)

        out, (self.prev_h, self.prev_c) = self.lstm(context.unsqueeze(1), (self.prev_h, self.prev_c))
        # else:
        #     out, (self.prev_h, self.prev_c) = self.lstm(context.unsqueeze(1), (self.prev_h, self.prev_c))
        context_out = out[:, -1, :] #+ context
        # context_out = self.norm(context_out)
        if self.ft:
            context_out = context_out + self.context_trans(context_out)
        context_q = self.Wq(context_out)
        context_k = self.cache[0]
        context_v = self.cache[1]

        # make head
        q = rearrange(context_q, 'b (1 e h) -> b h 1 e', h=8)
        k = rearrange(context_k, 'b n (e h) -> b h n e', h=8)
        v = rearrange(context_v, 'b n (e h) -> b h n e', h=8)
        mask = rearrange(td['mask'], 'b n -> b 1 1 n')
        att_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # need mask, yes
        # recover
        att_out = rearrange(att_out, 'b h l e -> b l (h e)')
        context_emb = self.linear_proj(att_out)  # batch_size, 1, 128

        logitK = rearrange(self.cache[2], 'b n e -> b e n')
        compatibility = torch.bmm(context_emb, logitK) / math.sqrt(128) # batch_size, 1, n
        compatibility = compatibility.squeeze(1)  # batch_size, n

        # mask
        compatibility = self.tanh_alpha * torch.tanh(compatibility)
        compatibility[td['mask'] == 1] = -torch.inf

        return compatibility
