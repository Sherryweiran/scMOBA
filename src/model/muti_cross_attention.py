import torch
from torch import nn
import torch.nn.functional as F
import math

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, att_dropout=0.0):
        super(CrossMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout
    

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, context, pad_mask=None):
        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        batch_size = x.size(0)

        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        Q = self.Wq(x)
        K = self.Wk(context)
        V = self.Wv(context)

        # 分头 [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 512/8=64]
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_szie, num_heads, seq_len, seq_len] = [3, 8, 5, 5]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, seq_len, seq_len] -> [batch_size, nums_head, seq_len, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)

        # 自己的多头注意力效果没有torch的好，我猜是因为它的dropout给了att权重，而不是fc
        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        # [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 64]
        output = torch.matmul(att_weights, V)

        # 不同头的结果拼接 [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        output = self.fc(output)


        return output
