import torch
import torch.nn as nn
from entmax import sparsemax
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions import Categorical


class ScaledDotProductTokenOPTAttention(nn.Module):
    def __init__(self, temperature, dropout_attn=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout_attn = nn.Dropout(dropout_attn)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), float('-inf'))
            # sparsemax cannot handle all -inf input
            inf_mask = mask.bool().all(dim=2).unsqueeze(2).repeat(1, 1, mask.size()[2])
            attn = attn.masked_fill(inf_mask, 0)

        attn = sparsemax(attn, dim=2)

        if mask is not None:
            # if agent is inactive and all entities were masked
            attn = attn.masked_fill(inf_mask, 0)

        attn = self.dropout_attn(attn)

        out = torch.bmm(attn, v)

        return out, attn


class MultiHeadTokenOPTAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, rnn_hidden_dim, dropout_attn=0.0, dropout_attn_out=0.0):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.fc_select = nn.Linear(emb_dim, n_heads)
        self.fc_latent = nn.Linear(rnn_hidden_dim + emb_dim, n_heads)

        self.w_k = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_q = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)

        self.attention = ScaledDotProductTokenOPTAttention(temperature=emb_dim ** 0.5, dropout_attn=dropout_attn)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)

        self.disentangle_x = []
        self.disentangle_classifier = nn.Linear(emb_dim, n_heads)
        self.disentangle_loss = nn.CrossEntropyLoss(reduce=False)
        self.cmi_attn_select = []
        self.cmi_attn_latent = []
        self.use_pattern = False

    def set_pattern(self, use_pattern):
        self.disentangle_x = []
        self.cmi_attn_select = []
        self.cmi_attn_latent = []
        self.use_pattern = use_pattern

    def forward(self, x, h, mask=None):
        b, t, e = x.size()
        n_heads = self.n_heads

        k = self.w_k(x).view(b, t, n_heads, e)
        q = self.w_q(x).view(b, t, n_heads, e)
        v = self.w_v(x).view(b, t, n_heads, e)

        k = k.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        q = q.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        v = v.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)

        if mask is not None:
            mask = mask.repeat(n_heads, 1, 1)

        out, _ = self.attention(q, k, v, mask=mask)

        out = out.view(n_heads, b, t, e)
        out = out.permute(1, 2, 0, 3).contiguous()

        if self.use_pattern:
            self.disentangle_x.append(out.view(b * t, n_heads, e))

        out = out.view(b, t, n_heads, e)

        attn_select = self.fc_select(torch.mean(x, dim=1))
        attn_select = F.softmax(attn_select, dim=1)

        if self.use_pattern:
            self.cmi_attn_select.append(attn_select.view(b, n_heads))
            attn_latent = self.fc_latent(torch.mean(torch.cat([h.detach(), x], dim=2), dim=1))
            attn_latent = F.softmax(attn_latent, dim=1)
            self.cmi_attn_latent.append(attn_latent.view(b, n_heads))

        attn_select = attn_select.view(b, 1, 1, n_heads)
        out = torch.matmul(attn_select, out)
        out = out.squeeze(2)

        out = self.dropout_attn_out(out)

        return out

    def cal_disentangle_loss(self):
        x = torch.cat(self.disentangle_x, dim=0)
        dist = torch.bmm(x, x.permute(0, 2, 1))
        positive_res = torch.exp(torch.diagonal(dist, dim1=-2, dim2=-1) - torch.max(dist, dim=2)[0])
        negative_res = torch.sum(torch.exp(dist - torch.max(dist, dim=2)[0].unsqueeze(2)), dim=2)
        loss = torch.mean(-torch.log(positive_res / negative_res), dim=1)

        return loss

    def cal_cmi_loss(self):
        attn_select = torch.cat(self.cmi_attn_select, dim=0)
        attn_latent = torch.cat(self.cmi_attn_latent, dim=0)
        distribution_select = Categorical(probs=attn_select)
        distribution_latent = Categorical(probs=attn_latent)
        entropy_loss = distribution_select.entropy()
        kl_loss = kl_divergence(distribution_select, distribution_latent)

        return entropy_loss, kl_loss


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_emb_dim, dropout_ff=0.0):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_dim, ff_emb_dim),
            nn.ReLU(),
            nn.Linear(ff_emb_dim, emb_dim)
        )

        self.dropout = nn.Dropout(dropout_ff)

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)

        return out


class TokenOPTTransformerBlock(nn.Module):

    def __init__(self, emb_dim, n_heads, ff_emb_dim, rnn_hidden_dim, dropout_attn=0.0, dropout_attn_out=0.0, dropout_ff=0.0):
        super().__init__()

        self.attn = MultiHeadTokenOPTAttention(emb_dim, n_heads, rnn_hidden_dim, dropout_attn, dropout_attn_out)
        self.ff = PositionWiseFeedForward(emb_dim, ff_emb_dim, dropout_ff)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, inputs):
        x, h, mask = inputs

        attn_x = self.attn(x, h, mask)
        x = self.norm1(attn_x + x)

        ff_x = self.ff(x)
        x = self.norm2(ff_x + x)

        return x, h, mask


class TokenOPTTransformer(nn.Module):

    def __init__(self, n_blocks, emb_dim, n_heads, ff_emb_dim, rnn_hidden_dim,
                 dropout_attn=0.0, dropout_attn_out=0.0, dropout_ff=0.0):
        super().__init__()

        self.transformer_blocks = nn.Sequential(*[
            TokenOPTTransformerBlock(emb_dim, n_heads, ff_emb_dim, rnn_hidden_dim, dropout_attn, dropout_attn_out, dropout_ff)
            for _ in range(n_blocks)])

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, h, mask=None):

        out, _, _ = self.transformer_blocks((x, h, mask))

        out = self.fc(out)

        return out

