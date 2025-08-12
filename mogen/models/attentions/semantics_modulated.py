import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS
from einops import rearrange

class StandardAttention(nn.Module):
    
    def __init__(self, latent_dim,
                      cond_latent_dim,
                      num_heads):
        super().__init__()
        self.num_heads = num_heads
        
        self.norm_x = nn.LayerNorm(latent_dim)
        self.norm_cond = nn.LayerNorm(cond_latent_dim)
        
        self.key_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.key_x = nn.Linear(latent_dim, latent_dim)
        self.value_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.value_x = nn.Linear(latent_dim, latent_dim)
        
        self.proj_y = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, query, x, cond_emb, src_mask, cond_mask):
        """
        query: B, T, D (已经生成的query向量)
        x: B, T, D (batch, sequence_length, dimension)
        cond_emb: B, C, D (batch, condition_length, dimension)
        src_mask: B, T, 1
        cond_mask: B, C, 1
        """
        B, T, D = x.shape
        C = cond_emb.shape[1]
        N = T + C
        H = self.num_heads
        
        # 重塑query为多头形式
        query = query.view(B, T, H, -1)
        
        # 生成key和value
        key = torch.cat((
            self.key_cond(self.norm_cond(cond_emb)),
            self.key_x(self.norm_x(x))
        ), dim=1).view(B, N, H, -1)
        
        value = torch.cat((
            self.value_cond(self.norm_cond(cond_emb)) * cond_mask,
            self.value_x(self.norm_x(x)) * src_mask
        ), dim=1).view(B, N, H, -1)
        
        # 应用mask到key上
        attention_mask = torch.cat((
            (1 - cond_mask) * -1000000,  # 对条件进行掩码
            (1 - src_mask) * -1000000    # 对输入进行掩码
        ), dim=1).view(B, 1, 1, N)
        
        # 计算注意力分数: q * k -> (b, t, h, n)
        attn_scores = torch.einsum('bthd,bnhd->bthn', query, key)
        
        # 应用mask并进行softmax
        attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=3)
        
        # 计算最终输出: (attn_probs * v) -> (b, t, h, d)
        y = torch.einsum('bthn,bnhd->bthd', attn_probs, value)
        
        # 重塑并投影
        y = y.reshape(B, T, D)
        y = self.proj_y(y)
        
        return y

class SubAttention(nn.Module):

    def __init__(self, latent_dim,
                       cond_latent_dim,
                       num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.norm_x = nn.LayerNorm(latent_dim)
        self.norm_cond = nn.LayerNorm(cond_latent_dim)
        
        self.key_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.value_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.key_x = nn.Linear(latent_dim, latent_dim)
        self.value_x = nn.Linear(latent_dim, latent_dim)

        self.proj_y = nn.Linear(latent_dim, latent_dim)
 
    # from line_profiler import profile
    # @profile
    def forward(self, query, x, cond_emb,  src_mask, cond_mask):
        """
        x: B, T, H, D
        """
        B, T, D = x.shape
        N = x.shape[1] + cond_emb.shape[1]
        H = self.num_heads
        # B, N, D

        key = torch.cat((
            self.key_cond(self.norm_cond(cond_emb)) + (1 - cond_mask) * -1000000, # -inf: 10%
            self.key_x(self.norm_x(x)) + (1 - src_mask) * -1000000
        ), dim=1) # bnhd [256, 371, 512]
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        # re_feat_value = cond_emb.reshape(B, -1, D)
        value = torch.cat((
            self.value_cond(self.norm_cond(cond_emb)) * cond_mask,
            self.value_x(self.norm_x(x)) * src_mask,
        ), dim=1).view(B, N, H, -1) # bnhl l=d
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = self.proj_y(y)
        return y

class SelfAttention(nn.Module):

    def __init__(self, latent_dim,
                       num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.norm_x = nn.LayerNorm(latent_dim)

        self.key_x = nn.Linear(latent_dim, latent_dim)
        self.value_x = nn.Linear(latent_dim, latent_dim)

        self.proj_y = nn.Linear(latent_dim, latent_dim)
 
    # from line_profiler import profile
    # @profile
    def forward(self, query, x,   src_mask):
        """
        x: B, T, H, D
        """
        B, T, D = x.shape
        N = x.shape[1]
        H = self.num_heads
        # B, N, D

        key = self.key_x(self.norm_x(x)) + (1 - src_mask) * -1000000 # bnhd [256, 371, 512]
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        # re_feat_value = cond_emb.reshape(B, -1, D)
        value = (self.value_x(self.norm_x(x)) * src_mask).view(B, N, H, -1) # bnhl l=d
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = self.proj_y(y)
        return y

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       stick_latent_dim,
                       num_heads,
                       dropout,
                       locus_dim,
                       time_embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.query_locus = nn.Linear(locus_dim, latent_dim)
        self.text_encoder = SubAttention(latent_dim, text_latent_dim, num_heads)
        self.stick_encoder = StandardAttention(latent_dim, stick_latent_dim, num_heads)
        self.y_encoder = SelfAttention(latent_dim, num_heads)
        # self.mid_proj = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )

        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    # from line_profiler import profile
    # @profile
    def forward(self, x, text_emb, stick_emb, other_emb, src_mask, cond_type, stick_mask, locus_emb, mid_query):
        """
        x: B, T, D
        xf: B, N, L # text features; re_dict: retrieval information
        cond_type [text, both, stick, none].sum() == batch_size
        """
        # B, T, D
        if type(mid_query) is torch.Tensor or type(mid_query) is nn.Parameter:
            _query = self.y_encoder(mid_query, x, src_mask)
            y = x + self.proj_out(_query, other_emb)
            return y

        query = self.query(self.norm(x))
        # B, N, D
        ci = [sum(cond_type[:i]) for i in range(len(cond_type))]

        text_query = query[:ci[2]]
        text_x = x[:ci[2]]
        text_x_mask = src_mask[:ci[2]]
        text_emb = text_emb[:ci[2]]

        stick_query = query[ci[1]:ci[3]]
        locus_emb = locus_emb[ci[1]:ci[3]]
        # stick_query = self.query_locus(torch.cat((stick_query, locus_emb), dim=-1))
        stick_x = x[ci[1]:ci[3]]
        stick_x = stick_x + self.query_locus(locus_emb)
        stick_x_mask = src_mask[ci[1]:ci[3]]
        stick_emb = stick_emb[ci[1]:ci[3]]
        stick_mask = stick_mask[ci[1]:ci[3]]
        
        text_y = self.text_encoder(text_query, text_x, text_emb, text_x_mask, 1)
        stick_y = self.stick_encoder(stick_query, stick_x, stick_emb, stick_x_mask, stick_mask)

        query[:ci[2]] = query[:ci[2]] + text_y
        query[ci[1]:ci[3]] = query[ci[1]:ci[3]] + stick_y

        if type(mid_query) == int and mid_query == -1:
            return x, query
        '''
        b_query = query[:ci[1]]
        # b1_query = query[:ci[1]]
        b1_text_offset = text_y[:ci[1]]
        
        # b2_query = query[ci[1]:ci[2]]
        b2_text_offset = text_y[ci[1]:ci[2]]
        b2_stick_offset = stick_y[:ci[2]-ci[1]]

        # b3_query = query[ci[2]:ci[3]]
        b3_stick_offset = stick_y[ci[2]-ci[1]:]
        
        # b4_query = query[ci[3]:]
        '''

        _query = self.y_encoder(query, x, src_mask)
        y = x + self.proj_out(_query, other_emb)


        return y
