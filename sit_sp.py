import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed,Attention,Mlp
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int (H = W)
    return:
        pos_embed: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid, grid)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half dim for height, half for width
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension
    pos: (M,) or (M,1)
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def modulate(x, shift, scale):
    return x* ( 1+ scale.unsqueeze(1)) + shift.unsqueeze(1)
##########################################################################
class TimestepEmbedder(nn.Module):
    def __init__(self,hidden_size, frequency_embedding_size = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias = True)
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t,dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args),torch.sin(args)], dim=-1)
        if dim % 2 : # if dimension is odd , we need to match to dimension
            embedding = torch.cat([embedding,torch.zeros_like(embedding[:,:1])], dim=-1)
        return embedding # (N,D)
    

    def forward(self,t):
        t_freq = self.timestep_embedding(t,self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
"""
basic block of SIT
mlp_ratio : h - r*h - h -> which means it decides how many tiems it will expand between middles
Layernorm :  to normalize base on token -> Layer norm is beneficial when they have different size of embedding
"""    

class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,6*hidden_size, bias= True)
        )
        # x is embedding vector(after patchfy and tokenized)
    def forward(self,x,c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6,dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x),shift_msa,scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x),shift_mlp,scale_mlp))
        return x

# reconstruct token into patch
class FinalLayer(nn.Module):
    def __init__(self,hidden_size,patch_size,out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False,eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*out_channels, bias = True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )
    # x is (B,N,D) , c is (B,D)
    def forward(self,x,c):
        shift, scale = self.adaLN_modulation(c).chunk(2,dim=1)
        x = modulate(self.norm(x), shift,scale)
        x= self.linear(x)# (B, N, P*P*C)
        return x
"""
process
x(image) -> patchfy ->token + class toekn + posistion embed

-> SiT Block(H or L) -> FinalLayer(emb->patch) -> unpatchfy-> predicted noise

"""
class SmallREG(nn.Module):
    def __init__(self, 
                 input_size=32,
                 patch_size=2,
                 in_channels=3,
                 hidden_size=384,
                 depth=8,
                 num_heads = 12,
                 num_classes=10,
                 class_dropout_prob=0.1,
                 high_low_split = 0.5,
                 split_threshold = 0.5,
                 overlap = 0.0
                 ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.high_low_split = high_low_split
        self.split_threshold = split_threshold
        self.overlap = overlap
        
        #1. Patch Embedder(32*32 ->16*16 = 256 patches)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches

        #2. class embedder(cls) (B, D)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size) # 10 class + null 

        #3. tiemstep embedder (B,)->(B,D)
        self.t_embedder = TimestepEmbedder(hidden_size)

        #4. Positional Embedding(B,N+1,D) -> for cls token, +1
        # it doesn't need to be trained, but should be stored together with 
        pos_embed = get_2d_sincos_pos_embed(embed_dim=hidden_size,grid_size=int(num_patches**0.5),cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0),requires_grad = False)

        #5. Transformer Blocks
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size,hidden_size)
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.normal_(self.pos_embed, std =0.02) # low std -> less meaningful, high std-> unstable attention
        nn.init.normal_(self.y_embedder.weight, std = 0.02)
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight,0)
            # nn.init.normal_(block.adaLN_modulation[-1].weight, std=1e-4)
            nn.init.constant_(block.adaLN_modulation[-1].bias,0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight,0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        # (N,T, patch_size**2 *C) ->(N,C,H,W)
        c = self.in_channels
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h*w ==x.shape[1]

        x = x.reshape(shape=(x.shape[0],h,w,p,p,c))
        x = torch.einsum('nhwpqc->nchpwq',x)
        imgs = x.reshape(shape=(x.shape[0],c,h*p,w*p))
        return imgs
    
    # x is (N,C,H,W) , t(N,), y(N,)
    def forward(self,x,t,y):
        x = self.x_embedder(x) + self.pos_embed[:,1:,:] # (N,L,D)
        t_emb = self.t_embedder(t) # (N,D)

        # Classifier-Free Guidance
        # learn both conditional(force to move) and unconditional(natural way)
        # objective is to grow the conditional direction keeping as much as natural way
        if self.training and self.class_dropout_prob > 0 :
            keep_mask = torch.rand(y.shape, device=y.device) > self.class_dropout_prob
            y = torch.where(keep_mask,y, torch.tensor(self.num_classes, device=y.device))

        y_emb = self.y_embedder(y) #(N,D)
        y_token = y_emb.unsqueeze(1) + self.pos_embed[:, :1, :]
        x = torch.cat([y_token,x],dim=1) #(N,L+1,D)
        c = t_emb + y_emb # (N,D)

        # for block in self.blocks:
        #     x = block(x,c)


        split_idx = int(self.depth * self.high_low_split)
        overlap = int(self.overlap * self.depth)

        # t [0,1000]q
        t_norm = t / 1000.0 if torch.max(t) > 1.0 else t
        high_mask = t_norm > self.split_threshold
        low_mask = ~high_mask
        
        for i, block in enumerate(self.blocks):
        # i: 현재 블록 번호 (0~7)
        # 1. 공통 블록 (Overlap) - 항상 작동
            if i < overlap or i > self.depth - overlap: 
                x = block(x, c)
            else:
                x_new = x.clone()
                # High일 때는 뒤쪽 블록(4,5,6) 위주로, Low일 때는 앞쪽(2,3) 위주로
                if high_mask.any() and i >= (self.depth // 2):
                    x_new[high_mask] = block(x[high_mask], c[high_mask])
                
                # Low-t 전문 블록 (예: 앞쪽 절반의 전문가 블록들)
                elif low_mask.any() and i < (self.depth // 2):
                    x_new[low_mask] = block(x[low_mask], c[low_mask])
                
                # 만약 High 구간인데 Low 전용 블록을 만났다면? 
                # -> 그냥 통과(Identity) 하되, Shared 구간이 분포를 잡아줌
                x = x_new
            
        x = self.adapter(x)
        x = self.final_layer(x[:,1:],c)
        x = self.unpatchify(x) # (N,C,H,W)

        return x
    






   
    

        



