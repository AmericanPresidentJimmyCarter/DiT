# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Conditioning                 #
#################################################################################

class CrossAttention2DConv(nn.Module):
    def __init__(self, x_sz, c_sz, nhead=16):
        super().__init__()

        self.proj_in = nn.Conv2d(
            x_sz,
            c_sz,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_out = nn.Conv2d(
            c_sz,
            x_sz,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.ln = nn.LayerNorm(x_sz)
        self.ln_out = nn.LayerNorm(x_sz)
        self.attn = torch.nn.MultiheadAttention(c_sz, nhead, bias=True, batch_first=True)

    def forward(self, x, c):
        norm_x = self.ln(x)
        norm_x = norm_x.unsqueeze(0)
        norm_x = rearrange(norm_x, 'a b c d -> a d c b')
        shaped_x = self.proj_in(norm_x)
        shaped_x = shaped_x.squeeze(0)
        shaped_x = rearrange(shaped_x, 'a b c -> c b a')
        x_out = self.attn(c, shaped_x, shaped_x, need_weights=False)[0]
        x_out = rearrange(shaped_x, 'c b a -> a b c')
        x_out = x_out.unsqueeze(0)
        reshaped_x = self.proj_out(x_out)
        reshaped_x = rearrange(reshaped_x, 'a d c b -> a b c d')
        reshaped_x = reshaped_x.squeeze(0)
        renorm_x = self.ln_out(reshaped_x)
        return renorm_x


class CrossAttention2D(nn.Module):
    def __init__(self, c_sz, nhead=16):
        super().__init__()

        self.proj_in = nn.Conv2d(
            c_sz,
            c_sz,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_out = nn.Conv2d(
            c_sz,
            c_sz,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.c_sz = c_sz
        self.ln1 = nn.LayerNorm(c_sz // 2)
        self.ln2 = nn.LayerNorm(c_sz // 2)
        self.ln_out = nn.LayerNorm(c_sz // 2)
        self.attn = torch.nn.MultiheadAttention(c_sz // 2, nhead, bias=True, 
            batch_first=True)

    def forward(self, c):
        '''
        Split the embeddings into their two constituents (CLIP and T5 text
        embeddings), then cross attention these.
        '''
        c1, c2 = c.split(self.c_sz // 2, dim=2)
        prenorm_c1 = self.ln1(c1)
        prenorm_c2 = self.ln2(c2)
        c_out = self.attn(prenorm_c1, prenorm_c2, prenorm_c2, need_weights=True)[0]
        renorm_c = self.ln_out(c_out)
        return renorm_c


def weighted_mean(tens: torch.Tensor, weights: list[float], dim: int) -> torch.Tensor:
    split_num = len(weights)
    chunks = tens.split(tens.size()[dim] // split_num, dim=dim)
    weighted_means = [torch.mean(chunks[i], dim=1) * w for
        i, w in enumerate(weights)]
    return torch.stack(weighted_means, dim=0).sum(dim=0) / split_num


class ConditionedTimestepEmbedder(nn.Module):
    """
    Embeds scalar timestep and conditioning into vector representations.
    """
    def __init__(self, hidden_size: int, c_mean_size: int, frequency_embedding_size: int=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size + c_mean_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, c):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_tens = torch.cat([t_freq, c], dim=1)
        t_emb = self.mlp(t_tens)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """
    def __init__(self, hidden_size, num_heads, cond_embed_size, mlp_ratio=4.0, frequency_embedding_size: int=256, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(frequency_embedding_size + cond_embed_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        context_dim=2048,
        skip_decay_factor=2.,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.skip_decay_factor = skip_decay_factor

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = ConditionedTimestepEmbedder(hidden_size, context_dim // 2)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False)

        self.dual_text_embed_cross_attn = CrossAttention2D(context_dim, num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, context_dim // 2, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Set up the attention emphasis scheduler for each block, so that we
        # can use windowed attention for every single block in the stack.
        depth_range = list(range(1, depth + 1))
        self.block_conditioning_mean_schedule = []
        for i in range(depth):
            as_deque = deque(reversed(depth_range))
            as_deque.rotate(i)
            self.block_conditioning_mean_schedule.append(list(as_deque))

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5,
        ))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if isinstance(block, DiTBlock):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if isinstance(block, DiTBlock):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, conditioning):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        c_attended = self.dual_text_embed_cross_attn(conditioning)

        for block_idx, block in enumerate(self.blocks):
            # (N, T, D) -> (N, D)
            c_block = weighted_mean(
                c_attended,
                self.block_conditioning_mean_schedule[block_idx],
                1, # Squish the token layer
            )
            c = self.t_embedder(t, c_block) # (N, D), (N, D) -> (N, D)
            x = block(x, c)                 # (N, T, D), (N, D) -> (N, T, D)
        x = self.final_layer(x, c)          # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)              # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, conditioning, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, conditioning)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XXL_2_CA(**kwargs):
    return DiT(depth=32, hidden_size=1280, patch_size=2, num_heads=16, **kwargs)

def DiT_S_2_CA(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=16, **kwargs)

if __name__ == "__main__":
    NUM_IMAGES = 2
    IMAGE_SIZE = 256
    device = "cuda"
    model = DiT(input_size=IMAGE_SIZE // 8, depth=32, hidden_size=1280, patch_size=2, num_heads=16) # .to(device)
    print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")
    x_random = torch.rand(NUM_IMAGES, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8) # .to(device)
    print('in', x_random.size())
    # c_random = torch.randn((1, 2048)).to(device)
    timestep_random = torch.randint(1, 250, (NUM_IMAGES,)) # .to(device)
    c_full_random = torch.randn((NUM_IMAGES, 77, 2048)) # .to(device)
    # c_full_random = torch.randn((1, 77, 2048)).to(device)
    # print('x random', x_random.size())
    out = model(x_random, timestep_random, c_full_random)
    print('out size', out.size())
    out2 = model.forward_with_cfg(x_random, timestep_random, c_full_random, 7.5)
    print('out2 size', out2.size())

    # out_flat = gumbel_sample(out_flat, temperature=1.0)
    # print('out_flat size', out_flat.size())
    # out_flat = out_flat.view(out.size(0), *out.shape[2:])
    # print('out_flat 2', out_flat.size())
