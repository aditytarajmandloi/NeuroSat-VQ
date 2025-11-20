# ========================
# model_v2_7.py
# ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

LATENT_CHANNELS = 96
SLICE_DIM = 4
NUM_EMBEDDINGS = 4096


# -----------------------------------------------------
# Memory-safe, numerically safe VectorQuantizerEMA
# -----------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5, chunk_size=0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.chunk_size = chunk_size

        embed = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32) * 0.01

        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer("ema_w", embed.clone())

    def reset_ema(self):
        self.ema_cluster_size.zero_()
        with torch.no_grad():
            self.ema_w.normal_(0, 1e-3)
            self.embedding.copy_(self.ema_w.clone())

    def _compute_distances(self, flat, codebook):
        flat_norm = (flat ** 2).sum(dim=1, keepdim=True)
        code_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)
        return flat_norm + code_norm - 2 * flat @ codebook.t()

    def forward(self, inputs):
        # inputs: [B, C, H, W]
        device = inputs.device
        self.embedding = self.embedding.to(device)
        self.ema_cluster_size = self.ema_cluster_size.to(device)
        self.ema_w = self.ema_w.to(device)

        x = inputs.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        flat = x.view(-1, C)  # N,D
        N = flat.shape[0]

        # ---------- DISTANCES ----------
        if self.chunk_size > 0 and N > self.chunk_size:
            all_idx = []
            for i in range(0, N, self.chunk_size):
                chunk = flat[i : i + self.chunk_size]
                dist = self._compute_distances(chunk, self.embedding)
                all_idx.append(torch.argmin(dist, dim=1))
            encoding_indices = torch.cat(all_idx)
        else:
            dist = self._compute_distances(flat, self.embedding)
            encoding_indices = torch.argmin(dist, dim=1)

        # ---------- GATHER ----------
        quantized_flat = self.embedding[encoding_indices]
        quantized = quantized_flat.view(B, H, W, C)

        # ---------- EMA UPDATE ----------
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(encoding_indices, minlength=self.num_embeddings).float()

                dw = torch.zeros(self.num_embeddings, C, device=device)
                dw.index_add_(0, encoding_indices, flat.float())

                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                cluster_size = torch.clamp(self.ema_cluster_size + self.epsilon, min=1e-2)
                new_embed = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.copy_(new_embed)

        # ---------- LOSSES ----------
        e_loss = F.mse_loss(quantized.detach().float(), x.float())
        vq_loss = e_loss * self.commitment_cost

        # ---------- STRAIGHT THROUGH ----------
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, vq_loss, encoding_indices


# -----------------------------------------------------
# SEBlock + ResBlock
# -----------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, False),
            nn.ReLU(True),
            nn.Linear(channel // reduction, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, channels, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU(False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU(False)
        self.se = SEBlock(channels)

    def forward(self, x):
        def _f(x_in):
            out = self.act1(self.norm1(self.conv1(x_in)))
            out = self.act2(self.norm2(self.conv2(out)))
            out = self.se(out)
            return out + x_in

        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(_f, x, use_reentrant=False)
        else:
            return _f(x)


# -----------------------------------------------------
# Autoencoder v2.7
# -----------------------------------------------------
class V2Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------- Encoder ----------
        self.enc_conv1 = nn.Conv2d(3, 64, 3, 2, 1)
        self.enc_res1 = ResBlock(64)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc_res2 = ResBlock(128)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.enc_res3 = ResBlock(256, use_checkpoint=True)
        self.enc_conv4 = nn.Conv2d(256, LATENT_CHANNELS, 3, 2, 1)

        # ---------- VQ ----------
        self.quantizer = VectorQuantizerEMA(
            NUM_EMBEDDINGS, SLICE_DIM, 0.25, 0.99, 1e-5, chunk_size=0
        )

        # ---------- Decoder ----------
        self.dec_conv1 = nn.Conv2d(96, 1024, 3, 1, 1)
        self.dec_ps1 = nn.PixelShuffle(2)
        self.dec_res1 = ResBlock(256, use_checkpoint=True)

        self.dec_conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.dec_ps2 = nn.PixelShuffle(2)
        self.dec_res2 = ResBlock(128)

        self.dec_conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.dec_ps3 = nn.PixelShuffle(2)
        self.dec_res3 = ResBlock(64)

        self.dec_conv4 = nn.Conv2d(64, 256, 3, 1, 1)
        self.dec_ps4 = nn.PixelShuffle(2)
        self.dec_res4 = ResBlock(64)

        self.final_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = F.silu(self.enc_conv1(x))
        x = self.enc_res1(x)
        x = F.silu(self.enc_conv2(x))
        x = self.enc_res2(x)
        x = F.silu(self.enc_conv3(x))
        x = self.enc_res3(x)
        z = self.enc_conv4(x)  # [B,96,8,8]

        B, C, H, W = z.shape
        n_slices = C // SLICE_DIM
        z_sliced = z.view(B * n_slices, SLICE_DIM, H, W)

        z_q, vq_loss, indices = self.quantizer(z_sliced)
        z_q = z_q.view(B, C, H, W)

        x = self.dec_ps1(self.dec_conv1(z_q))
        x = self.dec_res1(x)
        x = self.dec_ps2(self.dec_conv2(x))
        x = self.dec_res2(x)
        x = self.dec_ps3(self.dec_conv3(x))
        x = self.dec_res3(x)
        x = self.dec_ps4(self.dec_conv4(x))
        x = self.dec_res4(x)

        return torch.sigmoid(self.final_conv(x)), vq_loss, indices
