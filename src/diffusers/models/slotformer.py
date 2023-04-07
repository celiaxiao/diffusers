import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin

from diffusers.models.modeling_utils import ModelMixin
# from nerv.training import BaseModel

# from slotformer.base_slots.models import StoSAVi


def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    # ViT, BEiT etc. all use zero-init learnable pos enc
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    # in SlotFormer, we find out that sine P.E. is already good enough
    elif 'sin' in pos_enc:  # 'sin', 'sine'
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding


class Rollouter(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass


class SlotFormer(ModelMixin, ConfigMixin, Rollouter):
    """Transformer encoder only."""

    def __init__(
        self,
        num_slots,
        slot_size,
        horizon,  # burn-in steps
        t_pe='sin',  # temporal P.E.
        slots_pe='',  # slots P.E., None in SlotFormer
        # Transformer-related configs
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.horizon = horizon
        self.in_proj = nn.Linear(slot_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers)
        self.enc_t_pe = build_pos_enc(t_pe, horizon, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    def forward(self, x, pred_len=1):
        """Forward function.

        Args:
            x: [B, horizon, num_slots, slot_size]
            pred_len: int

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.horizon, 'wrong burn-in steps'

        B = x.shape[0]
        x = x.flatten(1, 2)  # [B, H * N, slot_size]
        in_x = x

        # temporal_pe repeat for each slot, shouldn't be None
        # [1, H, D] --> [B, H, N, D] --> [B, H * N, D]
        enc_pe = self.enc_t_pe.unsqueeze(2).\
            repeat(B, 1, self.num_slots, 1).flatten(1, 2)
        # slots_pe repeat for each timestep
        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1).\
                repeat(B, self.horizon, 1, 1).flatten(1, 2)
            enc_pe = slots_pe + enc_pe

        # generate future slots autoregressively
        pred_out = []
        for _ in range(pred_len):
            # project to latent space
            x = self.in_proj(in_x)
            # encoder positional encoding
            x = x + enc_pe
            # spatio-temporal interaction via transformer
            x = self.transformer_encoder(x)
            # take the last N output tokens to predict slots
            pred_slots = self.out_proj(x[:, -self.num_slots:])
            pred_out.append(pred_slots)
            # feed the predicted slots autoregressively
            in_x = torch.cat([in_x[:, self.num_slots:], pred_out[-1]], dim=1)

        return torch.stack(pred_out, dim=1)

    @property
    def dtype(self):
        return self.in_proj.weight.dtype

    @property
    def device(self):
        return self.in_proj.weight.device