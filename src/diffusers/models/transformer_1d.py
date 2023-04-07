from typing import Union, Optional, Tuple
import logging
import math
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.slotEncoderDecoder import TrajectoryToSlotEncoderDecoder, TrajectoryToSlotWithExtraTokenEncoder
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
logger = logging.getLogger(__name__)

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
class Transformer1DModel(ModuleAttrMixin, ModelMixin, ConfigMixin,):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Transformer1DModel):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    


    def forward(self, 
        sample: torch.Tensor, 
        cond: Optional[torch.Tensor]=None, 
        timestep: Union[torch.Tensor, float, int] = 0, 
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]

class SlotTransformerModel(Transformer1DModel):
    def __init__(self, 
                num_slots: int, 
                input_dim: int, # per object action_dim + obs_dim
                action_dim: int, # per object action dim
                extra_dim: int, # extra dim
                slot_size: int = 128,
                n_emb: int = 768,
                alpha: float = 1.0,
                *args, **kwargs,
                ) -> None:
        super().__init__(input_dim=input_dim * num_slots, n_emb=n_emb, *args, **kwargs)
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.action_dim = action_dim
        self.extra_dim = extra_dim
        self.alpha = alpha

        assert input_dim > action_dim
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim - action_dim, slot_size),
            nn.Flatten(start_dim=2)
        )

        # [B, H, N * slot_size]  -> [B, H, N, D]
        self.out_proj = nn.Sequential(
            nn.Linear(slot_size, input_dim - action_dim),
            nn.Flatten(start_dim=2)
        )
        
        self.extra_in_proj = nn.Sequential(
            nn.Linear(extra_dim, slot_size),

        )
        self.extra_out_proj = nn.Sequential(
            nn.Linear(slot_size, extra_dim),

        )
        
        self.slots_emb = nn.Parameter(
            get_sin_pos_enc(num_slots, slot_size), requires_grad=False)
        self.extra_token_emb = nn.Parameter(torch.zeros(1, slot_size), requires_grad=True)
        # extra token is also a slot
        num_slots += 1
        # [B, H, N * (slot_size + act_dim)] -> [B, H, n_emb]
        self.input_emb = nn.Linear(num_slots * slot_size + action_dim, n_emb)
        self.head = nn.Linear(n_emb, num_slots * slot_size + action_dim)
    
    def forward(self, 
        sample: torch.Tensor, 
        cond: Optional[torch.Tensor]=None, 
        timestep: Union[torch.Tensor, float, int] = 0, 
        **kwargs):
        """
        sample: [B, horizon, num_slots, input_dim]
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: [B, horizon, num_slots, input_dim]
        """

        extra_tokens = sample[:, :, -self.extra_dim:]
        x = sample[:, :, :-self.extra_dim]
        slots = x[:, :, self.action_dim:]
        actions = x[:, :, :self.action_dim]
        B, H = x.shape[:2]
        slots = slots.reshape(B, H, self.num_slots, -1)
        
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        slots = self.in_proj(slots)
        # encoder positional encoding
        enc_pe = self.slots_emb.unsqueeze(0).repeat(B, H, 1, 1).flatten(2)
        slots += enc_pe

        extra_tokens = extra_tokens.unsqueeze(2)        
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        extra_tokens = self.extra_in_proj(extra_tokens)
        # encoder positional encoding
        enc_pe = self.extra_token_emb.unsqueeze(0).repeat(B, H, 1, 1).flatten(2)
        extra_tokens += enc_pe

        slots = torch.cat([slots, extra_tokens], dim=2).flatten(2)
        # [B, H, N * (slot_size + act_dim)]
        x = torch.cat([actions, slots], dim=-1)
        # [B, H, n_emb]
        x = super().forward(x, cond, timestep, **kwargs)
        
        slots = x[:, :, self.action_dim:].reshape(B, H, -1, self.slot_size)
        actions = x[:, :, :self.action_dim]
        extra_slots = slots[:, :, -1]
        slots = slots[:, :, :-1]
        # [B, H, N, slot_size] -> [B, H, N, D]
        slots = slots.reshape(B, H, self.num_slots, -1)
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        slots = self.out_proj(slots)
        extra_slots = self.extra_out_proj(extra_slots.unsqueeze(2))
        slots = torch.cat([slots.flatten(2), extra_slots.flatten(2)], dim=-1)
        # [B, H, N, input_dim]
        x = torch.cat([actions, slots], dim=-1)
        return x
        
class SlotTransformer1DModel(Transformer1DModel):
    def __init__(self, 
                num_slots: int, 
                input_dim: int, # per object action_dim + obs_dim
                action_dim: int, # per object action dim
                extra_dim: int = 0, # extra information dimemsion contained in the observation
                slot_size: int = 128,
                n_emb: int = 768,
                alpha: float = 1.0,
                *args, **kwargs,
                ) -> None:
        super().__init__(input_dim=input_dim, n_emb=n_emb, *args, **kwargs)
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.action_dim = action_dim
        self.extra_dim = extra_dim
        assert extra_dim > 0
        
        assert input_dim > action_dim
        if extra_dim == 0:
            self.slot_encoder_decoder = TrajectoryToSlotEncoderDecoder(
                input_dim=input_dim,
                action_dim=action_dim,
                slot_size=slot_size,
                num_slots=num_slots,
                alpha=alpha,
            )
        else:
            self.slot_encoder_decoder = TrajectoryToSlotWithExtraTokenEncoder(
                input_dim=input_dim,
                action_dim=action_dim,
                extra_dim=extra_dim,
                slot_size=slot_size,
                num_slots=num_slots,
                alpha=alpha,
            )
            # extra token is another slot
            num_slots += 1
        # [B, H, N * slot_size + act_dim] -> [B, H, n_emb]
        self.input_emb = nn.Sequential(
            nn.Linear(num_slots * slot_size + action_dim, n_emb),
            nn.Flatten(start_dim=2)
        )
        self.head = nn.Sequential(
            nn.Linear(n_emb, num_slots * slot_size + action_dim),
            nn.Flatten(start_dim=2)
        )
    
    def forward(self, 
        sample: torch.Tensor, 
        cond: Optional[torch.Tensor]=None, 
        timestep: Union[torch.Tensor, float, int] = 0, 
        **kwargs):
        """
        sample: [B, horizon, input_dim + extra_dim]
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: [B, horizon, input_dim + extra_dim]
        """
        x = self.slot_encoder_decoder.encode(sample)
        # [B, H, (N * slot_size) + act_dim]
        x = super().forward(x, cond, timestep, **kwargs)
        
        x = self.slot_encoder_decoder.decode(x)
        return x


def test():
    # GPT with time embedding
    transformer = Transformer1DModel(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = Transformer1DModel(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = Transformer1DModel(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = Transformer1DModel(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

