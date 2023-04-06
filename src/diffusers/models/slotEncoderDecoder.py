import torch
import torch.nn as nn

class SlotEncoder(nn.Module):
    def __init__(self, num_slots, slot_size, in_dim):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.in_dim = in_dim

    def forward(self, x):
        pass

class SlotDecoder(nn.Module):
    def __init__(self, num_slots, slot_size, out_dim):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.out_dim = out_dim

    def forward(self, x):
        pass 

def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]

class TrajectoryToSlotEncoder(SlotEncoder):
    def __init__(self, num_slots: int, 
                in_dim: int, 
                slot_size: int = 128,
                alpha: float = 1.0, # positional encoding scaling, 0 for no positional encoding
                ):
        super().__init__(num_slots, slot_size, in_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, slot_size),
            nn.Flatten(start_dim=2)
        )
        self.slots_emb = nn.Parameter(
            get_sin_pos_enc(num_slots, slot_size), requires_grad=False)
        self.alpha = alpha
    
    def forward(self, x):
        """
        x: [B, horizon, input_dim = obs_dim * num_slots]
        output: [B, horizon, slot_size * num_slots]
        """
        B, H = x.shape[:2]
        slots = x.reshape(B, H, self.num_slots, -1)
        
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        slots = self.in_proj(slots)
        # encoder positional encoding
        enc_pe = self.slots_emb.unsqueeze(0).repeat(B, H, 1, 1).flatten(2)
        slots += self.alpha * enc_pe

        return slots

class TrajectoryToSlotDecoder(SlotDecoder):
    def __init__(self, num_slots: int, 
                out_dim: int, # output_dim
                slot_size: int = 128,
                ):
        super().__init__(num_slots, slot_size, out_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(slot_size, out_dim),
            nn.Flatten(start_dim=2)
        )
    
    def forward(self, x):
        """
        x: [B, horizon, slot_size]
        output: [B, horizon, input_dim = obs_dim * num_slots]
        """
        B, H = x.shape[:2]
        slots = x.reshape(B, H, self.num_slots, -1)
        
        # [B, H, N, D] -> [B, H, N, slot_size] -> [B, H, N * slot_size]
        slots = self.out_proj(slots)
        return slots

class TrajectoryToSlotEncoderDecoder(nn.Module):
    def __init__(self, num_slots: int, 
                input_dim: int, # per object action_dim + obs_dim
                action_dim: int, # per object action dim
                slot_size: int = 128,
                alpha: float = 1.0, # positional encoding scaling, 0 for no positional encoding
                ):
        super().__init__()
        self.encoder = TrajectoryToSlotEncoder(num_slots, 
                                               input_dim - action_dim, 
                                               slot_size, 
                                               alpha)
        self.decoder = TrajectoryToSlotDecoder(num_slots, 
                                               input_dim - action_dim,  
                                               slot_size)
        self.action_dim = action_dim

    def encode(self, x):
        slots = x[:, :, self.action_dim:]
        actions = x[:, :, :self.action_dim]
        slots = self.encoder(slots)
        # [B, H, N * (slot_size + act_dim)]
        x = torch.cat([actions, slots], dim=-1)
        return x

    def decode(self, x):
        slots = x[:, :, self.action_dim:]
        actions = x[:, :, :self.action_dim]
        slots = self.decoder(slots)
         # [B, H, N, input_dim]
        x = torch.cat([actions, slots], dim=-1)
        return x

class TrajectoryToSlotWithExtraTokenEncoder(TrajectoryToSlotEncoderDecoder):
    def __init__(self, num_slots: int, 
                input_dim: int, # action_dim + per object obs_dim
                action_dim: int, # action dim
                extra_dim: int, # extra token dim
                slot_size: int = 128,
                alpha: float = 1.0, # positional encoding scaling, 0 for no positional encoding
                ):
        super().__init__(num_slots, slot_size, input_dim)
        self.encoder = TrajectoryToSlotEncoder(num_slots, 
                                               input_dim - action_dim,  
                                               slot_size, 
                                               alpha)
        self.decoder = TrajectoryToSlotDecoder(num_slots, 
                                               input_dim - action_dim, 
                                               slot_size)
        self.extra_encoder = TrajectoryToSlotEncoder(1, 
                                               extra_dim, 
                                               slot_size, 
                                               alpha)
        self.extra_decoder = TrajectoryToSlotDecoder(1, 
                                               extra_dim, 
                                               slot_size)
        self.extra_dim = extra_dim

    def encode(self, x):
        """
        sample: [B, horizon, input_dim + extra_dim]
        output: [B, horizon, input_dim + extra_dim]
        """
        # [B, H, extra_dim] -> [B, H, 1, extra_dim]
        extra_tokens = x[:, :, -self.extra_dim:]
        x = x[:, :, :-self.extra_dim]
        slots = x[:, :, self.action_dim:]
        actions = x[:, :, :self.action_dim]
        slots = self.encoder(slots)
        extra_tokens = self.extra_encoder(extra_tokens)
        slots = torch.cat([slots, extra_tokens], dim=2).flatten(2)
        # [B, H, N * (slot_size + act_dim)]
        x = torch.cat([actions, slots], dim=-1)
        return x
    
    def decode(self, x):
        slots = x[:, :, self.action_dim:]
        actions = x[:, :, :self.action_dim]
        extra_slots = slots[:, :, -1]
        slots = slots[:, :, :-1]
        # [B, H, N, slot_size] -> [B, H, N, D]
        slots = self.decoder(slots)
        extra_slots = self.extra_decoder(extra_slots)
        slots = torch.cat([slots.flatten(2), extra_slots.flatten(2)], dim=2)
        # [B, H, N, input_dim]
        x = torch.cat([actions, slots], dim=-1)
        return x

