"""
SDE function architectures for latent dynamics in HSDE.

Available implementations:
- TimeConditionedSDE: Time-aware MLP with drift and diffusion
- GRUSDE: Recurrent dynamics with memory and stochastic noise

Key Design Pattern:
- f(t, x): Drift term μ(t, x) [required by torchsde]
- g(t, x): Diffusion term σ(t, x) [required by torchsde, must be > 0]
- forward(t, x): Internal implementation [called by f()]

This pattern allows both PyTorch (.forward) and torchsde (.f, .g) conventions.
"""

import torch
import torch.nn as nn


def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class TimeConditionedSDE(nn.Module):
    """
    Time-aware SDE with separate drift and diffusion networks.
    
    Solves: dz = μ(z,t) dt + σ(z,t) dB_t
    
    Required by torchsde.sdeint():
    - f(t, z) method for drift
    - g(t, z) method for diffusion
    """
    
    def __init__(
        self, 
        n_latent: int = 10, 
        n_hidden: int = 25,
        time_cond: str = 'concat'
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.time_cond = time_cond
        
        # ⚠️ REQUIRED by torchsde.sdeint()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        
        # Drift network
        if time_cond == 'concat':
            self.drift_fc1 = nn.Linear(n_latent + 1, n_hidden)
        elif time_cond == 'film':
            self.drift_fc1 = nn.Linear(n_latent, n_hidden)
            self.drift_time_scale = nn.Linear(1, n_hidden)
            self.drift_time_shift = nn.Linear(1, n_hidden)
        elif time_cond == 'add':
            self.drift_fc1 = nn.Linear(n_latent, n_hidden)
            self.drift_time_embed = nn.Linear(1, n_hidden)
        else:
            raise ValueError(f"Unknown time_cond: {time_cond}")
        
        self.drift_fc2 = nn.Linear(n_hidden, n_latent)
        self.drift_elu = nn.ELU()
        
        # Diffusion network
        if time_cond == 'concat':
            self.diffusion_fc1 = nn.Linear(n_latent + 1, n_hidden)
        elif time_cond == 'film':
            self.diffusion_fc1 = nn.Linear(n_latent, n_hidden)
            self.diffusion_time_scale = nn.Linear(1, n_hidden)
            self.diffusion_time_shift = nn.Linear(1, n_hidden)
        elif time_cond == 'add':
            self.diffusion_fc1 = nn.Linear(n_latent, n_hidden)
            self.diffusion_time_embed = nn.Linear(1, n_hidden)
        
        self.diffusion_fc2 = nn.Linear(n_hidden, n_latent)
        self.diffusion_elu = nn.ELU()
        
        self.apply(weight_init)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Drift term μ(z, t) - REQUIRED by torchsde.sdeint()
        
        torchsde looks for a method named 'f', not 'forward'
        """
        return self.forward(t, x)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DRIFT term: μ(z, t)
        
        Parameters
        ----------
        t : torch.Tensor
            Time point (scalar or 1D tensor)
        x : torch.Tensor of shape (batch_size, n_latent)
            Current latent state z_t
        
        Returns
        -------
        drift : torch.Tensor of shape (batch_size, n_latent)
            Time derivative dz/dt = μ(z_t, t)
        """
        batch_size = x.shape[0]
        
        # Broadcast time to match batch dimension
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.view(-1, 1).expand(batch_size, 1)
        
        # Apply conditioning strategy
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = self.drift_fc1(h)
        elif self.time_cond == 'film':
            h = self.drift_fc1(x)
            scale = self.drift_time_scale(t)
            shift = self.drift_time_shift(t)
            h = scale * h + shift
        else:  # 'add'
            h = self.drift_fc1(x) + self.drift_time_embed(t)
        
        h = self.drift_elu(h)
        drift = self.drift_fc2(h)
        return drift

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term σ(z, t) - REQUIRED by torchsde.sdeint()
        
        ⚠️ MUST RETURN POSITIVE VALUES!
        """
        batch_size = x.shape[0]
        
        # Broadcast time to match batch dimension
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.view(-1, 1).expand(batch_size, 1)
        
        # Apply conditioning strategy
        if self.time_cond == 'concat':
            h = torch.cat([x, t], dim=-1)
            h = self.diffusion_fc1(h)
        elif self.time_cond == 'film':
            h = self.diffusion_fc1(x)
            scale = self.diffusion_time_scale(t)
            shift = self.diffusion_time_shift(t)
            h = scale * h + shift
        else:  # 'add'
            h = self.diffusion_fc1(x) + self.diffusion_time_embed(t)
        
        h = self.diffusion_elu(h)
        diffusion_output = self.diffusion_fc2(h)
        
        # ⚠️ CRITICAL: softplus ensures σ > 0
        diffusion = torch.nn.functional.softplus(diffusion_output)
        
        return diffusion


class GRUSDE(nn.Module):
    """
    GRU-based SDE with recurrent memory and stochastic noise.
    
    Solves: dz = μ(z,t,h) dt + σ(z,t,h) dB_t
    
    Required by torchsde.sdeint():
    - f(t, z) method for drift
    - g(t, z) method for diffusion
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
        # ⚠️ REQUIRED by torchsde.sdeint()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        
        self.time_fc = nn.Linear(1, n_hidden)
        self.gru_cell = nn.GRUCell(n_latent, n_hidden)
        self.drift_fc = nn.Linear(n_hidden, n_latent)
        self.diffusion_fc = nn.Linear(n_hidden, n_latent)
        self.hidden = None
        
        self.apply(weight_init)
    
    def reset_hidden(self):
        """Reset hidden state for a new trajectory."""
        self.hidden = None
    
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Drift term μ(z, t, h) - REQUIRED by torchsde.sdeint()
        
        torchsde looks for a method named 'f', not 'forward'
        """
        return self.forward(t, x)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DRIFT term: μ(z, t, h)
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        x : torch.Tensor of shape (batch_size, n_latent)
            Current latent state z_t
        
        Returns
        -------
        drift : torch.Tensor of shape (batch_size, n_latent)
            Drift μ(z_t, t, h_t)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if needed
        if self.hidden is None or self.hidden.shape[0] != batch_size:
            self.hidden = torch.zeros(
                batch_size, self.n_hidden,
                device=x.device, dtype=x.dtype
            )
        
        # Update hidden state with current latent
        self.hidden = self.gru_cell(x, self.hidden)
        
        # Encode time information
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.view(-1, 1)
        time_info = torch.tanh(self.time_fc(t))
        
        # Modulate hidden state with time
        combined = self.hidden * time_info
        
        # Compute drift
        drift = self.drift_fc(combined)
        
        return drift

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term σ(z, t, h) - REQUIRED by torchsde.sdeint()
        
        ⚠️ MUST RETURN POSITIVE VALUES!
        """
        batch_size = x.shape[0]
        
        # Hidden state already updated by forward() call
        if self.hidden is None or self.hidden.shape[0] != batch_size:
            self.hidden = torch.zeros(
                batch_size, self.n_hidden,
                device=x.device, dtype=x.dtype
            )
        
        # Encode time information
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.view(-1, 1)
        time_info = torch.tanh(self.time_fc(t))
        
        # Modulate hidden state with time
        combined = self.hidden * time_info
        
        # Compute diffusion
        diffusion_output = self.diffusion_fc(combined)
        
        # ⚠️ CRITICAL: softplus ensures σ > 0
        diffusion = torch.nn.functional.softplus(diffusion_output)
        
        return diffusion


def create_sde_func(
    sde_type: str, 
    n_latent: int, 
    n_hidden: int,
    sde_calculus: str = 'ito',
    **kwargs
):
    """
    Factory function to create SDE function by type.
    
    Parameters
    ----------
    sde_type : str
        SDE function type: 'time_mlp' or 'gru'
    n_latent : int
        Latent space dimensionality
    n_hidden : int
        Hidden layer size
    sde_calculus : str, default='ito'
        SDE calculus type: 'ito' or 'stratonovich'
    **kwargs
        Additional arguments passed to specific SDE function
        (e.g., time_cond for TimeConditionedSDE)
    
    Returns
    -------
    sde_func : nn.Module
        Instantiated SDE function with f(), g(), noise_type, and sde_type
    
    Raises
    ------
    ValueError
        If sde_type is unknown
    """
    sde_type_lower = sde_type.lower()
    
    if sde_type_lower == 'time_mlp':
        time_cond = kwargs.get('time_cond', 'concat')
        sde_func = TimeConditionedSDE(n_latent, n_hidden, time_cond)
    elif sde_type_lower == 'gru':
        sde_func = GRUSDE(n_latent, n_hidden)
    else:
        raise ValueError(
            f"Unknown sde_type: {sde_type}. "
            f"Use 'time_mlp' or 'gru'"
        )
    
    # Set calculus type (may be used by solvers)
    sde_func.sde_type = sde_calculus
    
    return sde_func