
"""
ODE function architectures for latent dynamics in Liora.

Available implementations:
- LatentODEfunc: Original time-invariant MLP (legacy, not recommended)
- TimeConditionedODE: Time-aware MLP with multiple conditioning strategies
- GRUODE: Recurrent dynamics with memory (recommended for smooth trajectories)
"""

import torch
import torch.nn as nn


def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class LatentODEfunc(nn.Module):
    """
    [LEGACY] Original time-invariant ODE function.
    
    ⚠️ WARNING: This implementation ignores the time parameter t.
    Use TimeConditionedODE instead for proper time-dependent dynamics.
    
    Parameters
    ----------
    n_latent : int, default=10
        Latent space dimensionality
    n_hidden : int, default=25
        Hidden layer size
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
        self.elu = nn.ELU()
        
        self.apply(weight_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent dynamics gradient (time-invariant).
        
        Parameters
        ----------
        t : torch.Tensor
            Time point (IGNORED in this implementation)
        x : torch.Tensor of shape (batch_size, n_latent)
            Current latent state
        
        Returns
        -------
        dz : torch.Tensor of shape (batch_size, n_latent)
            Time derivative dz/dt
        """
        h = self.fc1(x)
        h = self.elu(h)
        dz = self.fc2(h)
        return dz


class TimeConditionedODE(nn.Module):
    """
    Time-aware ODE function with multiple conditioning strategies.
    
    Fixes the time-invariance issue in LatentODEfunc by explicitly
    incorporating time information into the dynamics.
    
    Parameters
    ----------
    n_latent : int, default=10
        Latent space dimensionality
    n_hidden : int, default=25
        Hidden layer size
    time_cond : str, default='concat'
        Time conditioning strategy:
        - 'concat': Concatenate time to latent state [x; t]
        - 'film': Feature-wise Linear Modulation (scale and shift)
        - 'add': Additive time embedding
    
    Examples
    --------
    >>> ode_func = TimeConditionedODE(n_latent=10, n_hidden=32, time_cond='concat')
    >>> t = torch.tensor(0.5)
    >>> x = torch.randn(16, 10)
    >>> dz = ode_func(t, x)  # (16, 10)
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
        
        if time_cond == 'concat':
            # Concatenate time to latent state
            self.fc1 = nn.Linear(n_latent + 1, n_hidden)
        elif time_cond == 'film':
            # Feature-wise Linear Modulation
            self.fc1 = nn.Linear(n_latent, n_hidden)
            self.time_scale = nn.Linear(1, n_hidden)
            self.time_shift = nn.Linear(1, n_hidden)
        elif time_cond == 'add':
            # Additive time embedding
            self.fc1 = nn.Linear(n_latent, n_hidden)
            self.time_embed = nn.Linear(1, n_hidden)
        else:
            raise ValueError(f"Unknown time_cond: {time_cond}. Use 'concat', 'film', or 'add'")
        
        self.fc2 = nn.Linear(n_hidden, n_latent)
        self.elu = nn.ELU()
        
        self.apply(weight_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent latent dynamics gradient.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point (scalar or tensor)
        x : torch.Tensor of shape (batch_size, n_latent)
            Current latent state
        
        Returns
        -------
        dz : torch.Tensor of shape (batch_size, n_latent)
            Time derivative dz/dt
        """
        batch_size = x.shape[0]
        
        # Broadcast time to match batch dimension
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.view(-1, 1).expand(batch_size, 1)
        
        # Apply conditioning strategy
        if self.time_cond == 'concat':
            # Concatenate time to input
            h = torch.cat([x, t], dim=-1)
            h = self.fc1(h)
        elif self.time_cond == 'film':
            # Feature-wise modulation: scale * h + shift
            h = self.fc1(x)
            scale = self.time_scale(t)
            shift = self.time_shift(t)
            h = scale * h + shift
        else:  # 'add'
            # Additive time embedding
            h = self.fc1(x) + self.time_embed(t)
        
        h = self.elu(h)
        dz = self.fc2(h)
        return dz


class GRUODE(nn.Module):
    """
    GRU-based ODE function with recurrent memory.
    
    Maintains internal hidden state to model trajectory history,
    making it well-suited for smooth developmental processes.
    
    ⚠️ Important: Call `reset_hidden()` before solving each new trajectory.
    
    Parameters
    ----------
    n_latent : int, default=10
        Latent space dimensionality
    n_hidden : int, default=25
        GRU hidden state size
    
    Examples
    --------
    >>> ode_func = GRUODE(n_latent=10, n_hidden=32)
    >>> ode_func.reset_hidden()  # Reset before new trajectory
    >>> t = torch.tensor(0.5)
    >>> x = torch.randn(16, 10)
    >>> dz = ode_func(t, x)  # (16, 10)
    
    Notes
    -----
    The hidden state accumulates information across time steps during
    ODE solving, which helps model smooth trajectories but requires
    explicit reset between independent trajectories.
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
        # Time encoding network
        self.time_fc = nn.Linear(1, n_hidden)
        
        # GRU cell for maintaining trajectory memory
        self.gru_cell = nn.GRUCell(n_latent, n_hidden)
        
        # Output projection
        self.output_fc = nn.Linear(n_hidden, n_latent)
        
        # Hidden state (initialized per trajectory)
        self.hidden = None
        
        self.apply(weight_init)
    
    def reset_hidden(self):
        """
        Reset hidden state for a new trajectory.
        
        Call this before solving ODE for a new batch or trajectory.
        """
        self.hidden = None
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent dynamics with recurrent memory.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point (scalar or tensor)
        x : torch.Tensor of shape (batch_size, n_latent)
            Current latent state
        
        Returns
        -------
        dz : torch.Tensor of shape (batch_size, n_latent)
            Time derivative dz/dt
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
        
        # Compute derivative
        dz = self.output_fc(combined)
        
        return dz


def create_ode_func(ode_type: str, n_latent: int, n_hidden: int, **kwargs):
    """
    Factory function to create ODE function by type.
    
    Parameters
    ----------
    ode_type : str
        ODE function type: 'legacy', 'time_mlp', or 'gru'
    n_latent : int
        Latent space dimensionality
    n_hidden : int
        Hidden layer size
    **kwargs
        Additional arguments passed to specific ODE function
        (e.g., time_cond for TimeConditionedODE)
    
    Returns
    -------
    ode_func : nn.Module
        Instantiated ODE function
    
    Examples
    --------
    >>> ode_func = create_ode_func('time_mlp', n_latent=10, n_hidden=32, time_cond='concat')
    >>> ode_func = create_ode_func('gru', n_latent=10, n_hidden=32)
    """
    ode_type = ode_type.lower()
    
    if ode_type == 'legacy':
        return LatentODEfunc(n_latent, n_hidden)
    elif ode_type == 'time_mlp':
        time_cond = kwargs.get('time_cond', 'concat')
        return TimeConditionedODE(n_latent, n_hidden, time_cond)
    elif ode_type == 'gru':
        return GRUODE(n_latent, n_hidden)
    else:
        raise ValueError(
            f"Unknown ode_type: {ode_type}. "
            f"Use 'legacy', 'time_mlp', or 'gru'"
        )
