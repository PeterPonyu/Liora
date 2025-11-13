
"""Core LiVAE model with loss computation."""

import torch
import torch.optim as optim
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class LiVAEModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    LiVAE model with multiple regularization losses.
    
    Combines standard VAE with:
    - Lorentz manifold regularization
    - Information bottleneck
    - Disentanglement losses (DIP, β-TC)
    - MMD regularization
    """
    
    def __init__(
        self,
        recon,
        irecon,
        lorentz,
        beta,
        dip,
        tc,
        info,
        state_dim,
        hidden_dim,
        latent_dim,
        i_dim,
        lr,
        device,
        use_bottleneck_lorentz=True,
        loss_type='nb',
        grad_clip=1.0,
        use_layer_norm=True,
        use_euclidean_manifold=False,
        **kwargs
    ):
        self.recon = recon
        self.irecon = irecon
        self.lorentz = lorentz
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.use_euclidean_manifold = use_euclidean_manifold
        self.device = device
        
        self.nn = VAE(
            state_dim,
            hidden_dim,
            latent_dim,
            i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold
        ).to(device)
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []
    
    def take_latent(self, state):
        """Extract latent representation."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_z, _, _, _, _, _, _, _, _, _, _ = self.nn(state)
        return q_z.cpu().numpy()
    
    def _compute_reconstruction_loss(self, x_raw, pred_x, dropout_x):
        """Compute reconstruction loss based on likelihood type."""
        # Scale predictions by library size
        lib_size = torch.clamp(x_raw.sum(dim=-1, keepdim=True), min=1.0)
        pred_x = pred_x * lib_size
        
        if self.loss_type == 'nb':
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_nb(x_raw, pred_x, disp).sum(dim=-1).mean()
        elif self.loss_type == 'zinb':
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_zinb(x_raw, pred_x, disp, dropout_x).sum(dim=-1).mean()
        elif self.loss_type == 'poisson':
            return -self._log_poisson(x_raw, pred_x).sum(dim=-1).mean()
        elif self.loss_type == 'zip':
            return -self._log_zip(x_raw, pred_x, dropout_x).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
    def update(self, states_norm, states_raw):
        """Perform one gradient update."""
        states_norm = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        states_raw = torch.tensor(states_raw, dtype=torch.float32).to(self.device)
        
        # Validate inputs
        if torch.isnan(states_norm).any() or torch.isinf(states_norm).any():
            print("Warning: Invalid input data, skipping batch")
            return
        
        # Forward pass
        q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl = \
            self.nn(states_norm)
        
        if torch.isnan(q_m).any() or torch.isnan(q_s).any():
            print("Warning: NaN in encoder output, skipping batch")
            return
        
        # Reconstruction loss
        recon_loss = self.recon * self._compute_reconstruction_loss(states_raw, pred_x, dropout_x)
        
        # Geometric regularization (Lorentz OR Euclidean)    # MODIFY THIS BLOCK
        geometric_loss = torch.tensor(0.0, device=self.device)
        if self.lorentz > 0:
            if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                # Choose distance metric based on manifold type
                if self.use_euclidean_manifold:
                    from .utils import euclidean_distance
                    dist = euclidean_distance(z_manifold, ld_manifold)
                else:
                    from .utils import lorentz_distance
                    dist = lorentz_distance(z_manifold, ld_manifold)
                
                if not torch.isnan(dist).any():
                    geometric_loss = self.lorentz * dist.mean()
        
        # Information bottleneck reconstruction
        irecon_loss = torch.tensor(0.0, device=self.device)
        if self.irecon > 0:
            irecon_loss = self.irecon * self._compute_reconstruction_loss(
                states_raw, pred_xl, dropout_xl
            )
        
        # KL divergence (standard VAE)
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()
        
        # Additional regularizations
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = recon_loss + irecon_loss + geometric_loss + kl_div + dip_loss + tc_loss + mmd_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss - skipping batch")
            return
        
        # Backpropagation
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
        
        self.nn_optimizer.step()
        
        # Log losses
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            geometric_loss.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item()
        ))
