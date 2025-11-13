
"""Core Liora model with loss computation and ODE support."""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class LioraModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Liora model with multiple regularization losses and optional ODE regularization.
    
    Combines standard VAE with:
    - Lorentz manifold regularization
    - Information bottleneck
    - Disentanglement losses (DIP, β-TC)
    - MMD regularization
    - Optional Neural ODE dynamics
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
        use_ode=False,
        vae_reg=0.5,
        ode_reg=0.5,
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
        self.use_ode = use_ode
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        
        self.nn = VAE(
            state_dim,
            hidden_dim,
            latent_dim,
            i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_ode=use_ode
        ).to(device)
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []
    
    @torch.no_grad()
    def take_latent(self, state):
        """Extract latent representation (combined VAE + ODE if applicable)."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            return (self.vae_reg * q_z + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            q_z, _, _, _ = self.nn.encoder(state)
            return q_z.cpu().numpy()
    
    @torch.no_grad()
    def take_iembed(self, state):
        """Extract information bottleneck embedding."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            
            le = self.nn.latent_encoder(q_z)
            le_ode = self.nn.latent_encoder(q_z_ode)
            return (self.vae_reg * le + self.ode_reg * le_ode).cpu().numpy()
        else:
            outputs = self.nn(state)
            le = outputs[4]  # Information bottleneck encoding
            return le.cpu().numpy()
    
    @torch.no_grad()
    def take_time(self, state):
        """Extract predicted time values (ODE mode only)."""
        if not self.use_ode:
            raise ValueError("take_time() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        _, _, _, _, t = self.nn.encoder(state)
        return t.cpu().numpy()
    
    @torch.no_grad()
    def take_grad(self, state):
        """Extract ODE gradients (ODE mode only)."""
        if not self.use_ode:
            raise ValueError("take_grad() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        """Compute transition matrix based on ODE dynamics."""
        if not self.use_ode:
            raise ValueError("take_transition() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        def sparsify_transitions(trans_matrix, top_k=top_k):
            n_cells = trans_matrix.shape[0]
            sparse_trans = np.zeros_like(trans_matrix)
            for i in range(n_cells):
                top_indices = np.argsort(trans_matrix[i])[::-1][:top_k]
                sparse_trans[i, top_indices] = trans_matrix[i, top_indices]
                sparse_trans[i] /= sparse_trans[i].sum()
            return sparse_trans

        transition_matrix = sparsify_transitions(transition_matrix)
        return transition_matrix
    
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
        
        # Forward pass (different unpacking for ODE vs non-ODE)
        if self.use_ode:
            (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
             dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode,
             x_sorted, t) = self.nn(states_norm)
            
            # ODE divergence loss (key ODE regularization)
            qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()
            
            # Reconstruction losses (both VAE and ODE paths)
            recon_loss = self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x, dropout_x
            )
            recon_loss += self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x_ode, dropout_x_ode
            )
            
            # Geometric regularization (only for non-ODE paths)
            geometric_loss = torch.tensor(0.0, device=self.device)
            if self.lorentz > 0:
                if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                    if self.use_euclidean_manifold:
                        from .utils import euclidean_distance
                        dist = euclidean_distance(z_manifold, ld_manifold)
                    else:
                        dist = lorentz_distance(z_manifold, ld_manifold)
                    
                    if not torch.isnan(dist).any():
                        geometric_loss = self.lorentz * dist.mean()
            
            # Information bottleneck reconstruction (both paths)
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    x_sorted, pred_xl, dropout_xl
                )
        
        else:
            # Non-ODE forward pass
            q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl = \
                self.nn(states_norm)
            
            qz_div = torch.tensor(0.0, device=self.device)
            
            # Reconstruction loss
            recon_loss = self.recon * self._compute_reconstruction_loss(
                states_raw, pred_x, dropout_x
            )
            
            # Geometric regularization
            geometric_loss = torch.tensor(0.0, device=self.device)
            if self.lorentz > 0:
                if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                    if self.use_euclidean_manifold:
                        from .utils import euclidean_distance
                        dist = euclidean_distance(z_manifold, ld_manifold)
                    else:
                        dist = lorentz_distance(z_manifold, ld_manifold)
                    
                    if not torch.isnan(dist).any():
                        geometric_loss = self.lorentz * dist.mean()
            
            # Information bottleneck reconstruction
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    states_raw, pred_xl, dropout_xl
                )
        
        if torch.isnan(q_m).any() or torch.isnan(q_s).any():
            print("Warning: NaN in encoder output, skipping batch")
            return
        
        # KL divergence (standard VAE)
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()
        
        # Additional regularizations
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)
        
        # Total loss (includes qz_div for ODE mode)
        total_loss = (
            recon_loss + irecon_loss + geometric_loss + qz_div + 
            kl_div + dip_loss + tc_loss + mmd_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss - skipping batch")
            return
        
        # Backpropagation
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
        
        self.nn_optimizer.step()
        
        # Log losses (now includes qz_div)
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            geometric_loss.item(),
            qz_div.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item()
        ))
