"""
Core HSDE model implementing loss computation and optimization.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class HSDEModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    HSDE model combining VAE with geometric regularization and optional Neural SDE dynamics.
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
        use_sde=False,
        vae_reg=0.5,
        sde_reg=0.5,
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        sde_type: str = 'time_mlp',
        sde_time_cond: str = 'concat',
        sde_hidden_dim: Optional[int] = None,
        sde_solver_method: str = 'euler',
        sde_step_size: Optional[float] = None,
        sde_rtol: Optional[float] = None,
        sde_atol: Optional[float] = None,
        sde_sde_type: str = 'ito',
        sde_drift_reg: float = 0.01,
        sde_diffusion_reg: float = 0.01,
        sde_smoothness_reg: float = 0.01,
        ou_theta: float = 0.5,
        ou_sigma: float = 0.1,
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
        self.use_sde = use_sde
        self.vae_reg = vae_reg
        self.sde_reg = sde_reg
        self.device = device
        
        self.sde_drift_reg = sde_drift_reg
        self.sde_diffusion_reg = sde_diffusion_reg
        self.sde_smoothness_reg = sde_smoothness_reg
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        
        self.nn = VAE(
            state_dim,
            hidden_dim,
            latent_dim,
            i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_sde=use_sde,
            device=device,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len,
            sde_type=sde_type,
            sde_time_cond=sde_time_cond,
            sde_hidden_dim=sde_hidden_dim,
            sde_solver_method=sde_solver_method,
            sde_step_size=sde_step_size,
            sde_rtol=sde_rtol,
            sde_atol=sde_atol,
            sde_sde_type=sde_sde_type,
        )
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []
    
    @torch.no_grad()
    def take_latent(self, state):
        """Extract latent representation."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_sde:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            
            t_cpu = t.cpu().numpy()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t_cpu, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted, dtype=torch.float32)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0].unsqueeze(0)
            
            if hasattr(self.nn.sde_solver, 'reset_hidden'):
                self.nn.sde_solver.reset_hidden()
            
            q_z_sde = self.nn.solve_sde(
                self.nn.sde_solver, z0, t_sorted,
                method=self.nn.sde_solver_method,
                step_size=self.nn.sde_step_size,
                rtol=self.nn.sde_rtol,
                atol=self.nn.sde_atol,
                sde_type=self.nn.sde_sde_type,
            ).squeeze(1)
            q_z_sde = q_z_sde[sort_idxr]
            
            combined = self.vae_reg * q_z + self.sde_reg * q_z_sde
            return combined.cpu().numpy()
        else:
            q_z, _, _, _ = self.nn.encoder(state)
            return q_z.cpu().numpy()
    
    @torch.no_grad()
    def take_iembed(self, state):
        """Extract information bottleneck embedding."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_sde:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            
            t_cpu = t.cpu().numpy()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t_cpu, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted, dtype=torch.float32)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0].unsqueeze(0)
            
            if hasattr(self.nn.sde_solver, 'reset_hidden'):
                self.nn.sde_solver.reset_hidden()
            
            q_z_sde = self.nn.solve_sde(
                self.nn.sde_solver, z0, t_sorted,
                method=self.nn.sde_solver_method,
                step_size=self.nn.sde_step_size,
                rtol=self.nn.sde_rtol,
                atol=self.nn.sde_atol,
                sde_type=self.nn.sde_sde_type,
            ).squeeze(1)
            q_z_sde = q_z_sde[sort_idxr]
            
            le = self.nn.latent_encoder(q_z)
            le_sde = self.nn.latent_encoder(q_z_sde)
            
            combined = self.vae_reg * le + self.sde_reg * le_sde
            return combined.cpu().numpy()
        else:
            outputs = self.nn(state)
            le = outputs[4]
            return le.cpu().numpy()
    
    @torch.no_grad()
    def take_time(self, state):
        """Extract predicted pseudotime."""
        if not self.use_sde:
            raise ValueError("take_time() requires use_sde=True")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        _, _, _, _, t = self.nn.encoder(state)
        return t.cpu().numpy()
    
    @torch.no_grad()
    def take_grad(self, state):
        """Extract SDE drift field."""
        if not self.use_sde:
            raise ValueError("take_grad() requires use_sde=True")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        drift = self.nn.sde_solver.f(t, q_z)
        return drift.cpu().numpy()
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        """Compute cell-to-cell transition matrix from SDE dynamics."""
        if not self.use_sde:
            raise ValueError("take_transition() requires use_sde=True")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        
        drift = self.nn.sde_solver.f(t, q_z).cpu().numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * drift
        
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)
        
        def sparsify_transitions(trans_matrix, top_k):
            n_cells = trans_matrix.shape[0]
            sparse_trans = np.zeros_like(trans_matrix)
            for i in range(n_cells):
                top_indices = np.argsort(trans_matrix[i])[::-1][:top_k]
                sparse_trans[i, top_indices] = trans_matrix[i, top_indices]
                sparse_trans[i] /= sparse_trans[i].sum()
            return sparse_trans
        
        transition_matrix = sparsify_transitions(transition_matrix, top_k)
        return transition_matrix
    
    def _compute_reconstruction_loss(self, x_raw, pred_x, dropout_x):
        """Compute reconstruction loss with count-appropriate likelihood."""
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
    
    def _compute_sde_drift_regularization(self, q_z_sde: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Regularize drift to match OU process: f(t,z) ≈ -θ*z"""
        drift = self.nn.sde_solver.f(t, q_z_sde)
        prior_drift = -self.ou_theta * q_z_sde
        return torch.mean((drift - prior_drift) ** 2)
    
    def _compute_sde_diffusion_regularization(self, q_z_sde: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Regularize diffusion magnitude: g(t,z) ≈ σ"""
        diffusion = self.nn.sde_solver.g(t, q_z_sde)
        target_diffusion = self.ou_sigma * torch.ones_like(diffusion)
        return torch.mean((diffusion - target_diffusion) ** 2)
    
    def _compute_sde_trajectory_smoothness(self, q_z_sde: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Penalize trajectory acceleration (second derivative)."""
        if len(q_z_sde) <= 2:
            return torch.tensor(0.0, device=q_z_sde.device)
        
        dz = q_z_sde[1:] - q_z_sde[:-1]
        dt_diff = (t[1:] - t[:-1]).unsqueeze(-1)
        velocity = dz / (dt_diff + 1e-8)
        
        if len(velocity) <= 1:
            return torch.tensor(0.0, device=q_z_sde.device)
        
        acceleration = velocity[1:] - velocity[:-1]
        return torch.mean(acceleration ** 2)
    
    def _compute_sde_losses(
        self,
        q_z_sde: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all SDE regularization losses."""
        drift_loss = torch.tensor(0.0, device=q_z_sde.device)
        diffusion_loss = torch.tensor(0.0, device=q_z_sde.device)
        smoothness_loss = torch.tensor(0.0, device=q_z_sde.device)
        
        if self.sde_drift_reg > 0:
            drift_loss = self._compute_sde_drift_regularization(q_z_sde, t)
        
        if self.sde_diffusion_reg > 0:
            diffusion_loss = self._compute_sde_diffusion_regularization(q_z_sde, t)
        
        if self.sde_smoothness_reg > 0:
            smoothness_loss = self._compute_sde_trajectory_smoothness(q_z_sde, t)
        
        return drift_loss, diffusion_loss, smoothness_loss
    
    def update(self, states_norm, states_raw):
        """Perform one gradient descent step."""
        states_norm = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        states_raw = torch.tensor(states_raw, dtype=torch.float32).to(self.device)
        
        if torch.isnan(states_norm).any() or torch.isinf(states_norm).any():
            print("Warning: Invalid input data, skipping batch")
            return
        
        if self.use_sde:
            (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
             dropout_x, dropout_xl, q_z_sde, pred_x_sde, dropout_x_sde,
             x_sorted, t) = self.nn(states_norm)
            
            qz_div = F.mse_loss(q_z, q_z_sde, reduction="none").sum(-1).mean()
            
            recon_loss = self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x, dropout_x
            )
            recon_loss += self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x_sde, dropout_x_sde
            )
            
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    x_sorted, pred_xl, dropout_xl
                )
            
            sde_drift_loss, sde_diffusion_loss, sde_smoothness_loss = self._compute_sde_losses(
                q_z_sde, t
            )
            sde_drift_loss = self.sde_drift_reg * sde_drift_loss
            sde_diffusion_loss = self.sde_diffusion_reg * sde_diffusion_loss
            sde_smoothness_loss = self.sde_smoothness_reg * sde_smoothness_loss
            
        else:
            q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl = \
                self.nn(states_norm)
            
            qz_div = torch.tensor(0.0, device=self.device)
            
            recon_loss = self.recon * self._compute_reconstruction_loss(
                states_raw, pred_x, dropout_x
            )
            
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    states_raw, pred_xl, dropout_xl
                )
            
            sde_drift_loss = torch.tensor(0.0, device=self.device)
            sde_diffusion_loss = torch.tensor(0.0, device=self.device)
            sde_smoothness_loss = torch.tensor(0.0, device=self.device)
        
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
        
        if torch.isnan(q_m).any() or torch.isnan(q_s).any():
            print("Warning: NaN in encoder output, skipping batch")
            return
        
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()
        
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)
        
        total_loss = (
            recon_loss + irecon_loss + geometric_loss + qz_div + 
            kl_div + dip_loss + tc_loss + mmd_loss +
            sde_drift_loss + sde_diffusion_loss + sde_smoothness_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Invalid loss, skipping batch")
            return
        
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
        
        self.nn_optimizer.step()
        
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            geometric_loss.item(),
            qz_div.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item(),
            sde_drift_loss.item() if self.use_sde else 0.0,
            sde_diffusion_loss.item() if self.use_sde else 0.0,
            sde_smoothness_loss.item() if self.use_sde else 0.0
        ))